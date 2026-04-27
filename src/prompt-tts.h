#pragma once
// prompt-tts.h: build OmniVoice TTS prompt and CFG batch from
// (text, language, instruct, num_target_tokens). Mirrors the reference
// _prepare_inference_inputs and the cond+uncond stacking done in the
// reference iterative decoder.
//
// Output layout :
//   input_ids       [B', K, S_max] int32   B' = 2*B (cond rows then uncond)
//   audio_mask      [B', S_max]    int32   0 / 1
//   attention_mask  [B', S_max, S_max] int32   0 / 1
//
// Cond rows hold the full prompt (style + text + target audio mask slots).
// Uncond rows hold only the trailing target window, padded with audio_mask_id
// at the head and a diagonal True at the padding positions.

#include "bpe.h"
#include "lang-map.h"
#include "omnivoice-llm.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

struct PromptTTS {
    int B;                                // logical batch size (always 1 in this version)
    int B_prime;                          // 2 * B
    int K;                                // num_audio_codebook
    int S_max;                            // padded sequence length (= c_len for B=1)
    int c_len;                            // cond effective length
    int u_len;                            // uncond effective length (= num_target_tokens)

    std::vector<int32_t> input_ids;       // [B', K, S_max]
    std::vector<int32_t> audio_mask;      // [B', S_max]
    std::vector<int32_t> attention_mask;  // [B', S_max, S_max]
};

// 13 non-verbal tags from the reference NONVERBAL_PATTERN.
static const char * const PROMPT_NONVERBAL_TAGS[] = {
    "[laughter]",    "[sigh]",        "[confirmation-en]",     "[question-en]", "[question-ah]",
    "[question-oh]", "[question-ei]", "[question-yi]",         "[surprise-ah]", "[surprise-oh]",
    "[surprise-wa]", "[surprise-yo]", "[dissatisfaction-hnn]",
};
static const int PROMPT_NONVERBAL_N = 13;

// Decode one UTF-8 sequence at p, write the codepoint to *cp, return the
// byte count (0 on malformed).
static int prompt_tts_utf8_decode(const char * p, size_t avail, uint32_t * cp) {
    if (avail == 0) {
        return 0;
    }
    uint8_t c = (uint8_t) p[0];
    if (c < 0x80) {
        *cp = c;
        return 1;
    }
    if ((c & 0xE0) == 0xC0 && avail >= 2) {
        *cp = ((uint32_t) (c & 0x1F) << 6) | ((uint32_t) (uint8_t) p[1] & 0x3F);
        return 2;
    }
    if ((c & 0xF0) == 0xE0 && avail >= 3) {
        *cp = ((uint32_t) (c & 0x0F) << 12) | ((uint32_t) ((uint8_t) p[1] & 0x3F) << 6) |
              ((uint32_t) (uint8_t) p[2] & 0x3F);
        return 3;
    }
    if ((c & 0xF8) == 0xF0 && avail >= 4) {
        *cp = ((uint32_t) (c & 0x07) << 18) | ((uint32_t) ((uint8_t) p[1] & 0x3F) << 12) |
              ((uint32_t) ((uint8_t) p[2] & 0x3F) << 6) | ((uint32_t) (uint8_t) p[3] & 0x3F);
        return 4;
    }
    return 0;
}

// Append a codepoint as UTF-8 bytes to out.
static void prompt_tts_utf8_append(std::string & out, uint32_t cp) {
    if (cp < 0x80) {
        out.push_back((char) cp);
    } else if (cp < 0x800) {
        out.push_back((char) (0xC0 | (cp >> 6)));
        out.push_back((char) (0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back((char) (0xE0 | (cp >> 12)));
        out.push_back((char) (0x80 | ((cp >> 6) & 0x3F)));
        out.push_back((char) (0x80 | (cp & 0x3F)));
    } else {
        out.push_back((char) (0xF0 | (cp >> 18)));
        out.push_back((char) (0x80 | ((cp >> 12) & 0x3F)));
        out.push_back((char) (0x80 | ((cp >> 6) & 0x3F)));
        out.push_back((char) (0x80 | (cp & 0x3F)));
    }
}

// CJK Unified Ideographs main range, matching the [\u4e00-\u9fff] regex used
// in the reference _combine_text.
static bool prompt_tts_is_cjk(uint32_t cp) {
    return cp >= 0x4E00 && cp <= 0x9FFF;
}

// _combine_text : five steps, pixel perfect with the reference utility.
//   1. concat ref_text + " " + text after stripping each
//   2. drop CR / LF
//   3. replace full-width parens U+FF08 / U+FF09 with ASCII ( / )
//   4. collapse runs of space and tab into a single space
//   5. drop any space adjacent to a CJK ideograph U+4E00..U+9FFF
static std::string prompt_tts_combine_text(const std::string & text, const std::string & ref_text) {
    auto strip = [](const std::string & s) {
        size_t a = 0, b = s.size();
        while (a < b && (s[a] == ' ' || s[a] == '\t')) {
            a++;
        }
        while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t')) {
            b--;
        }
        return s.substr(a, b - a);
    };

    std::string raw = ref_text.empty() ? strip(text) : (strip(ref_text) + " " + strip(text));

    // Decode to codepoints, apply step 2 (drop CR / LF) and step 3 (replace
    // full-width parens) inline.
    std::vector<uint32_t> cps;
    cps.reserve(raw.size());
    size_t i = 0;
    while (i < raw.size()) {
        uint32_t cp = 0;
        int      n  = prompt_tts_utf8_decode(raw.data() + i, raw.size() - i, &cp);
        if (n == 0) {
            i++;
            continue;
        }
        i += n;
        if (cp == '\r' || cp == '\n') {
            continue;
        }
        if (cp == 0xFF08) {
            cp = '(';
        } else if (cp == 0xFF09) {
            cp = ')';
        }
        cps.push_back(cp);
    }

    // Step 4 : collapse space / tab runs into one space.
    std::vector<uint32_t> collapsed;
    collapsed.reserve(cps.size());
    bool in_space = false;
    for (uint32_t cp : cps) {
        bool is_ws = (cp == ' ' || cp == '\t');
        if (is_ws) {
            if (!in_space) {
                collapsed.push_back(' ');
                in_space = true;
            }
        } else {
            collapsed.push_back(cp);
            in_space = false;
        }
    }

    // Step 5 : drop spaces adjacent to CJK ideographs.
    std::string out;
    out.reserve(raw.size());
    for (size_t j = 0; j < collapsed.size(); j++) {
        if (collapsed[j] == ' ') {
            bool prev_cjk = (j > 0 && prompt_tts_is_cjk(collapsed[j - 1]));
            bool next_cjk = (j + 1 < collapsed.size() && prompt_tts_is_cjk(collapsed[j + 1]));
            if (prev_cjk || next_cjk) {
                continue;
            }
        }
        prompt_tts_utf8_append(out, collapsed[j]);
    }
    return out;
}

// Tokenize a text segment that may contain non-verbal tags. Each tag is
// tokenized independently from the surrounding context so its id stream
// is stable across languages.
static std::vector<int> prompt_tts_tokenize_nonverbal(const BPETokenizer * tok, const std::string & text) {
    std::vector<int> ids;
    size_t           pos = 0;
    while (pos < text.size()) {
        size_t best_pos = std::string::npos;
        int    best_idx = -1;
        for (int i = 0; i < PROMPT_NONVERBAL_N; i++) {
            const std::string tag = PROMPT_NONVERBAL_TAGS[i];
            size_t            p   = text.find(tag, pos);
            if (p != std::string::npos && p < best_pos) {
                best_pos = p;
                best_idx = i;
            }
        }
        if (best_idx < 0) {
            std::vector<int> seg = bpe_encode(tok, text.substr(pos), /*add_eos=*/false);
            ids.insert(ids.end(), seg.begin(), seg.end());
            break;
        }
        if (best_pos > pos) {
            std::vector<int> seg = bpe_encode(tok, text.substr(pos, best_pos - pos), /*add_eos=*/false);
            ids.insert(ids.end(), seg.begin(), seg.end());
        }
        std::string      tag     = PROMPT_NONVERBAL_TAGS[best_idx];
        std::vector<int> tag_ids = bpe_encode(tok, tag, /*add_eos=*/false);
        ids.insert(ids.end(), tag_ids.begin(), tag_ids.end());
        pos = best_pos + tag.size();
    }
    return ids;
}

// Build prompt + CFG batch. ref_audio_tokens may be NULL (TTS pure path) or a
// pointer to ref_T audio frames laid out [K, ref_T] row-major (k slow, t fast).
// ref_text is concatenated with text via _combine_text and contributes to the
// shared text segment. The <|denoise|> marker is emitted if and only if
// denoise=true AND ref_audio_tokens is not NULL (mirrors the reference).
static bool prompt_tts_build(PromptTTS *          out,
                             const BPETokenizer * tok,
                             const OmniVoiceLM *  lm,
                             const std::string &  text,
                             const std::string &  lang,
                             const std::string &  instruct,
                             int                  num_target_tokens,
                             bool                 denoise,
                             const std::string &  ref_text,
                             const int32_t *      ref_audio_tokens,
                             int                  ref_T) {
    if (num_target_tokens <= 0) {
        fprintf(stderr, "[Prompt] FATAL: num_target_tokens must be positive (got %d)\n", num_target_tokens);
        return false;
    }
    if (ref_audio_tokens != NULL && ref_T <= 0) {
        fprintf(stderr, "[Prompt] FATAL: ref_audio_tokens != NULL requires ref_T > 0 (got %d)\n", ref_T);
        return false;
    }
    *out         = {};
    out->B       = 1;
    out->B_prime = 2;
    out->K       = lm->num_audio_codebook;

    std::string style_text;
    if (denoise && ref_audio_tokens != NULL) {
        style_text += "<|denoise|>";
    }

    // Mirror Python _resolve_language: pass-through valid ISO IDs, lowercase
    // and look up names like "French" -> "fr". Empty result falls back to
    // "None" so the special tokens still wrap a single placeholder token.
    std::string lang_resolved = resolve_language(lang);
    std::string lang_str      = lang_resolved.empty() ? "None" : lang_resolved;
    std::string instruct_str  = instruct.empty() ? "None" : instruct;
    style_text += "<|lang_start|>" + lang_str + "<|lang_end|>";
    style_text += "<|instruct_start|>" + instruct_str + "<|instruct_end|>";

    std::vector<int> style_ids = bpe_encode(tok, style_text, /*add_eos=*/false);

    std::string      full_text = prompt_tts_combine_text(text, ref_text);
    std::string      wrapped   = "<|text_start|>" + full_text + "<|text_end|>";
    std::vector<int> text_ids  = prompt_tts_tokenize_nonverbal(tok, wrapped);

    const int N1      = (int) style_ids.size();
    const int N2      = (int) text_ids.size();
    const int Stgt    = num_target_tokens;
    const int Sref    = (ref_audio_tokens != NULL) ? ref_T : 0;
    const int c_len   = N1 + N2 + Sref + Stgt;
    const int u_len   = Stgt;
    const int K       = out->K;
    const int mask_id = lm->audio_mask_id;

    out->c_len = c_len;
    out->u_len = u_len;
    out->S_max = c_len;

    out->input_ids.assign((size_t) out->B_prime * (size_t) K * (size_t) c_len, mask_id);
    out->audio_mask.assign((size_t) out->B_prime * (size_t) c_len, 0);
    out->attention_mask.assign((size_t) out->B_prime * (size_t) c_len * (size_t) c_len, 0);

    auto cond_at = [&](int k, int s) -> int32_t & {
        return out->input_ids[((size_t) 0 * K + (size_t) k) * (size_t) c_len + (size_t) s];
    };
    auto uncond_at = [&](int k, int s) -> int32_t & {
        return out->input_ids[((size_t) 1 * K + (size_t) k) * (size_t) c_len + (size_t) s];
    };

    // Style + text tokens are duplicated across all K codebooks (the reference
    // does .repeat(num_audio_codebook, 1)).
    for (int k = 0; k < K; k++) {
        for (int n = 0; n < N1; n++) {
            cond_at(k, n) = (int32_t) style_ids[n];
        }
        for (int n = 0; n < N2; n++) {
            cond_at(k, N1 + n) = (int32_t) text_ids[n];
        }
    }

    // Reference audio tokens occupy [N1+N2, N1+N2+Sref) on the cond row, with
    // the actual codebook values per k. The trailing [c_len-Stgt, c_len) keeps
    // mask_id since the assign initialised the buffer to mask_id everywhere.
    const int ref_start   = N1 + N2;
    const int audio_start = ref_start;  // audio_mask covers ref + target
    if (Sref > 0) {
        for (int k = 0; k < K; k++) {
            for (int t = 0; t < Sref; t++) {
                cond_at(k, ref_start + t) = ref_audio_tokens[(size_t) k * (size_t) Sref + (size_t) t];
            }
        }
    }

    // Cond audio_mask covers ref_audio + target window, all in one block.
    for (int s = audio_start; s < c_len; s++) {
        out->audio_mask[(size_t) 0 * (size_t) c_len + (size_t) s] = 1;
    }

    // Uncond row holds only the trailing target window (Stgt mask slots),
    // mirroring the reference batch_input_ids[B+i, :, :u_len] = input_ids[..., -u_len:].
    for (int k = 0; k < K; k++) {
        for (int s = 0; s < u_len; s++) {
            uncond_at(k, s) = cond_at(k, c_len - Stgt + s);
        }
    }
    for (int s = 0; s < u_len; s++) {
        out->audio_mask[(size_t) 1 * (size_t) c_len + (size_t) s] = 1;
    }

    auto attn_at = [&](int b, int sq, int skv) -> int32_t & {
        return out->attention_mask[((size_t) b * (size_t) c_len + (size_t) sq) * (size_t) c_len + (size_t) skv];
    };

    // Cond row : full bidirectional attention on the c_len window.
    for (int sq = 0; sq < c_len; sq++) {
        for (int skv = 0; skv < c_len; skv++) {
            attn_at(0, sq, skv) = 1;
        }
    }

    // Uncond row : full bidirectional inside the u_len target window, diagonal
    // pads on the trailing positions [u_len, c_len).
    for (int sq = 0; sq < u_len; sq++) {
        for (int skv = 0; skv < u_len; skv++) {
            attn_at(1, sq, skv) = 1;
        }
    }
    for (int sq = u_len; sq < c_len; sq++) {
        attn_at(1, sq, sq) = 1;
    }

    fprintf(stderr, "[Prompt] Built: B'=%d K=%d S=%d N1=%d N2=%d Sref=%d Stgt=%d c_len=%d u_len=%d denoise=%d\n",
            out->B_prime, K, out->S_max, N1, N2, Sref, Stgt, c_len, u_len, (int) (denoise && Sref > 0));
    return true;
}
