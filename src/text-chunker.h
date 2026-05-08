#pragma once
// text-chunker.h: long-form text splitter for OmniVoice TTS
//
// chunk_text_punctuation: 1:1 port of omnivoice/utils/text.py.
// Splits text on sentence-ending punctuation (skipping abbreviation periods),
// then merges sentences into chunks of at most chunk_len UTF-8 codepoints.
// Optional min_chunk_len merges undersized chunks into a neighbour.
// Strings are UTF-8 in, UTF-8 out. Comparison and length are codepoint-based,
// matching Python str semantics.

#include <set>
#include <string>
#include <vector>

// Returns the byte length of the UTF-8 codepoint starting at b (1, 2, 3 or 4).
// Falls back to 1 on invalid first bytes so iteration always advances.
static inline int chunker_utf8_len(unsigned char b) {
    if ((b & 0x80) == 0x00) {
        return 1;
    }

    if ((b & 0xE0) == 0xC0) {
        return 2;
    }

    if ((b & 0xF0) == 0xE0) {
        return 3;
    }

    if ((b & 0xF8) == 0xF0) {
        return 4;
    }

    return 1;
}

// Sentence-ending punctuation. Mirrors SPLIT_PUNCTUATION in text.py.
static const std::set<std::string> & chunker_split_punctuation() {
    static const std::set<std::string> s = {
        ".",
        ",",
        ";",
        ":",
        "!",
        "?",
        "\xe3\x80\x82",  // U+3002 ideographic full stop
        "\xef\xbc\x8c",  // U+FF0C fullwidth comma
        "\xef\xbc\x9b",  // U+FF1B fullwidth semicolon
        "\xef\xbc\x9a",  // U+FF1A fullwidth colon
        "\xef\xbc\x81",  // U+FF01 fullwidth exclamation mark
        "\xef\xbc\x9f",  // U+FF1F fullwidth question mark
    };
    return s;
}

// Closing marks attach to the preceding sentence. Mirrors CLOSING_MARKS.
static const std::set<std::string> & chunker_closing_marks() {
    static const std::set<std::string> s = {
        "\"",           "'", "]", ">",
        "\xe2\x80\x9c",  // U+201C left double quotation mark
        "\xe2\x80\x9d",  // U+201D right double quotation mark
        "\xe2\x80\x98",  // U+2018 left single quotation mark
        "\xe2\x80\x99",  // U+2019 right single quotation mark
        "\xef\xbc\x89",  // U+FF09 fullwidth right parenthesis
        "\xe3\x80\x8b",  // U+300B right double angle bracket
        "\xe3\x80\x8d",  // U+300D right corner bracket
        "\xe3\x80\x91",  // U+3011 right black lenticular bracket
    };
    return s;
}

// Abbreviations that suppress the period as a sentence break. ASCII only,
// matched on the last whitespace-delimited word ending with the period.
// Mirrors ABBREVIATIONS in text.py.
static const std::set<std::string> & chunker_abbreviations() {
    static const std::set<std::string> s = {
        "Mr.",  "Mrs.",  "Ms.",  "Dr.",  "Prof.", "Sr.",   "Jr.",     "Rev.", "Fr.",   "Hon.", "Pres.",
        "Gov.", "Capt.", "Gen.", "Sen.", "Rep.",  "Col.",  "Maj.",    "Lt.",  "Cmdr.", "Sgt.", "Cpl.",
        "Co.",  "Corp.", "Inc.", "Ltd.", "Est.",  "Dept.", "St.",     "Ave.", "Blvd.", "Rd.",  "Mt.",
        "Ft.",  "No.",   "Jan.", "Feb.", "Mar.",  "Apr.",  "Aug.",    "Sep.", "Sept.", "Oct.", "Nov.",
        "Dec.", "i.e.",  "e.g.", "vs.",  "Vs.",   "Etc.",  "approx.", "fig.", "def.",
    };
    return s;
}

// Returns true if cp (a UTF-8 codepoint string) is whitespace per Python's
// str.split() / str.strip() definition : ASCII (\t \n \v \f \r space) plus
// the Unicode whitespace block. Mirrors str.isspace() semantics on the
// codepoints produced by tokenising text into characters.
static bool chunker_is_unicode_whitespace(const std::string & cp) {
    if (cp.size() == 1) {
        unsigned char b = (unsigned char) cp[0];
        return b == ' ' || b == '\t' || b == '\n' || b == '\r' || b == '\v' || b == '\f';
    }

    if (cp.size() == 2) {
        // U+0085 NEL (C2 85), U+00A0 NBSP (C2 A0).
        if ((unsigned char) cp[0] == 0xC2) {
            unsigned char b1 = (unsigned char) cp[1];
            return b1 == 0x85 || b1 == 0xA0;
        }
        return false;
    }

    if (cp.size() == 3) {
        unsigned char b0 = (unsigned char) cp[0];
        unsigned char b1 = (unsigned char) cp[1];
        unsigned char b2 = (unsigned char) cp[2];
        uint32_t      u  = ((uint32_t) (b0 & 0x0F) << 12) | ((uint32_t) (b1 & 0x3F) << 6) | ((uint32_t) (b2 & 0x3F));

        // U+1680 OGHAM SPACE MARK
        if (u == 0x1680) {
            return true;
        }
        // U+2000..U+200A en quad/em/thin/hair spaces
        if (u >= 0x2000 && u <= 0x200A) {
            return true;
        }
        // U+2028 LINE SEP, U+2029 PARAGRAPH SEP, U+202F NARROW NBSP
        if (u == 0x2028 || u == 0x2029 || u == 0x202F) {
            return true;
        }
        // U+205F MEDIUM MATHEMATICAL SPACE
        if (u == 0x205F) {
            return true;
        }
        // U+3000 IDEOGRAPHIC SPACE
        if (u == 0x3000) {
            return true;
        }
        // U+FEFF ZERO WIDTH NO-BREAK SPACE (BOM)
        if (u == 0xFEFF) {
            return true;
        }
        return false;
    }

    return false;
}

// Returns the last whitespace-delimited word of s, or s itself if no
// whitespace. Used to detect abbreviation periods. Matches Python
// str.split()[-1] semantics : whitespace is the Unicode whitespace block,
// not just ASCII, so a non-breaking space before "Mr." still detects the
// abbreviation.
static std::string chunker_last_word(const std::string & s) {
    // Walk codepoints, find the byte index right after the last whitespace
    // run that is followed by at least one non-whitespace codepoint.
    const unsigned char * p   = (const unsigned char *) s.data();
    const unsigned char * end = p + s.size();

    size_t last_word_byte_start = 0;
    bool   prev_was_ws          = true;
    bool   any_non_ws           = false;
    size_t trailing_ws_start_at = s.size();

    while (p < end) {
        int n = chunker_utf8_len(*p);
        if (p + n > end) {
            n = (int) (end - p);
        }
        std::string cp((const char *) p, (size_t) n);
        size_t      byte_pos = (size_t) (p - (const unsigned char *) s.data());

        bool is_ws = chunker_is_unicode_whitespace(cp);
        if (is_ws) {
            if (!prev_was_ws) {
                trailing_ws_start_at = byte_pos;
            }
        } else {
            if (prev_was_ws) {
                last_word_byte_start = byte_pos;
            }
            any_non_ws           = true;
            trailing_ws_start_at = byte_pos + (size_t) n;
        }
        prev_was_ws = is_ws;
        p += n;
    }

    if (!any_non_ws) {
        return std::string();
    }
    return s.substr(last_word_byte_start, trailing_ws_start_at - last_word_byte_start);
}

// Strips leading and trailing Unicode whitespace from s. Matches Python
// str.strip() : ASCII whitespace plus NBSP, ideographic space, thin
// spaces, BOM and the rest of the Unicode whitespace block.
static std::string chunker_strip(const std::string & s) {
    const unsigned char * p   = (const unsigned char *) s.data();
    const unsigned char * end = p + s.size();

    // Walk forward to find first non whitespace codepoint.
    size_t start_byte = 0;
    while (p < end) {
        int n = chunker_utf8_len(*p);
        if (p + n > end) {
            n = (int) (end - p);
        }
        std::string cp((const char *) p, (size_t) n);
        if (!chunker_is_unicode_whitespace(cp)) {
            break;
        }
        p += n;
        start_byte += (size_t) n;
    }

    if (p >= end) {
        return std::string();
    }

    // Walk forward through the remainder, tracking the byte index after the
    // last non whitespace codepoint.
    size_t after_last_non_ws = start_byte;
    while (p < end) {
        int n = chunker_utf8_len(*p);
        if (p + n > end) {
            n = (int) (end - p);
        }
        std::string cp((const char *) p, (size_t) n);
        size_t      pos = (size_t) (p - (const unsigned char *) s.data());
        if (!chunker_is_unicode_whitespace(cp)) {
            after_last_non_ws = pos + (size_t) n;
        }
        p += n;
    }

    return s.substr(start_byte, after_last_non_ws - start_byte);
}

// Splits text on sentence-ending punctuation (skipping abbreviations) and
// merges sentences into chunks of at most chunk_len codepoints. If
// min_chunk_len > 0, undersized chunks are merged with a neighbour.
// Returns a list of stripped chunk strings (UTF-8). Empty chunks are dropped.
//
// Strict 1:1 port of chunk_text_punctuation in omnivoice/utils/text.py.
static std::vector<std::string> chunk_text_punctuation(const std::string & text, int chunk_len, int min_chunk_len) {
    // Step 1: tokenise into UTF-8 codepoints, then split on punctuation.
    // sentences holds vectors of codepoints (each codepoint is a std::string).
    std::vector<std::vector<std::string>> sentences;
    std::vector<std::string>              current;

    const std::set<std::string> & split_set   = chunker_split_punctuation();
    const std::set<std::string> & closing_set = chunker_closing_marks();
    const std::set<std::string> & abbrev_set  = chunker_abbreviations();

    const unsigned char * p   = (const unsigned char *) text.data();
    const unsigned char * end = p + text.size();

    while (p < end) {
        int n = chunker_utf8_len(*p);
        if (p + n > end) {
            n = (int) (end - p);
        }

        std::string cp((const char *) p, (size_t) n);
        p += n;

        bool is_split   = split_set.count(cp) > 0;
        bool is_closing = closing_set.count(cp) > 0;

        // Leading punctuation glues onto the previous sentence.
        if (current.empty() && !sentences.empty() && (is_split || is_closing)) {
            sentences.back().push_back(cp);
            continue;
        }

        current.push_back(cp);

        if (!is_split) {
            continue;
        }

        // Period after an abbreviation does not break the sentence.
        bool is_abbreviation = false;
        if (cp == ".") {
            std::string joined;
            for (const auto & c : current) {
                joined += c;
            }

            std::string last = chunker_last_word(joined);
            if (!last.empty() && abbrev_set.count(last) > 0) {
                is_abbreviation = true;
            }
        }

        if (!is_abbreviation) {
            sentences.push_back(current);
            current.clear();
        }
    }

    if (!current.empty()) {
        sentences.push_back(current);
    }

    // Step 2: greedy merge of sentences into chunks of at most chunk_len
    // codepoints. A sentence that does not fit starts a new chunk by itself,
    // even if it is longer than chunk_len.
    std::vector<std::vector<std::string>> merged;
    std::vector<std::string>              cur_chunk;

    for (const auto & sent : sentences) {
        if ((int) (cur_chunk.size() + sent.size()) <= chunk_len) {
            for (const auto & c : sent) {
                cur_chunk.push_back(c);
            }
        } else {
            if (!cur_chunk.empty()) {
                merged.push_back(cur_chunk);
            }

            cur_chunk = sent;
        }
    }

    if (!cur_chunk.empty()) {
        merged.push_back(cur_chunk);
    }

    // Step 3: merge undersized chunks. The first chunk, if short, is folded
    // into the second. Subsequent short chunks fold into the previous one.
    std::vector<std::vector<std::string>> finals;
    if (min_chunk_len > 0) {
        bool first_short = !merged.empty() && (int) merged[0].size() < min_chunk_len;

        for (size_t i = 0; i < merged.size(); i++) {
            const auto & chunk = merged[i];

            if (i == 1 && first_short) {
                for (const auto & c : chunk) {
                    finals.back().push_back(c);
                }
                continue;
            }

            if ((int) chunk.size() >= min_chunk_len) {
                finals.push_back(chunk);
                continue;
            }

            if (finals.empty()) {
                finals.push_back(chunk);
            } else {
                for (const auto & c : chunk) {
                    finals.back().push_back(c);
                }
            }
        }
    } else {
        finals = merged;
    }

    // Step 4: join codepoints, strip whitespace, drop empty.
    std::vector<std::string> result;
    result.reserve(finals.size());

    for (const auto & chunk : finals) {
        std::string joined;
        for (const auto & c : chunk) {
            joined += c;
        }

        std::string stripped = chunker_strip(joined);
        if (!stripped.empty()) {
            result.push_back(stripped);
        }
    }

    return result;
}

// Counts UTF-8 codepoints in s. Used to derive the per-chunk character budget
// from the average tokens-per-character of the full text, matching Python's
// len(text) which counts codepoints.
static int chunker_utf8_count(const std::string & text) {
    const unsigned char * p   = (const unsigned char *) text.data();
    const unsigned char * end = p + text.size();
    int                   n   = 0;

    while (p < end) {
        int s = chunker_utf8_len(*p);
        if (p + s > end) {
            s = (int) (end - p);
        }

        p += s;
        n += 1;
    }

    return n;
}

// Punctuation considered "terminal" by add_punctuation. Mirrors END_PUNCTUATION
// in text.py. ASCII first, then UTF-8 byte sequences for fancy quotes, ellipsis
// and Chinese variants.
static const std::set<std::string> & chunker_end_punctuation() {
    static const std::set<std::string> s = {
        ";",
        ":",
        ",",
        ".",
        "!",
        "?",
        ")",
        "]",
        "}",
        "\"",
        "'",
        "\xe2\x80\xa6",  // U+2026 horizontal ellipsis
        "\xe2\x80\x9c",  // U+201C left double quotation mark
        "\xe2\x80\x9d",  // U+201D right double quotation mark
        "\xe2\x80\x98",  // U+2018 left single quotation mark
        "\xe2\x80\x99",  // U+2019 right single quotation mark
        "\xef\xbc\x9b",  // U+FF1B fullwidth semicolon
        "\xef\xbc\x9a",  // U+FF1A fullwidth colon
        "\xef\xbc\x8c",  // U+FF0C fullwidth comma
        "\xe3\x80\x82",  // U+3002 ideographic full stop
        "\xef\xbc\x81",  // U+FF01 fullwidth exclamation mark
        "\xef\xbc\x9f",  // U+FF1F fullwidth question mark
        "\xe3\x80\x81",  // U+3001 ideographic comma
        "\xef\xbc\x89",  // U+FF09 fullwidth right parenthesis
        "\xe3\x80\x91",  // U+3011 right black lenticular bracket
    };
    return s;
}

// Returns the last UTF-8 codepoint of s as a std::string, or empty if s is
// empty. Walks the byte sequence to find the start of the last codepoint.
static std::string chunker_last_codepoint(const std::string & s) {
    if (s.empty()) {
        return std::string();
    }

    size_t i = s.size();
    while (i > 0) {
        unsigned char b = (unsigned char) s[i - 1];
        if ((b & 0xC0) != 0x80) {
            return s.substr(i - 1);
        }

        i--;
    }

    return s;
}

// Returns true if any codepoint of s falls inside the CJK Unified Ideographs
// block (U+4E00..U+9FFF). Mirrors the Chinese-detection heuristic in
// add_punctuation upstream.
static bool chunker_contains_chinese(const std::string & s) {
    const unsigned char * p   = (const unsigned char *) s.data();
    const unsigned char * end = p + s.size();

    while (p < end) {
        int n = chunker_utf8_len(*p);
        if (p + n > end) {
            return false;
        }

        if (n == 3) {
            uint32_t cp =
                ((uint32_t) (p[0] & 0x0F) << 12) | ((uint32_t) (p[1] & 0x3F) << 6) | ((uint32_t) (p[2] & 0x3F));
            if (cp >= 0x4E00 && cp <= 0x9FFF) {
                return true;
            }
        }

        p += n;
    }

    return false;
}

// Strips text and appends a terminal punctuation if missing. Mirrors
// add_punctuation in omnivoice/utils/text.py: appends "." for non-Chinese
// text, and the ideographic full stop "。" for text containing CJK.
static std::string add_punctuation(const std::string & text) {
    std::string s = chunker_strip(text);
    if (s.empty()) {
        return s;
    }

    std::string  last    = chunker_last_codepoint(s);
    const auto & end_set = chunker_end_punctuation();
    if (end_set.count(last) > 0) {
        return s;
    }

    if (chunker_contains_chinese(s)) {
        s += "\xe3\x80\x82";  // U+3002
    } else {
        s += ".";
    }

    return s;
}
