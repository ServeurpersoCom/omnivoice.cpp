#pragma once
// pipeline-tts.h: TTS generation pipeline for OmniVoice.
//
// Owns the LLM weights and exposes load / free plus debug entry points used
// by tests/*-cossim.py to validate each stage in isolation. Mirrors the
// layout of pipeline-codec.h: single backend, single shared WeightCtx, one
// ggml_gallocr per graph at compute time.

#include "backend.h"
#include "ggml-backend.h"
#include "omnivoice-llm.h"
#include "weight-ctx.h"

#include <cstdint>
#include <string>
#include <vector>

struct BPETokenizer;
struct MaskgitConfig;
struct PipelineCodec;

struct PipelineTTS {
    // Base GGUF kept open across module loads, closed on success.
    GGUFModel gguf;

    // LLM weights (Qwen3 backbone + audio_embeddings + audio_heads).
    OmniVoiceLM lm;

    // All LLM tensors share this WeightCtx, allocated once at end of load.
    WeightCtx wctx;

    // Backend pair (GPU + CPU fallback) and scheduler. The scheduler routes
    // ops to the CPU backend when the GPU does not implement them (typical
    // case : K-quant get_rows on CUDA, asserts in the CUDA backend without
    // a scheduler). Compute uses ggml_backend_sched_graph_compute, never
    // ggml_backend_graph_compute directly.
    BackendPair          bp;
    ggml_backend_t       backend;
    ggml_backend_sched_t sched;

    // Flash attention is enabled when a GPU backend is present and not
    // disabled by --no-fa. FP16 clamp is opt-in via --clamp-fp16 to avoid
    // overflow on sub-Ampere CUDA where matmul accumulates in FP16.
    bool use_flash_attn;
    bool clamp_fp16;
};

// Load the LLM GGUF, copy all weights to the backend, close the GGUF mapping.
// Returns true on success. Leaves the struct in a clean state on failure.
bool pipeline_tts_load(PipelineTTS * pt, const char * gguf_path, BackendPair bp, bool use_fa, bool clamp_fp16);

// Release weights. Safe on a zeroed struct.
void pipeline_tts_free(PipelineTTS * pt);

// Full LLM forward in a single graph : custom embed -> 28L stack -> audio_heads
// reshape. Output is audio_logits in GGML layout (V fast, K mid, S slow).
// input_ids is laid out [K, S] row-major (k slow, s fast), audio_mask is [S]
// with 0 / 1 entries. attention_mask is optional [S, S] int with 0 / 1
// entries: 1 = attended, 0 = blocked. NULL means bidirectional (no padding).
// Positions are 0..S-1.
std::vector<float> pipeline_tts_llm_forward(PipelineTTS *   pt,
                                            const int32_t * input_ids,
                                            const int32_t * audio_mask,
                                            const int32_t * attention_mask,
                                            int             K,
                                            int             S,
                                            const char *    dump_hidden_dir  = nullptr,
                                            const char *    dump_hidden_name = nullptr);

// Pre-computed inputs that stay constant across the 32 MaskGIT steps for a
// chunk : audio_mask (B' * S floats), inv_mask, positions (S int32), and
// attn_f16 (B' * S * S F16 bias). Building these once instead of 32 times
// saves ~9 ms / step on the typical voice cloning shape (S ~ 1880, B'=2).
// Holds the original int32 pointers too so the debug loop path that calls
// the single forward can still hand them down unchanged.
struct MaskgitBatchedCtx {
    int B_prime;
    int S;

    std::vector<float>    mask_f;      // [B', S]
    std::vector<float>    inv_mask_f;  // [B', S]
    std::vector<int32_t>  positions;   // [S]
    std::vector<uint16_t> attn_f16;    // [B', S, S], empty when no mask
    bool                  has_attn_mask;

    const int32_t * audio_mask_raw;
    const int32_t * attn_mask_raw;
};

// Pre-compute the batched context from the prompt buffers. The original
// int32 pointers must stay valid for the lifetime of the context (the
// debug loop path reads them directly).
void pipeline_tts_llm_batched_ctx_init(MaskgitBatchedCtx * ctx,
                                       const int32_t *     audio_mask,
                                       const int32_t *     attention_mask,
                                       int                 B_prime,
                                       int                 S);

// Batched version : runs B' independent forwards (cond + uncond stacked).
// input_ids  [B', K, S]      row-major (b slow, k mid, s fast). Mutates
//                              between MaskGIT steps as tokens get demasked.
// ctx                          pre-computed audio_mask / inv / positions /
//                              attn_f16 buffers shared across the 32 steps
//                              of one chunk. See MaskgitBatchedCtx.
// T_audio    when 0, output is the full logits [B', V, K, S] (debug path).
//            when > 0, output is the audio-only window concatenated as
//            [cond_audio | uncond_audio], each [V, K, T_audio]. cond row
//            takes positions [S - T_audio, S), uncond row takes [0, T_audio).
//            Avoids transferring ~5.6x more data on the GPU->CPU bus.
// Output     either [B', V, K, S] (T_audio == 0) or 2*V*K*T_audio (T_audio > 0).
std::vector<float> pipeline_tts_llm_forward_batched(PipelineTTS *             pt,
                                                    const int32_t *           input_ids,
                                                    const MaskgitBatchedCtx * ctx,
                                                    int                       K,
                                                    int                       T_audio         = 0,
                                                    const char *              dump_hidden_dir = nullptr);

// Public TTS entry : tokenize text, build prompt + CFG batch, run the MaskGIT
// iterative decoder. Returns flat audio_tokens of size K * T (k slow, t fast)
// or an empty vector on failure. ref_text and ref_audio_tokens enable the
// voice cloning path : when ref_audio_tokens is non-NULL it must point to
// ref_T audio frames laid out [K, ref_T] and ref_text should hold the
// transcript (concatenated to text via _combine_text). Pass NULL / 0 / ""
// for the pure TTS path. The denoise flag triggers the <|denoise|> marker
// only when ref_audio_tokens is non-NULL, matching the reference.
std::vector<int32_t> pipeline_tts_generate(PipelineTTS *         pt,
                                           const BPETokenizer *  tok,
                                           const std::string &   text,
                                           const std::string &   lang,
                                           const std::string &   instruct,
                                           int                   T,
                                           bool                  denoise,
                                           const MaskgitConfig & mg_cfg,
                                           const std::string &   ref_text,
                                           const int32_t *       ref_audio_tokens,
                                           int                   ref_T,
                                           const char *          dump_dir,
                                           uint32_t *            ctr_lo_inout = nullptr);

// Full TTS synthesis : pipeline_tts_generate followed by pipeline_codec_decode.
// Returns mono waveform at 24 kHz of length T * codec.hop_length, empty on
// failure. Refuses to decode if any audio_token equals lm.audio_mask_id, which
// would corrupt the RVQ lookup. ref_text and ref_audio_tokens follow the same
// convention as pipeline_tts_generate. ctr_lo_inout threads the Philox counter
// across calls when chunking, see maskgit_generate.
std::vector<float> pipeline_tts_synthesize(PipelineTTS *         pt,
                                           PipelineCodec *       pc,
                                           const BPETokenizer *  tok,
                                           const std::string &   text,
                                           const std::string &   lang,
                                           const std::string &   instruct,
                                           int                   T,
                                           bool                  denoise,
                                           const MaskgitConfig & mg_cfg,
                                           const std::string &   ref_text,
                                           const int32_t *       ref_audio_tokens,
                                           int                   ref_T,
                                           const char *          dump_dir,
                                           uint32_t *            ctr_lo_inout = nullptr);

// Long-form TTS with automatic chunking and post-processing. If
// chunk_duration_sec <= 0 or the estimated audio length fits below
// chunk_threshold_sec, behaves as a single pipeline_tts_synthesize call. For
// longer text, splits on punctuation via chunk_text_punctuation, generates
// chunk 0 with no reference, then uses chunk 0 audio tokens as voice prompt
// for chunks 1..N (matching the no-ref branch of _generate_chunked in
// omnivoice.py). Cross-fades chunks, applies remove_silence + final volume
// adjustment + fade_and_pad. Returns mono float PCM at the codec sample rate
// (24 kHz). External voice cloning via ref_audio_tokens propagates to every
// chunk identically (matching the all-have-ref branch upstream).
// T_override > 0 forces the single-shot path with that exact frame count,
// bypassing the duration estimator and the chunker. Use 0 for auto.
// ref_rms < 0 means no external reference -> apply peak/0.5 normalisation.
// 0 <= ref_rms < 0.1 -> rescale output audio by ref_rms / 0.1 (matches the
// quiet-ref branch of _post_process_audio). ref_rms >= 0.1 -> no rescale.
std::vector<float> pipeline_tts_synthesize_long(PipelineTTS *         pt,
                                                PipelineCodec *       pc,
                                                const BPETokenizer *  tok,
                                                const std::string &   text,
                                                const std::string &   lang,
                                                const std::string &   instruct,
                                                int                   T_override,
                                                float                 chunk_duration_sec,
                                                float                 chunk_threshold_sec,
                                                bool                  denoise,
                                                const MaskgitConfig & mg_cfg,
                                                const std::string &   ref_text,
                                                const int32_t *       ext_ref_tokens,
                                                int                   ext_ref_T,
                                                float                 ref_rms,
                                                const char *          dump_dir);
