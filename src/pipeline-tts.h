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
#include "omnivoice.h"
#include "static-graph.h"
#include "weight-ctx.h"

#include <cstdint>
#include <string>
#include <vector>

struct BPETokenizer;
struct MaskgitConfig;
struct PipelineCodec;
struct VoiceDesign;

struct PipelineTTS {
    // Base GGUF kept open across module loads, closed on success.
    GGUFModel gguf;

    // LLM weights (Qwen3 backbone + audio_embeddings + audio_heads).
    OmniVoiceLM lm;

    // All LLM tensors share this WeightCtx, allocated once at end of load.
    WeightCtx wctx;

    // Backend pair (GPU + CPU fallback) and scheduler. The scheduler routes
    // ops to the CPU backend when the GPU does not implement them (typical
    // case: K-quant get_rows on CUDA, asserts in the CUDA backend without
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

// Full LLM forward in a single graph: custom embed -> 28L stack -> audio_heads
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
// chunk: audio_mask (B' * S floats), inv_mask, positions (S int32), and
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

    // Static LM graph replayed across the MaskGIT steps. The topology is fixed
    // within a request, so the graph is built once, allocated once directly when the backend supports every op,
    // otherwise through the scheduler, and replayed with a fresh input_ids upload per step. lm_key_K
    // and lm_key_T_audio detect a shape change that forces a rebuild.
    struct ggml_context * lm_gctx         = nullptr;
    struct ggml_cgraph *  lm_gf           = nullptr;
    struct ggml_tensor *  lm_text_ids     = nullptr;
    struct ggml_tensor *  lm_shifted      = nullptr;
    struct ggml_tensor *  lm_mask         = nullptr;
    struct ggml_tensor *  lm_inv_mask     = nullptr;
    struct ggml_tensor *  lm_positions    = nullptr;
    struct ggml_tensor *  lm_attn         = nullptr;
    struct ggml_tensor *  lm_cond_audio   = nullptr;
    struct ggml_tensor *  lm_uncond_audio = nullptr;
    struct ggml_tensor *  lm_logits       = nullptr;
    StaticGraph           lm_graph;
    std::vector<int32_t>  shifted;
    std::vector<int32_t>  text_ids;
    int                   lm_key_K       = 0;
    int                   lm_key_T_audio = -1;
    bool                  lm_built       = false;
};

// Pre-compute the batched context from the prompt buffers. The original
// int32 pointers must stay valid for the lifetime of the context (the
// debug loop path reads them directly).
void pipeline_tts_llm_batched_ctx_init(MaskgitBatchedCtx * ctx,
                                       const int32_t *     audio_mask,
                                       const int32_t *     attention_mask,
                                       int                 B_prime,
                                       int                 S);

// Batched version: runs B' independent forwards (cond + uncond stacked).
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
std::vector<float> pipeline_tts_llm_forward_batched(PipelineTTS *       pt,
                                                    const int32_t *     input_ids,
                                                    MaskgitBatchedCtx * ctx,
                                                    int                 K,
                                                    int                 T_audio         = 0,
                                                    const char *        dump_hidden_dir = nullptr);

// Release the static LM graph held by the context. Resets the scheduler so a
// later graph of a different shape re-splits, then frees the graph context.
// Safe to call on a context that never built a graph.
void pipeline_tts_llm_batched_ctx_free(PipelineTTS * pt, MaskgitBatchedCtx * ctx);

// Low-level token generator: tokenize text, build prompt + CFG batch, run
// the MaskGIT iterative decoder. Returns flat audio_tokens of size K * T (k
// slow, t fast) or an empty vector on failure. ref_text and ref_audio_tokens
// enable the voice cloning path: when ref_audio_tokens is non-NULL it must
// point to ref_T audio frames laid out [K, ref_T] and ref_text should hold
// the transcript (concatenated to text via _combine_text). Pass NULL / 0 /
// ""  for the pure TTS path. The denoise flag triggers the <|denoise|>
// marker only when ref_audio_tokens is non-NULL, matching the reference.
// instruct must already be resolved against the VoiceDesign vocabulary, see
// pipeline_tts_resolve_instruct. Used by the --maskgit-test debug path and
// internally by pipeline_tts_synthesize.
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

// Validate and normalise the raw instruct string against the voice-design
// vocabulary. The target language is selected from the synthesis text: any
// CJK ideograph in text -> Chinese, otherwise English. On error, prints a
// "[TTS] ERROR: ..." line and returns false. Bypassed by callers that go
// through pipeline_tts_synthesize, which resolves the instruct internally.
bool pipeline_tts_resolve_instruct(const VoiceDesign * vd,
                                   const std::string & text,
                                   const std::string & raw,
                                   std::string *       out);

// Convert a duration in seconds to a frame count using the codec frame rate
// (sample_rate / hop_length). Clamps to a minimum of 1 frame. Single source
// of truth shared by every CLI / API entry that exposes --duration.
int pipeline_tts_duration_sec_to_tokens(const PipelineCodec * pc, float duration_sec);

// Public TTS synthesis entry. Resolves the instruct string against vd, picks
// between the single-shot, chunked auto-voice and voice-cloning paths from
// the params struct, and fills `out` with mono float PCM at the codec sample
// rate (24 kHz). Returns OV_STATUS_OK on success; on failure returns a
// negative ov_status describing the cause and leaves out empty. The whole
// pipeline mirrors _post_process_audio in omnivoice.py: when no reference
// audio is provided the output is peak/0.5 normalised, when a reference is
// provided its original RMS is threaded into post-proc to rescale the output
// back to reference loudness.
//
// The ov_tts_params struct, the ov_audio struct, the ov_status enum and the
// ov_audio_free entry are declared in omnivoice.h; this entry consumes them
// directly.
ov_status pipeline_tts_synthesize(PipelineTTS *         pt,
                                  PipelineCodec *       pc,
                                  const BPETokenizer *  tok,
                                  const VoiceDesign *   vd,
                                  const ov_tts_params * params,
                                  ov_audio *            out);
