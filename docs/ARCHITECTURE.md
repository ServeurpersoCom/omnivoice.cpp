# Architecture

Technical reference for omnivoice.cpp, the GGML port of OmniVoice
(k2-fsa/OmniVoice). This document covers the model, the conversion to
GGUF, the inference pipeline, the GGML graph conventions, and the CLI
tools.

## Upstream model

OmniVoice (Xiaomi / k2-fsa, Apache 2.0) is a multilingual zero-shot
text-to-speech system covering 646 languages. It targets three modes :

  voice cloning  reference audio plus reference transcript drive the
                 target speaker identity
  voice design   six attribute categories (gender, age, pitch, style,
                 volume, emotion) drive a synthesised speaker
  auto voice     no reference, the model picks a coherent speaker per
                 utterance

The system is a non autoregressive, mask-predict (MaskGIT) generative
model running on top of a Qwen3 backbone with custom audio
input/output and a separate audio tokenizer. The audio tokenizer is
the Higgs Audio v2 codec (`bosonai/higgs-audio-v2-tokenizer`,
Apache 2.0), which combines a HuBERT semantic stream, a DAC acoustic
stream, and an 8-codebook residual vector quantiser at 25 frames per
second over 24 kHz mono audio.

Single public checkpoint : `k2-fsa/OmniVoice` (3.1 GB).

  Backbone        Qwen3 0.6B (28 layers, hidden 1024, GQA 16/8)
  Audio codebooks 8 residual, 1024 entries each plus 1 mask token
  Audio framerate 25 Hz
  Hop length      960 samples
  Sample rate     24 kHz mono
  Semantic SR     16 kHz (HuBERT input)
  MaskGIT steps   32 default, configurable

## Build

```
git clone --recurse-submodules https://github.com/ServeurpersoCom/omnivoice.cpp.git
cd omnivoice.cpp
./buildcuda.sh      # NVIDIA GPU
./buildvulkan.sh    # AMD/Intel GPU (Vulkan)
./buildcpu.sh       # CPU only
./buildall.sh       # all backends, runtime DL loading
```

The GGML submodule lives at `https://github.com/ServeurpersoCom/ggml.git`
and provides two custom ops required by the codec :
`GGML_OP_SNAKE` and `GGML_OP_COL2IM_1D`. Both have CPU, CUDA, Metal,
and Vulkan kernels.

## Model conversion

```
./checkpoints.sh    # hf download k2-fsa/OmniVoice -> checkpoints/OmniVoice/
./convert.py        # 2 GGUFs in BF16 -> models/
./quantize.sh       # base LM Q8_0 (tokenizer stays at native dtype)
```

Outputs :

```
models/omnivoice-base-BF16.gguf       1.2 GB    LLM + audio_emb + audio_heads + tokenizer
models/omnivoice-base-Q8_0.gguf       626 MB    quantized base, 1.9x smaller
models/omnivoice-tokenizer-F32.gguf   702 MB    HuBERT + DAC + RVQ + fc/fc2 (native F32)
```

The audio tokenizer GGUF preserves the source dtype 1:1. The reference
checkpoint stores the codec at F32, so the GGUF stays F32 to avoid
truncation noise across the 8-stage RVQ residual chain. Late codebooks
fall below 50 percent codebook match against the reference if any
intermediate weight is rounded to BF16.

Quantisation policy : Q8_0 only on the base LM. The 612 M parameter
backbone is small enough that lower quants degrade quality without
meaningful size gains.

## GGUF layout

`omnivoice-base-{quant}.gguf` (arch `omnivoice-lm`) :

```
metadata
  general.architecture                   omnivoice-lm
  block_count                            28
  embedding_length                       1024
  feed_forward_length                    3072
  head_count                             16
  head_count_kv                          8        (GQA 2:1)
  key_length                             128
  vocab_size                             151676
  context_length                         40960
  layer_norm_rms_eps                     1e-6
  rope_freq_base                         1e6
  omnivoice.tie_word_embeddings          true
  omnivoice.num_audio_codebook           8
  omnivoice.audio_vocab_size             1025
  omnivoice.audio_mask_id                1024
  omnivoice.audio_codebook_weights       [8, 8, 6, 6, 4, 4, 2, 2]
  omnivoice.special.denoise              151669
  omnivoice.special.lang_start           151670
  omnivoice.special.lang_end             151671
  omnivoice.special.instruct_start       151672
  omnivoice.special.instruct_end         151673
  omnivoice.special.text_start           151674
  omnivoice.special.text_end             151675
  tokenizer (Qwen2 BPE, 151676 vocab, 151387 merges, 33 added_tokens)

tensors (312)
  llm.embed_tokens.weight                (151676, 1024)
  llm.norm.weight                        (1024,)
  llm.layers.0..27.{q,k,v,o}_proj.weight                              GQA, no bias
  llm.layers.0..27.self_attn.{q_norm, k_norm}.weight                  per-head RMSNorm (128,)
  llm.layers.0..27.{input,post_attention}_layernorm.weight            RMSNorm
  llm.layers.0..27.mlp.{gate,up,down}_proj.weight                     SwiGLU, no bias
  audio_embeddings.weight                (8200, 1024)                 8 codebooks * 1025 vocab
  audio_heads.weight                     (8200, 1024)                 audio output, no bias
```

`omnivoice-tokenizer-{quant}.gguf` (arch `omnivoice-tokenizer`) :

```
metadata
  omnivoice.sample_rate                             24000
  omnivoice.semantic_sample_rate                    16000
  omnivoice.downsample_factor                       320
  omnivoice.codebook_size                           1024
  omnivoice.codebook_dim                            64
  omnivoice.acoustic.encoder_hidden_size            64
  omnivoice.acoustic.decoder_hidden_size            1024
  omnivoice.acoustic.hidden_size                    256
  omnivoice.acoustic.n_codebooks                    9        (only 8 used)
  omnivoice.acoustic.hop_length                     960
  omnivoice.acoustic.upsampling_ratios              [8, 5, 4, 2, 3]
  omnivoice.acoustic.downsampling_ratios            [8, 5, 4, 2, 3]
  omnivoice.semantic.hidden_size                    768       (HuBERT base)
  omnivoice.semantic.intermediate_size              3072
  omnivoice.semantic.num_attention_heads            12
  omnivoice.semantic.num_hidden_layers              12
  omnivoice.semantic.num_feat_extract_layers        7
  omnivoice.semantic.conv_dim                       [512]*7
  omnivoice.semantic.conv_kernel                    [10, 3, 3, 3, 3, 2, 2]
  omnivoice.semantic.conv_stride                    [5, 2, 2, 2, 2, 2, 2]
  omnivoice.semantic.num_conv_pos_embeddings        128
  omnivoice.semantic.num_conv_pos_embedding_groups  16
  omnivoice.semantic.layer_norm_eps                 1e-5

tensors (486)
  acoustic_encoder.*                     DAC encoder, 5 blocks, downsamples 8 5 4 2 3
  acoustic_decoder.*                     DAC decoder, 5 blocks, upsamples 8 5 4 2 3
  encoder_semantic.*                     semantic conv blocks
  semantic_model.*                       HuBERT base, weight_norm folded
  quantizer.quantizers.0..7.{codebook.embed, project_in.{w,b}, project_out.{w,b}}
  fc.{weight, bias}                      1024 -> 1024 (after concat acoustic + semantic)
  fc2.{weight, bias}                     1024 -> 256 (before DAC decoder)
```

Single weight_norm fold at convert time :
`semantic_model.encoder.pos_conv_embed.conv.weight`, formula
`weight = v * g / ||v||_{dim=(0,1)}` matching
`torch._weight_norm(v, g, dim=2)`. Validated bit-perfect, max abs diff
3.9e-7 against the PyTorch reference.

## Component architecture

### Qwen3 0.6B backbone with custom IO

Standard Qwen3 modulo two changes :

  input embed   hybrid text plus audio, weighted sum across 8
                codebooks gated by `audio_mask`
  output head   custom `audio_heads` Linear (8200, 1024), no text
                `lm_head`

```
input_ids [B, 8, S] int          (text on row 0, audio codes on rows 1..7)
audio_mask [B, S] bool

text_emb  = embed_tokens(input_ids[:, 0, :])               (B, S, 1024)
shifted   = input_ids * audio_mask + offsets[None, :, None]
                                  offsets = arange(8) * 1025
audio_emb = audio_embeddings(shifted).sum(dim=1)           (B, S, 1024)
inputs    = where(audio_mask, audio_emb, text_emb)         (B, S, 1024)

x = qwen3_forward(inputs, attention_mask, position_ids)    (B, S, 1024)

logits_flat = x @ audio_heads.weight.T                     (B, S, 8200)
logits      = reshape (B, 8, S, 1025)
```

Qwen3 specifics already in llama.cpp :

  28 layers, hidden 1024, intermediate 3072
  16 query heads + 8 KV heads (GQA 2:1), head_dim 128
  per-head RMSNorm on Q and K (q_norm, k_norm shape (128,)) before RoPE
  no bias on Q/K/V/O/MLP
  RoPE theta = 1e6
  SwiGLU MLP
  tie_word_embeddings = true (`lm_head` absent, output goes through audio_heads)

### MaskGIT decoder

Iterative non autoregressive decoder, no KV cache. Each step is a full
prefill of the LLM on the current input.

Prompt (per item, broadcast across 8 codebooks) :

```
[<|denoise|>]?
<|lang_start|> {iso_code or "None"} <|lang_end|>
<|instruct_start|> {style or "None"} <|instruct_end|>
<|text_start|> {ref_text + " " + text} <|text_end|>
{ref_audio_codes}?
{MASK x num_target_tokens}
```

Unconditional prompt for CFG = the trailing `num_target_tokens` mask
tokens only. Batched (cond + uncond) doubles the batch dim.

```
for step in 0..num_step-1 :
    forward(input_ids, audio_mask, attention_mask)         (2B, 8, S, 1025)
    log_probs = log_softmax(c + cfg_scale * (c - u))
    log_probs[..., MASK_ID] = -inf
    if class_temp > 0 :
        keep_top_k_ratio(log_probs, 0.1)
        gumbel_sample(temp = class_temp)
    pred  = argmax(log_probs)
    score = log_probs.max - layer_idx * layer_penalty      (5.0)
    if pos_temp > 0 :
        score += gumbel * pos_temp
    score[already_unmasked] = -inf
    topk_idx = topk(score.flatten(), schedule[step])
    tokens[topk_idx] = pred[topk_idx]
    update batch_input_ids cond and uncond
```

Schedule of newly unmasked positions per step is computed from
`_get_time_steps(t_start=0, t_end=1, num_step, t_shift=0.1)` then
`ceil(N_total * (t[step+1] - t[step]))`. 32 steps default.

KV cache is not usable across MaskGIT steps. The attention is fully
bidirectional, so the prefix hidden states depend on the current
target state through every layer. As tokens get progressively
unmasked the K and V tensors of the prefix at every layer above the
embeddings drift, which forbids the standard prefix-cache trick that
works for causal LMs. Each step is therefore a full prefill of the
LLM at cost `2 * forward_full(B, S)` (the 2 accounts for the cond +
uncond CFG rows).

Determinism. With `class_temperature = 0` and `position_temperature = 0`
the decoder is bit deterministic. Higher temperatures rely on a
seedable Philox4x32-10 PRNG. The pipeline threads the Philox counter
across MaskGIT calls so that chunked inference matches the global RNG
drift of the PyTorch reference.

#### Inner-loop optimisations

The num_step iterations of one chunk run on a fixed shape (`S`, `K`, `B'`) so
the per-step overhead can be cut without touching the math.

`pipeline_tts_llm_forward_batched` accepts a `T_audio` parameter that
narrows the GPU output to the audio window only. The MaskGIT decoder
reads cond logits at `[S - T, S)` on row 0 and uncond logits at
`[0, T)` on row 1, so the full `[B', V, K, S]` tensor is wasteful.
With `T_audio > 0` the function builds two `ggml_view_4d` over those
ranges, makes them contiguous via `ggml_cont`, and only those two
sub-tensors are flagged as graph outputs. The GPU to CPU copy shrinks
from `B' * V * K * S` floats to `2 * V * K * T_audio` floats, around
5.6x less on the typical voice cloning shape (S ~ 1880, T ~ 350).
When `T_audio == 0` the function falls back to the full output, used
by the debug dump path that needs every position.

`MaskgitBatchedCtx` holds the inputs that stay constant across the
steps : the F32 audio mask and its complement, the RoPE position
vector, and the F16 attention bias. The bias is the heaviest piece,
`B' * S * S` F16 conversions per step (about 7 M ops on the typical
shape). `pipeline_tts_llm_batched_ctx_init` precomputes the bias once
per chunk, with a single `ggml_fp32_to_fp16` call for each of the two
distinct values (1.0 and 0.0) hoisted out of the conversion loop. The
context also keeps the original int32 pointers so the debug loop path
can hand them down to the single forward unchanged.

Both optimisations preserve the math exactly, the only side effect is
a slight reordering of the GPU FP32 reductions when the scheduler
fuses the new output nodes differently, which moves the audio cosine
similarity by a few times 1e-6. Token-level results stay 100 percent
exact against the PyTorch reference.

### Audio tokenizer pipeline

Encode (voice cloning reference path) :

```
ref_audio @ 24 kHz                                  (1, 1, T_samples)
  -> resample 16 kHz                                (kaiser polyphase)
  -> pad 160 each side
  -> HuBERT.feature_extractor (320x downsample)
  -> HuBERT.feature_projection (LayerNorm + Linear)
  -> + pos_conv_embed (folded)
  -> 12 transformer layers
  -> mean over 13 hidden states                     (1, 768, T_sem)
  -> downsample by 2                                (semantic_downsample_factor)
  -> SemanticEncoder (conv blocks)                  (1, 768, T_frames)
  -> e_acoustic (DAC encoder, 5 down-blocks)        (1, 256, T_frames)
  -> concat dim=1                                   (1, 1024, T_frames)
  -> fc Linear (1024 -> 1024)                       (1, 1024, T_frames)
  -> RVQ encode (8 codebooks residual)              (1, 8, T_frames) int @ 25 fps
```

Decode (TTS path) :

```
codes [B, 8, T] int
  -> RVQ decode :
     for k in 0..7 :
         e_k = codebook[k].embed[codes[k, :]]        (B, T, 64)
         p_k = e_k @ project_out[k].W.T + bias[k]    (B, T, 1024)
         out += p_k
  -> transpose (B, 1024, T)
  -> fc2 Linear (1024 -> 256)                        (B, 256, T)
  -> acoustic_decoder DAC :
     conv1 (256 -> 1024, k=7, pad=3)
     for block in 0..4, ratios [8, 5, 4, 2, 3] :
         snake1 (alpha)
         conv_t1 (IC -> OC, k=2*r, stride=r,
                  padding=ceil(r/2), output_padding=r%2)
         for res_unit in 0..2, dilations [1, 3, 9] :
             snake1 (alpha)
             conv1 (OC, OC, k=7, dil=d, pad=3*d)
             snake2 (alpha)
             conv2 (OC, OC, k=1)
             residual add
     snake1 (alpha) final
     conv2 (32 -> 1, k=7, pad=3)
  -> audio (B, 1, 960*T)
```

960x upsample = 8 * 5 * 4 * 2 * 3. T_in @ 25 fps -> T_out @ 24 kHz exact.

### DAC decoder block channels

```
block 0 : IC=1024  OC=512  stride=8  K=16  pad=4  output_pad=0
block 1 : IC=512   OC=256  stride=5  K=10  pad=3  output_pad=1
block 2 : IC=256   OC=128  stride=4  K=8   pad=2  output_pad=0
block 3 : IC=128   OC=64   stride=2  K=4   pad=1  output_pad=0
block 4 : IC=64    OC=32   stride=3  K=6   pad=2  output_pad=1
final   : 32 -> 1
```

PyTorch ConvTranspose1d formula :
`T_out = (T_in - 1)*stride - 2*padding + dilation*(kernel - 1) + output_padding + 1`

With our parameters (d=1, k=2*s, p=ceil(s/2), op=s%2) the formula
collapses to `T_out = stride * T_in` exactly for all five blocks.

### Snake activation

DAC reference formula (Hugging Face `Snake1d.forward`) :
`y = x + (alpha + 1e-9).reciprocal() * sin(alpha * x)^2`

`ggml_snake(x, a, inv_b)` computes `y = x + sin^2(a * x) * inv_b`.
Mapping :

  a       = alpha                       (loaded direct, BF16 to F32)
  inv_b   = 1/(alpha + 1e-9)            (precomputed CPU side at load, F32)

Both stored as F32 `[1, C]` tensors. `alpha` shape in checkpoint :
`(1, C, 1)`, ggml ne = (1, C, 1). C lives on ne[1]. Loader reads C from
`mt->ne[1]`.

### ConvTranspose1d via GEMM + col2im_1d

PyTorch `nn.ConvTranspose1d(IC, OC, kernel=K, stride=s, padding=p)` with
weight shape `(IC, OC, K)`. GGML decomposition :

```
1. Permute weight (IC, OC, K) PyTorch -> (IC, K*OC) ggml at load time.
   Layout : dst[(oc*K + k) * IC + ic] = src[ic*OC*K + oc*K + k]
   This makes k vary faster than oc inside the K*OC axis, matching
   what ggml_compute_forward_col2im_1d_impl expects :
     col_data[(oc * K + k) + t_in * K_OC]

2. Build runtime graph :
   xt   = ggml_cont(ctx, ggml_transpose(ctx, x))           # [IC, T_in]
   col  = ggml_mul_mat(ctx, w, xt)                         # [K*OC, T_in]
   y    = ggml_col2im_1d(ctx, col, stride, OC, padding)    # [T_no_op, OC]
   if (output_pad > 0)
       y = ggml_pad(ctx, y, output_pad, 0, 0, 0)           # right-pad zeros
   if (bias)
       y = ggml_add(ctx, y, bias_2d)
```

Validated math : `T_no_op = (T_in - 1)*stride + K - 2*pad`. Adding
`output_pad` right-pad gives the PyTorch output size exactly.

### RVQ codec

Per-codebook tensors (k = 0..7) :

```
codebook.embed         (1024, 64) PyTorch -> ggml ne=(64, 1024)
project_in.weight      (64, 1024) PyTorch -> ggml ne=(1024, 64)   encode-only
project_in.bias        (64,)                                      encode-only
project_out.weight     (1024, 64) PyTorch -> ggml ne=(64, 1024)
project_out.bias       (1024,)
```

Decode graph (per codebook k, accumulated) :

```
codes_k = ggml_view_1d(codes, T, k * stride)               # [T] i32
e_k     = ggml_get_rows(embed[k], codes_k)                 # [64, T]
p_k     = ggml_mul_mat(project_out_w[k], e_k)              # [1024, T]
p_k     = ggml_add(p_k, project_out_b[k])
acc    += p_k
```

Encode (residual loop) :

```
residual = embeddings_in
for k in 0..7 :
    e_k       = project_in[k](residual)
    codes_k   = argmin_i ||e_k - codebook[k].embed[i]||^2
    quantized = project_out[k](codebook[k].embed[codes_k])
    residual -= quantized
```

### HuBERT semantic encoder

12 transformer layers Pre-LN, GELU FFN, MHA 12 heads * 64 dim, biases
on all QKVO. Pre-conv feature extractor : 7 Conv1D layers, kernels
`[10, 3, 3, 3, 3, 2, 2]`, strides `[5, 2, 2, 2, 2, 2, 2]`, GroupNorm on
the first only, GELU between. Feature projection LayerNorm + Linear
(512 -> 768). Positional embedding via grouped Conv1D (128 kernel,
16 groups), `weight_norm` folded at convert time. Final LayerNorm.

Output computation :

```
mean(stack(all_13_hidden_states, dim=1), dim=1) # (B, T_sem, 768)
```

This is unusual : the encoder averages across the initial input plus
the 12 transformer layer outputs, not just the last hidden state.

## Long-form TTS pipeline

`pipeline_tts_synthesize_long` orchestrates inputs longer than the
chunking threshold. It mirrors `_generate_chunked` in
`omnivoice/models/omnivoice.py`.

```
1. Estimate total target tokens via duration_estimate_tokens.
2. If T_total <= chunk_threshold_sec * frame_rate, run a single shot
   pipeline_tts_synthesize and skip chunking.
3. Otherwise split text on punctuation with chunk_text_punctuation,
   targeting chunk_duration_sec seconds per chunk.
4. Generate chunks sequentially :
     - chunk 0 with no reference (auto voice / voice design path) or
       with the external reference (cloning path)
     - in the auto voice case, the audio tokens of chunk 0 become the
       voice prompt for chunks 1..N, locking in the speaker identity
5. Cross-fade decoded chunks with cross_fade_chunks(rate, 0.3 s).
6. Apply post-processing on the merged waveform.
```

A shared Philox counter `ctr_lo` is threaded across MaskGIT calls so
PRNG state advances continuously between chunks, matching the global
`torch.cuda.manual_seed` behaviour on the reference side.

### Text chunking

`chunk_text_punctuation(text, chunk_len, min_chunk_len)` splits text on
sentence-ending punctuation (skipping abbreviation periods), then
merges sentences into chunks of at most `chunk_len` UTF-8 codepoints.
Undersized chunks (< `min_chunk_len`) are merged into a neighbour.
The function operates on UTF-8 strings and treats length as codepoints,
matching Python `len(str)` semantics. Per chunk character budget :

```
n_chars              = utf8_codepoint_count(full_text)
avg_tokens_per_char  = T_total / n_chars
chunk_len            = (int)(chunk_duration_sec * frame_rate / avg_tokens_per_char)
```

`add_punctuation(text)` appends a terminal `.` (Latin) or its CJK
equivalent when missing. Used on the reference transcript when
`preprocess_prompt` is on.

### Audio post-processing

`audio-postproc.h` is a strict math port of `omnivoice/utils/audio.py`
plus the relevant `pydub.silence` routines. All public functions take
and return float32 mono PCM in [-1, 1] at the pipeline rate (24 kHz).
Silence detection runs on int16 samples to match pydub bit-for-bit.

```
remove_silence(buf, min_silence_ms, keep_silence_ms,
               seek_step_ms, threshold_dbfs)

  Splits buf on contiguous silent regions where every
  seek_step_ms-long frame stays below threshold_dbfs (RMS, S16,
  default -50 dBFS), keeps keep_silence_ms of leading and trailing
  silence around each retained segment, and concatenates the result.

cross_fade_chunks(chunks, rate, fade_seconds)

  Concatenates audio chunks with a linear cross-fade of fade_seconds
  at each junction.

peak_normalize_half(buf)

  Scales buf so peak |x| equals 0.5. Used in pure auto voice when no
  reference RMS is available.

fade_and_pad(buf, rate, fade_seconds, pad_seconds)

  Applies a linear fade in / fade out and pads silence at the start
  and end. Default fade 0.05 s, pad 0.05 s. Mirrors final reference
  post-step.
```

`ref_rms` is plumbed end to end and decides the volume branch :

  ref_rms < 0      pure auto voice, peak_normalize_half on the
                    cross-faded waveform
  ref_rms < 0.1    quiet reference, rescale by ref_rms / 0.1
  otherwise         no-op

When a reference WAV is provided, the CLI computes its RMS on the F32
samples after optional silence trimming and passes it down. The same
quantity is used on the PyTorch side.

### Voice modes

```
auto voice        no ref-wav. Chunk 0 generates with no reference,
                   subsequent chunks reuse chunk 0 audio tokens as the
                   voice prompt. peak_normalize_half on output.

voice design      no ref-wav, --instruct provides one or more attribute
                   markers (gender, age, pitch, style, volume, emotion)
                   resolved by voice_design.h to the EN/ZH instruct
                   string the reference uses. Chunking behaves like
                   auto voice.

voice cloning    --ref-wav and --ref-text provided. The reference is
                   resampled to 16 kHz, run through the audio tokenizer
                   encoder, and the resulting RVQ codes are reused as
                   the voice prompt for every chunk. The reference RMS
                   sets the target loudness.
```

## Public API

Two layers, picked by use case.

### Top-level public ABI : src/omnivoice.h

Single-header, plain C99, linkage `extern "C"`. The opaque `ov_context`
handle aggregates the GGML backend pair, the LM pipeline, the audio
tokenizer codec, the BPE tokenizer and the voice-design vocabulary.
One init, one free, one synthesize call covers the full TTS path.
Same names, same struct layout, same calling convention from C, C++,
Python ctypes, Rust bindgen, Go cgo or any other binding generator.

```c
#include "omnivoice.h"

struct ov_init_params iparams;
ov_init_default_params(&iparams);
iparams.model_path = "models/omnivoice-base-Q8_0.gguf";
iparams.codec_path = "models/omnivoice-tokenizer-F32.gguf";

struct ov_context * ov = ov_init(&iparams);

struct ov_tts_params params;
ov_tts_default_params(&params);
params.text = "Hello world.";
params.lang = "English";

struct ov_audio audio = { 0 };
enum ov_status rc = ov_synthesize(ov, &params, &audio);
if (rc == OV_STATUS_OK) {
    /* audio.samples is a malloc'd buffer of audio.n_samples floats
       at audio.sample_rate Hz, audio.channels = 1 (mono) */
}
ov_audio_free(&audio);
ov_free(ov);
```

Status codes :

```
OV_STATUS_OK                 0
OV_STATUS_INVALID_PARAMS    -1   (mutually exclusive ref inputs etc.)
OV_STATUS_INSTRUCT_INVALID  -2   (instruct rejected by VoiceDesign)
OV_STATUS_GENERATE_FAILED   -3   (any internal generate / decode fail)
OV_STATUS_OOM               -4   (output samples allocation failed)
OV_STATUS_CANCELLED         -5   (cancel callback returned true)
```

`ov_tts_params` exposes `cancel` and `cancel_user_data`. The pipeline
polls between chunks of long-form output, so cancel granularity is
roughly `chunk_duration_sec` (15 s by default).

The MaskGIT sampler config is flattened directly into `ov_tts_params`
as seven `mg_*` fields ; `ov_tts_default_params` initialises them to
the reference defaults (`num_step=32, guidance_scale=2.0, t_shift=0.1,
layer_penalty_factor=5.0, position_temperature=5.0,
class_temperature=0.0, seed=42`).

`ov_version()` returns a static string of the form
`"MAJOR.MINOR.PATCH (git-hash, date)"`. The macros `OV_VERSION_MAJOR`,
`OV_VERSION_MINOR`, `OV_VERSION_PATCH` are also available at
compile time for feature-detection.

### ABI guarantee

`tests/abi-c.c` is built on every build with
`-std=c99 -Wall -Werror -pedantic`. It includes the public header,
calls every entry through its early-return path, and is wired into
the default build target. Any regression that breaks plain C
consumability fails the main build, not an opt-in step.

The static library `libomnivoice-core.a` is the default build
artefact. For binding consumers, configure with
`-DOMNIVOICE_SHARED=ON` to add a `libomnivoice.so` (or `.dll` /
`.dylib`) shared target that exports only the `ov_*` symbols ;
every internal `pipeline_*` and `backend_*` stays hidden behind
`-fvisibility=hidden`. Install rules follow `GNUInstallDirs`.

### Low-level API : src/pipeline-tts.h, src/pipeline-codec.h

Direct access to the LM forward (`pipeline_tts_llm_forward`,
`pipeline_tts_llm_forward_batched`), the MaskGIT-only path
(`pipeline_tts_generate`), the codec encode / decode
(`pipeline_codec_encode`, `pipeline_codec_decode`), the instruct
resolver (`pipeline_tts_resolve_instruct`) and the manual init / free
(`pipeline_tts_load`, `pipeline_codec_load`, `backend_init`).

Used by `--llm-test` and `--maskgit-test` in `omnivoice-tts`, by
`omnivoice-codec` for the standalone codec roundtrip, and by the
Python cossim harness through dump files.

This layer is intentionally not part of the public ABI (C++ types in
the signatures, no visibility export). It exists for the in-tree
debug paths and stays available as long as the bundled CLI tools
need it. The handle layer above is the recommended entry for
everything else.

## CLI tools

### omnivoice-tts

End-to-end synthesis : text on stdin, WAV file on disk. Verbatim
`--help` (the binary also prints an `omnivoice.cpp <hash> (<date>)`
banner line first) :

```
Usage: omnivoice-tts --model <gguf> --codec <gguf> [options] -o <out.wav> < text.txt

Required:
  --model <gguf>          LLM GGUF (F32 / BF16 / Q8_0)
  --codec <gguf>          Codec GGUF (omnivoice-tokenizer-*.gguf)
  -o <path>               Output WAV (24 kHz mono). '-' streams to stdout (pipe friendly).

Input:
  stdin                   Target text to synthesise. With -o '-', stdin is read
                          incrementally and synthesis starts as soon as the first
                          sentence boundary is reached. With -o file.wav, stdin is
                          read fully then synthesised in one shot.
  --srt <path>            Dub an SRT: synth each cue into its time slot, write one
                          timeline WAV ready to mux. Pairs with --ref-wav / --ref-rvq
                          for a cloned voice. Per cue duration comes from the SRT.

Optional:
  --format <fmt>          WAV output format: wav16, wav24, wav32 (default: wav16)
  --lang <str>            Language label (default 'None')
  --instruct <str>        Style instruction (default 'None')
  --duration <sec>        Output duration in seconds (default: estimate from text)
  --no-denoise            Omit the <|denoise|> prefix
  --ref-wav <path>        Reference WAV for voice cloning
  --ref-text <path>       Transcript file for the reference (required with --ref-wav / --ref-rvq)
  --ref-rvq <path>        Pre-encoded reference codes from omnivoice-codec (replaces --ref-wav)
  --seed <int>            Sampling seed (default: -1 for random)
  --steps <int>           MaskGIT decode steps (default: 32, fewer is faster)
  --no-preprocess-prompt  Skip ref-wav silence trim and ref-text terminal punctuation
  --chunk-duration <sec>  Long-form chunk duration (default: 15.0, <= 0 disables chunking)
  --chunk-threshold <sec> Activate chunking above this estimated duration (default: 30.0)
  --stream-by-line        Flush synthesis at each newline, one WAV header per line (-o '-')

Debug:
  --no-fa                 Disable flash attention
  --clamp-fp16            Clamp hidden states to FP16 range
  --dump <dir>            Dump intermediate tensors (f32) to <dir>
  --llm-test <input.bin>  Full LLM forward, dump audio_logits
  --maskgit-test          Greedy MaskGIT decoder, dump audio_tokens [K, T]
                          (no codec decode, reads target text from stdin)
```

### omnivoice-codec

Audio tokenizer round-trip : WAV to RVQ codes, RVQ codes to WAV.
Verbatim `--help` :

```
Usage: omnivoice-codec --model <gguf> -i <input>

Required:
  --model <gguf>          Codec GGUF (omnivoice-tokenizer-*.gguf)
  -i <path>               Input. WAV -> encode, .rvq -> decode

Optional:
  --format <fmt>          WAV output format: wav16, wav24, wav32 (default: wav16)

Output is auto-named next to input : clip.wav -> clip.rvq, clip.rvq -> clip.wav.
Encode applies the TTS reference preprocessing (RMS auto-gain, silence trim,
hop truncation); the resulting .rvq feeds omnivoice-tts --ref-rvq directly.
```

The `.rvq` file is a small binary container with shape `[8, T]` int32
codes plus a header carrying the sample rate and frame rate.

## Module map

```
src/
  backend.h            GGML backend init, scheduler factory, env override
  weight-ctx.h         Generic weight context for GGUF loaders
  gguf-weights.h       mmap GGUF, gf_load_tensor, gf_get_*
  audio-io.h           WAV read, mono write (S16 / S24 / F32)
  audio-resample.h     Kaiser polyphase 24 kHz <-> 16 kHz
  audio-postproc.h     remove_silence, peak_normalize_half, fade_and_pad,
                       cross_fade_chunks. Strict pydub / utils.audio port.
  wav.h                WAV header reader (PCM16/24/F32, mono/stereo)
  philox.h             Philox4x32-10 counter-based PRNG, PyTorch CUDA aligned
  debug.h              Tensor dumper for cossim tests

  bpe.h                Qwen2 / GPT-2 byte-level BPE tokenizer, GGUF loader
  lang-map.h           Language name to ISO 639-3 ID resolution
                       (auto-generated from omnivoice/utils/lang_map.py)
  voice-design.h       Speaker attribute validation and EN / ZH instruct
                       resolution (mirrors voice_design.py)
  text-chunker.h       chunk_text_punctuation, add_punctuation, END_PUNCTUATION
  duration-estimator.h RuleDurationEstimator port (per-script weights,
                       Unicode category fallback)

  rvq-codec.h          Residual VQ encode + decode (8 codebooks)
  dac-decoder.h        DAC acoustic decoder (5 blocks, ratios 8 5 4 2 3)
  dac-encoder.h        DAC acoustic encoder (mirror of decoder)
  semantic-enc.h       SemanticEncoder convs (768 -> 768)
  hubert-enc.h         HuBERT base (feature extractor + pos_conv +
                       12 transformer layers + final LN)

  qwen3-enc.h          Qwen3 transformer building blocks
  omnivoice-llm.h      OmniVoice TTS LLM weights and graph helpers
  prompt-tts.h         Prompt builder (denoise + lang + instruct + text +
                       ref + mask) and CFG batch stacking
  maskgit-tts.h        Iterative non autoregressive decoder, configurable
                       step count (32 default), CFG, layer penalty, gumbel
                       sampling, deterministic in greedy mode

  pipeline-codec.{h,cpp} Audio tokenizer end-to-end (encode and decode)
  pipeline-tts.{h,cpp}   Full TTS orchestration, single shot and chunked,
                         plus low-level entries kept available for the
                         debug paths and the cossim test harness
  omnivoice.{h,cpp}      Public ABI : opaque ov_context handle, plain C99
                         header in extern "C", consumable from C, C++,
                         Python ctypes, Rust bindgen, Go cgo

tools/
  omnivoice-tts.cpp    CLI : text to WAV (auto / design / clone)
  omnivoice-codec.cpp  CLI : codes <-> WAV
  quantize.cpp         GGUF requantizer
  version.cmake        Embeds the git short hash into the binary

tests/
  debug-tts-cossim.py     Byte-level comparison of every pipeline stage
                          against the PyTorch reference, voice design path
  debug-clone-cossim.py   Same, voice cloning path
  cross-decode.py         Cross check : decode C++ tokens through PyTorch
                          codec and vice versa
  prompt.txt              Long-form English TTS sample
  ref-audio.wav           Voice cloning reference clip
  ref-text.txt            Transcript matching ref-audio.wav
  abi-c.c                 Plain C99 smoke test for the public ABI ; built
                          with -Wall -Werror -pedantic on every build,
                          locks in C consumability and symbol linkage
```

## GGML conventions

### Tensor shape and layout

PyTorch shape `(out, in)` for a Linear weight stores as ggml
`ne[0]=in, ne[1]=out`. The GGUF tensor-shape array is reversed, so
reading `reversed(t.shape)` from gguf-py yields the PyTorch shape
directly.

For PyTorch `Conv1d` weight `(OC, IC, K)`, ggml ne is `(K, IC, OC)`.
The kernel axis is innermost (contiguous in memory).

For PyTorch `ConvTranspose1d` weight `(IC, OC, K)`, the convert-time
permutation to ggml `(IC, K*OC)` rearranges
`(oc*K + k) * IC + ic` so that `ggml_col2im_1d` receives the correct
column matrix.

`ggml_mul_mat(A, B)` : with A.ne[0] = K (must match B.ne[0]),
A.ne[1] = M, B.ne[1] = N, output has ne = (N, M). In PyTorch terms,
A is `(M, K)`, B is `(N, K)`, output is `(M, N)`, which equals
`A @ B^T`.

### Custom GGML ops

Provided by the `ServeurpersoCom/ggml` fork :

`ggml_snake(ctx, x, a, inv_b)` : `y = x + sin^2(a * x) * inv_b`.
F32 / F16 / BF16 input/output. CPU + CUDA + Metal + Vulkan.

`ggml_col2im_1d(ctx, a, s0, oc, p0)` : scatter-add `[K*OC, T_in]`
columns into `[T_out, OC]` signal where
`T_out = (T_in - 1)*s0 + K - 2*p0`. Layout requires k to vary faster
than oc inside the K*OC axis. F32 / F16 / BF16. All backends. The
fork also folds the padding crop into this op via the `p0` parameter,
removing a follow-up `ggml_view` for the typical ConvTranspose1d use
case.

### Backend lifecycle

`backend_init("MOD")` then `backend_sched_new(bp, max_nodes)`. Backend
handles are shared across modules in the same binary, refcounted. The
GPU backend is the default, the CPU backend is kept as a scheduler
fallback.

## Validation

The reference comparison harness is `tests/debug-tts-cossim.py` and
`tests/debug-clone-cossim.py`. They run the same input through the
PyTorch reference (with TF32 disabled, eager attention) and through the
C++ binary, dump each pipeline stage to disk, and report cosine
similarity per stage. Latest run, chunked path, English long-form
prompt :

```
TTS    chunked: Logits cos=1.000000 max 3.5e-04
                Step0 pred_tokens 99.93% (2 FP flips)
                Tokens 1.000000 exact 100.00%
                Audio  0.999991

Clone  chunked: Lf hidden cos=1.000000 max 1.9e-03
                Logits  cos=1.000000
                Step0 pred_tokens 100.00%
                Tokens 1.000000 exact 100.00%
                Audio  0.999989
```

The few Step0 token flips are argmax ties at the FP epsilon (~2e-5
between top1 and top2 at those positions), inherent to the mixed cuBLAS
vs GGML kernel arithmetic. They resorb over the 32 MaskGIT steps so
the final tokens match bit for bit and decoded audio cosine is
> 0.9999.

## Glossary

  RVQ        Residual Vector Quantisation. Stack of codebooks where
              each one quantises the residual from the previous
              codebook reconstruction.

  DAC        Descript Audio Codec. Convolutional encoder/decoder over
              residual VQ codes.

  HuBERT     Hidden-Unit BERT. Transformer encoder pretrained with
              masked acoustic unit prediction. Used here to extract
              semantic embeddings from raw audio.

  Snake      Periodic activation introduced in BigVGAN,
              `y = x + (1/alpha) * sin^2(alpha * x)`. Replaces
              LeakyReLU in the DAC encoder/decoder.

  CFG        Classifier-Free Guidance. The model is run twice
              (conditional and unconditional) and the outputs combined
              as `c + scale * (c - u)` to amplify the conditional
              signal.

  MaskGIT    Masked Generative Image Transformer (Chang et al.,
              arXiv:2202.04200). Iterative non autoregressive decoder
              where masked tokens are progressively unmasked over a
              fixed number of steps, prioritising high-confidence
              positions per step. Originally introduced for image
              generation, adapted here to audio codes.

  Philox     Counter-based PRNG used by PyTorch CUDA. Thread safe and
              skip-ahead friendly, well suited to deterministic
              chunked inference.
