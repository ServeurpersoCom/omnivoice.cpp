#!/usr/bin/env python3
"""Cossim debug : C++ omnivoice-tts vs Python OmniVoice on voice cloning.

Inputs (relative to CWD = tests/) :
    ../examples/prompt.txt       target text fed to both pipelines
    ../examples/freeman.txt      transcript of the cloning reference
    ../examples/freeman.wav      cloning reference audio, any rate, any layout

Both sides run with seed=42, F32 weights, language=French, no pre or post
process. The reference audio is resampled to 24 kHz mono inside both
pipelines.

Dumps land in cpp/ (C++) and python/ (Python) and are compared pair by
pair. All paths are relative.
"""

import argparse
import os
import struct
import subprocess
import sys

# Strict F32 matmul on both sides. NVIDIA_TF32_OVERRIDE=0 forces full FP32
# mantissa in cuBLAS for both PyTorch and the C++ child via inheritance.
# Must be set BEFORE torch imports so the cuBLAS handle reads it on init.
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

# Belt and suspenders : disable PyTorch's own TF32 toggles too. Some code
# paths bypass NVIDIA_TF32_OVERRIDE through cudnn or torch internal flags.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32       = False

from omnivoice import OmniVoice
from omnivoice.utils.common import fix_random_seed

BIN        = "../build/omnivoice-tts"
MODEL_LM   = "../models/omnivoice-base-F32.gguf"
MODEL_CDC  = "../models/omnivoice-tokenizer-F32.gguf"
CKPT       = "../checkpoints/OmniVoice"
DUMP_CPP   = "cpp"
DUMP_PT    = "python"

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_dump(path, data):
    if isinstance(data, torch.Tensor):
        data = data.detach().to(torch.float32).cpu().numpy()
    data  = np.ascontiguousarray(data.astype(np.float32))
    shape = data.shape
    with open(path, "wb") as f:
        f.write(struct.pack("i", len(shape)))
        for s in shape:
            f.write(struct.pack("i", s))
        f.write(data.tobytes())

def load_dump(path):
    raw   = np.fromfile(path, dtype=np.uint8)
    ndim  = int(np.frombuffer(raw[0:4], dtype=np.int32)[0])
    shape = tuple(int(x) for x in np.frombuffer(raw[4:4 + 4 * ndim], dtype=np.int32))
    body  = np.frombuffer(raw[4 + 4 * ndim:], dtype=np.float32)
    return body.reshape(shape), shape

def cos(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    # log_probs contains -inf at the audio_mask_id slot. Mask these symmetric
    # positions to 0 on both sides so they cancel out of the inner product.
    bad = ~(np.isfinite(a) & np.isfinite(b))
    if bad.any():
        a = np.where(bad, 0.0, a)
        b = np.where(bad, 0.0, b)
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / d) if d > 1e-10 else 0.0

def stft_cos(a, b, win=2048, hop=512):
    # STFT magnitude cosine. Drops phase, so a constant time shift between
    # the two waveforms does not collapse the score. The plain cos() on
    # raw samples falls to ~0 the moment chunks land a few samples apart.
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    window = np.hanning(win)
    frames = (n - win) // hop + 1
    if frames <= 0:
        return 0.0
    sa = np.zeros((frames, win // 2 + 1))
    sb = np.zeros((frames, win // 2 + 1))
    for i in range(frames):
        s = i * hop
        sa[i] = np.abs(np.fft.rfft(a[s:s + win] * window))
        sb[i] = np.abs(np.fft.rfft(b[s:s + win] * window))
    return cos(sa.ravel(), sb.ravel())

def install_hooks(model, dump_dir):
    # Capture the raw input_ids row k=0 for cond and uncond at the first
    # forward pass. Lets a divergence in prompt construction (style, text,
    # ref_audio tokens) localize before the LM runs.
    seen_embed = {"done": False}
    orig_prepare = model._prepare_embed_inputs
    def hooked_prepare(input_ids, audio_mask):
        out = orig_prepare(input_ids, audio_mask)
        if not seen_embed["done"] and input_ids.dim() == 3 and input_ids.shape[0] >= 2:
            cond_ids   = input_ids[0, 0, :].detach().to(torch.float32).cpu().numpy()
            uncond_ids = input_ids[1, 0, :].detach().to(torch.float32).cpu().numpy()
            save_dump(os.path.join(dump_dir, "prompt-cond-ids.bin"),   cond_ids)
            save_dump(os.path.join(dump_dir, "prompt-uncond-ids.bin"), uncond_ids)
            seen_embed["done"] = True
        return out
    model._prepare_embed_inputs = hooked_prepare

    seen = {"step0": False, "mg_tokens": False, "audio": False}
    orig_pred = model._predict_tokens_with_scoring
    def hooked_pred(c_logits, u_logits, gen_config):
        is_first = not seen["step0"]
        if is_first:
            c = c_logits.detach().to(torch.float32).cpu().numpy()
            u = u_logits.detach().to(torch.float32).cpu().numpy()
            if c.ndim == 4:
                c = c[0]
            if u.ndim == 4:
                u = u[0]
            save_dump(os.path.join(dump_dir, "lm-logits-step0-cond.bin"),   c)
            save_dump(os.path.join(dump_dir, "lm-logits-step0-uncond.bin"), u)
        pred_tokens, scores = orig_pred(c_logits, u_logits, gen_config)
        if is_first:
            # Reconstruct log_probs with the same op sequence as the Python
            # function so the dump is the actual log_probs the model used.
            with torch.no_grad():
                if gen_config.guidance_scale != 0:
                    cl = F.log_softmax(c_logits, dim=-1)
                    ul = F.log_softmax(u_logits, dim=-1)
                    lp = torch.log_softmax(cl + gen_config.guidance_scale * (cl - ul), dim=-1)
                else:
                    lp = F.log_softmax(c_logits, dim=-1)
                lp[..., model.config.audio_mask_id] = -float("inf")
            lp_arr = lp.detach().to(torch.float32).cpu().numpy()
            if lp_arr.ndim == 4:
                lp_arr = lp_arr[0]
            save_dump(os.path.join(dump_dir, "mg-log-probs-step0.bin"), lp_arr)

            pt_arr = pred_tokens.detach().to(torch.float32).cpu().numpy()
            if pt_arr.ndim == 3:
                pt_arr = pt_arr[0]
            save_dump(os.path.join(dump_dir, "mg-pred-tokens-step0.bin"), pt_arr)

            sc_arr = scores.detach().to(torch.float32).cpu().numpy()
            if sc_arr.ndim == 3:
                sc_arr = sc_arr[0]
            save_dump(os.path.join(dump_dir, "mg-scores-step0.bin"), sc_arr)
            seen["step0"] = True
        return pred_tokens, scores
    model._predict_tokens_with_scoring = hooked_pred

    orig_generate = model._generate_iterative
    def hooked_generate(task, gen_config):
        out = orig_generate(task, gen_config)
        if not seen["mg_tokens"]:
            save_dump(os.path.join(dump_dir, "mg-tokens.bin"), out[0])
            seen["mg_tokens"] = True
        return out
    model._generate_iterative = hooked_generate

    orig_decode = model.audio_tokenizer.decode
    def hooked_decode(*args, **kwargs):
        out = orig_decode(*args, **kwargs)
        if seen["audio"]:
            return out
        wav = getattr(out, "audio_values", out)
        if isinstance(wav, torch.Tensor):
            arr = wav.detach().to(torch.float32).cpu().numpy()
        else:
            arr = np.asarray(wav, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[0, 0]
        elif arr.ndim == 2:
            arr = arr[0]
        save_dump(os.path.join(dump_dir, "output-audio.bin"), arr)
        seen["audio"] = True
        return out
    model.audio_tokenizer.decode = hooked_decode

    # Capture the full reference audio code matrix [K, T_ref] before it goes
    # into the prompt, so a divergence in the codec encoder is visible at the
    # source rather than only via prompt-cond-ids row k=0.
    orig_encode = model.audio_tokenizer.encode
    seen_enc = {"done": False}
    def hooked_encode(*args, **kwargs):
        out = orig_encode(*args, **kwargs)
        if not seen_enc["done"]:
            codes = out.audio_codes if hasattr(out, "audio_codes") else out
            if isinstance(codes, torch.Tensor):
                arr = codes.detach().to(torch.float32).cpu().numpy()
                if arr.ndim == 3:
                    arr = arr[0]
                save_dump(os.path.join(dump_dir, "ref-audio-codes.bin"), arr)
                seen_enc["done"] = True
        return out
    model.audio_tokenizer.encode = hooked_encode

    # Capture the 16 kHz post resample audio and the HuBERT semantic features
    # at the boundary inside the audio tokenizer. Lets us bisect the codec
    # encoder : raw 24k -> resample 16k -> hubert -> sem_enc -> RVQ -> codes.
    orig_extract = model.audio_tokenizer._extract_semantic_features
    seen_sem = {"done": False}
    def hooked_extract(input_values):
        if not seen_sem["done"]:
            cfg = model.audio_tokenizer.config
            if cfg.sample_rate != cfg.semantic_sample_rate:
                v16 = torchaudio.functional.resample(input_values, cfg.sample_rate, cfg.semantic_sample_rate)
            else:
                v16 = input_values
            arr16 = v16[:, 0, :].squeeze(0).detach().to(torch.float32).cpu().numpy()
            save_dump(os.path.join(dump_dir, "ref-audio-16k.bin"), arr16)
        out = orig_extract(input_values)
        if not seen_sem["done"]:
            feat = out.squeeze(0).detach().to(torch.float32).cpu().numpy()
            save_dump(os.path.join(dump_dir, "ref-hubert-features.bin"), feat)
            seen_sem["done"] = True
        return out
    model.audio_tokenizer._extract_semantic_features = hooked_extract

    # Bisect HuBERT internals. The C++ counterpart marks the same 9 stages as
    # graph outputs in pipeline_codec_hubert_features_test and dumps them post
    # compute. Names match exactly so the Python compare loop can pair them.
    sm = model.audio_tokenizer.semantic_model
    seen_hub = {"feat": False, "proj_ln": False, "proj": False, "init": False,
                "l0":   False, "l5":      False, "l7":   False, "l9":   False, "l11":  False}

    def hub_hook_feat(module, inputs, output):
        # HF feature_extractor returns (B, C=512, T_feat). Squeeze gives
        # (C, T_feat) which matches the C++ ne=(T, C) tensor whose linear
        # buffer is laid out slow-first as (C, T). No transpose needed.
        if seen_hub["feat"]:
            return
        arr = output[0].detach().to(torch.float32).cpu().numpy()
        save_dump(os.path.join(dump_dir, "hubert-feat-extract.bin"), arr)
        seen_hub["feat"] = True
    sm.feature_extractor.register_forward_hook(hub_hook_feat)

    # feature_projection internal split : layer_norm first, then projection
    # Linear. The C++ side dumps the post LN tensor before the Linear, so we
    # mirror with a forward hook on the LN sub-module.
    def hub_hook_proj_ln(module, inputs, output):
        if seen_hub["proj_ln"]:
            return
        out_t = output[0] if isinstance(output, tuple) else output
        arr = out_t[0].detach().to(torch.float32).cpu().numpy()
        save_dump(os.path.join(dump_dir, "hubert-feat-proj-ln.bin"), arr)
        seen_hub["proj_ln"] = True
    sm.feature_projection.layer_norm.register_forward_hook(hub_hook_proj_ln)

    def hub_hook_proj(module, inputs, output):
        # feature_projection returns (B, T_feat, 768) already T-first.
        if seen_hub["proj"]:
            return
        out_t = output[0] if isinstance(output, tuple) else output
        arr = out_t[0].detach().to(torch.float32).cpu().numpy()
        save_dump(os.path.join(dump_dir, "hubert-feat-proj.bin"), arr)
        seen_hub["proj"] = True
    sm.feature_projection.register_forward_hook(hub_hook_proj)

    # encoder.layer_norm is the LN that follows pos_conv_embed add. Its output
    # equals the C++ enc_init output : x + pos_conv(x) -> LN.
    def hub_hook_init(module, inputs, output):
        if seen_hub["init"]:
            return
        arr = output[0].detach().to(torch.float32).cpu().numpy()
        save_dump(os.path.join(dump_dir, "hubert-enc-init.bin"), arr)
        seen_hub["init"] = True
    sm.encoder.layer_norm.register_forward_hook(hub_hook_init)

    # Mid stack taps : layer 0, 5, 7, 9, 11. Mirror the C++ states[1], [6],
    # [8], [10], [12] indexing (states[0] is enc-init dumped above). l7 and
    # l9 give the resolution to localize the explosion seen between l5 and l11.
    def make_layer_tap(key, fname):
        def hook(module, inputs, output):
            if seen_hub[key]:
                return
            h = output[0] if isinstance(output, tuple) else output
            arr = h[0].detach().to(torch.float32).cpu().numpy()
            save_dump(os.path.join(dump_dir, fname), arr)
            seen_hub[key] = True
        return hook
    sm.encoder.layers[0].register_forward_hook(make_layer_tap("l0",  "hubert-l0.bin"))
    sm.encoder.layers[5].register_forward_hook(make_layer_tap("l5",  "hubert-l5.bin"))
    sm.encoder.layers[7].register_forward_hook(make_layer_tap("l7",  "hubert-l7.bin"))
    sm.encoder.layers[9].register_forward_hook(make_layer_tap("l9",  "hubert-l9.bin"))
    sm.encoder.layers[11].register_forward_hook(make_layer_tap("l11", "hubert-l11.bin"))

    # Bisect the LM forward by dumping per layer hidden states at step 0.
    # The C++ pipeline dumps cond and uncond at layers 0, 6, 13, 20 and the
    # final norm. Mirror that exactly so a per layer max_abs_diff comparison
    # localizes where the drift starts.
    seen_lm = {"done": False}
    def make_layer_hook(name):
        def hook(module, inputs, output):
            if seen_lm["done"]:
                return
            h = output[0] if isinstance(output, tuple) else output
            if not isinstance(h, torch.Tensor) or h.ndim < 3:
                return
            cond_arr = h[0].detach().to(torch.float32).cpu().numpy()
            save_dump(os.path.join(dump_dir, f"lm-hidden-step0-cond-{name}.bin" if name else "lm-hidden-step0-cond.bin"), cond_arr)
            if h.shape[0] >= 2:
                unc_arr = h[1].detach().to(torch.float32).cpu().numpy()
                save_dump(os.path.join(dump_dir, f"lm-hidden-step0-uncond-{name}.bin" if name else "lm-hidden-step0-uncond.bin"), unc_arr)
        return hook
    lm_hook_handles = []
    for idx, name in [(0, "l0"), (1, "l1"), (2, "l2"), (3, "l3"), (4, "l4"), (5, "l5"), (6, "l6"),
                      (13, "l13"), (14, "l14"), (15, "l15"), (16, "l16"), (17, "l17"), (18, "l18"),
                      (19, "l19"), (20, "l20")]:
        if idx < len(model.llm.layers):
            lm_hook_handles.append(model.llm.layers[idx].register_forward_hook(make_layer_hook(name)))
    lm_hook_handles.append(model.llm.norm.register_forward_hook(make_layer_hook("")))
    # Trip seen_lm["done"] right after the first LM forward by hooking the
    # outermost forward via the existing _predict_tokens_with_scoring path
    # already gated on step0. We instead use an extra one-shot hook on the
    # last collected layer (norm) to flip the flag after its dump fires.
    def flip_done(module, inputs, output):
        seen_lm["done"] = True
    lm_hook_handles.append(model.llm.norm.register_forward_hook(flip_done))

    # Per-layer hooks above already cover layers 0-6, 13-20 and final norm.
    # Add the embed boundary on top and a sub-module bisect inside layer 1 to
    # locate where the L0 -> L1 jump (15x in one layer) originates.
    orig_prepare_embed = model._prepare_embed_inputs
    def hooked_prepare_embed(input_ids, audio_mask):
        out = orig_prepare_embed(input_ids, audio_mask)
        if not seen_lm["done"]:
            arr = out.detach().to(torch.float32).cpu().numpy()
            save_dump(os.path.join(dump_dir, "lm-hidden-step0-cond-embed.bin"), arr[0])
            if arr.shape[0] >= 2:
                save_dump(os.path.join(dump_dir, "lm-hidden-step0-uncond-embed.bin"), arr[1])
        return out
    model._prepare_embed_inputs = hooked_prepare_embed

    # Layer 1 sub-module bisect : dump after input_layernorm, after self_attn
    # (pre-residual), after post_attn_layernorm and after mlp (pre-residual).
    # Suffixes : -l1-norm1, -l1-attn, -l1-norm2, -l1-mlp.
    def make_sub_hook(suffix):
        def hook(module, inputs, output):
            if seen_lm["done"]:
                return
            h = output[0] if isinstance(output, tuple) else output
            if not isinstance(h, torch.Tensor) or h.ndim < 3:
                return
            arr = h.detach().to(torch.float32).cpu().numpy()
            save_dump(os.path.join(dump_dir, f"lm-hidden-step0-cond-l1-{suffix}.bin"), arr[0])
            if arr.shape[0] >= 2:
                save_dump(os.path.join(dump_dir, f"lm-hidden-step0-uncond-l1-{suffix}.bin"), arr[1])
        return hook
    if len(model.llm.layers) > 1:
        l1 = model.llm.layers[1]
        lm_hook_handles.append(l1.input_layernorm.register_forward_hook(make_sub_hook("norm1")))
        lm_hook_handles.append(l1.self_attn.register_forward_hook(make_sub_hook("attn")))
        lm_hook_handles.append(l1.post_attention_layernorm.register_forward_hook(make_sub_hook("norm2")))
        lm_hook_handles.append(l1.mlp.register_forward_hook(make_sub_hook("mlp")))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt",    default="../examples/prompt.txt")
    ap.add_argument("--ref-text",  default="../examples/freeman.txt")
    ap.add_argument("--ref-audio", default="../examples/freeman.wav")
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--lang",      default="English")
    ap.add_argument("--duration",  type=float, default=None)
    ap.add_argument("--out-cpp",   default="cpp/clone-cpp.wav")
    ap.add_argument("--out-pt",    default="python/clone-python.wav")
    args = ap.parse_args()

    ensure_dir(DUMP_CPP)
    ensure_dir(DUMP_PT)
    os.makedirs(os.path.dirname(args.out_cpp) or ".", exist_ok=True)

    with open(args.prompt, "r", encoding="utf-8") as f:
        text = f.read().strip()
    with open(args.ref_text, "r", encoding="utf-8") as f:
        ref_text = f.read().strip()
    print(f"[Input] Prompt: {len(text)} chars: {text[:60]}{'...' if len(text) > 60 else ''}")
    print(f"[Input] RefText: {len(ref_text)} chars: {ref_text[:60]}{'...' if len(ref_text) > 60 else ''}")
    print(f"[Input] RefWav: {args.ref_audio}")
    print(f"[Input] Language: {args.lang}")
    print(f"[Input] Seed: {args.seed}")

    fix_random_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = OmniVoice.from_pretrained(
        CKPT,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).to(device).eval()
    install_hooks(model, DUMP_PT)
    gen_kwargs = dict(
        text=text,
        language=args.lang,
        ref_text=ref_text,
        ref_audio=args.ref_audio,
        duration=args.duration,
    )
    audios = model.generate(**gen_kwargs)
    audio_pt = np.asarray(audios[0], dtype=np.float32)
    sf.write(args.out_pt, audio_pt, 24000, subtype="FLOAT")
    print(f"[Python] Audio: {audio_pt.shape[0]} samples {audio_pt.shape[0] / 24000:.2f}s -> {args.out_pt}")

    del model
    torch.cuda.empty_cache()

    # Strict F32 conformance: disable FlashAttention (eager attention) and
    # force strict F32 cuBLAS to match the Python side with allow_tf32=False
    # set at the top of this file. Both pipelines encode the reference audio
    # independently so every encoder stage gets compared frontally.
    cmd = [
        BIN,
        "--model",         MODEL_LM,
        "--codec",         MODEL_CDC,
        "--seed",          str(args.seed),
        "--ref-wav",       args.ref_audio,
        "--ref-text",      args.ref_text,
        "--lang",          args.lang,
        "--format",        "wav32",
        "--dump",          DUMP_CPP,
        "-o",              args.out_cpp,
        "--no-fa",
    ]
    if args.duration:
        cmd += ["--duration", str(args.duration)]
    print(f"[GGML] Cmd: {' '.join(cmd)}")
    r = subprocess.run(cmd, input=text, text=True)
    if r.returncode != 0:
        sys.exit(r.returncode)
    audio_cpp, sr = sf.read(args.out_cpp)
    if audio_cpp.ndim > 1:
        audio_cpp = audio_cpp[:, 0]
    audio_cpp = audio_cpp.astype(np.float32)
    print(f"[GGML] Audio: {audio_cpp.shape[0]} samples {sr} Hz {audio_cpp.shape[0] / sr:.2f}s -> {args.out_cpp}")

    # Cossim in pipeline order: prompt-ids -> logits -> tokens -> audio.
    # Cond and uncond on the same line so a drift localizes to the originating
    # stage. PromptIDs split into zones (style, text, ref-audio, target) when
    # the lengths can be inferred from the prompt log.
    def pair(name):
        a, _ = load_dump(os.path.join(DUMP_CPP, name))
        b, _ = load_dump(os.path.join(DUMP_PT,  name))
        return a, b

    ca, cb = pair("prompt-cond-ids.bin")
    ua, ub = pair("prompt-uncond-ids.bin")
    n = min(ca.size, cb.size)
    cai = ca.astype(np.int64).ravel()[:n]
    cbi = cb.astype(np.int64).ravel()[:n]
    cdiffs = np.where(cai != cbi)[0]
    n2 = min(ua.size, ub.size)
    uai = ua.astype(np.int64).ravel()[:n2]
    ubi = ub.astype(np.int64).ravel()[:n2]
    udiffs = np.where(uai != ubi)[0]
    print(f"[Cossim] PromptIDs cond exact: {100.0 * (1 - cdiffs.size / max(n, 1)):.2f}% diffs: {cdiffs.size} uncond exact: {100.0 * (1 - udiffs.size / max(n2, 1)):.2f}% diffs: {udiffs.size}")
    if cdiffs.size > 0:
        print(f"[Cossim] PromptIDs cond first diff range: s={cdiffs[0]}..{cdiffs[-1]}")
    if udiffs.size > 0:
        print(f"[Cossim] PromptIDs uncond first diff range: s={udiffs[0]}..{udiffs[-1]}")

    def maxabs(a, b):
        a = a.astype(np.float64).ravel()
        b = b.astype(np.float64).ravel()
        n = min(len(a), len(b))
        if not n:
            return 0.0
        a, b = a[:n], b[:n]
        # log_probs has -inf at the audio_mask_id slot on both sides. Drop
        # positions that are non finite on either side so the subtract is safe.
        keep = np.isfinite(a) & np.isfinite(b)
        if not keep.all():
            a = a[keep]
            b = b[keep]
        return float(np.max(np.abs(a - b))) if a.size else 0.0

    ra, rb = pair("ref-audio-16k.bin")
    print(f"[Cossim] RefAudio16k max_abs_diff: {maxabs(ra, rb):.3e} cossim: {cos(ra, rb):.6f} samples: {min(ra.size, rb.size)}")

    fa, fb = pair("ref-hubert-features.bin")
    print(f"[Cossim] HuBERT-features cossim: {cos(fa, fb):.6f} max_abs_diff: {maxabs(fa, fb):.3e} shape_cpp: {fa.shape} shape_pt: {fb.shape}")

    # HuBERT bisect taps in pipeline order : feat-extract (post 7 conv1d) ->
    # feat-proj-ln (post LN, pre Linear) -> feat-proj (post 512 -> 768 Linear)
    # -> enc-init (post pos_conv add + LN) -> l0 -> l5 -> l7 -> l9 -> l11. The
    # downsample 2x and the 13-state mean that follow are already covered by
    # HuBERT-features above.
    for tap in ["hubert-feat-extract", "hubert-feat-proj-ln", "hubert-feat-proj",
                "hubert-enc-init", "hubert-l0", "hubert-l5", "hubert-l7", "hubert-l9", "hubert-l11"]:
        ta, tb = pair(tap + ".bin")
        print(f"[Cossim] {tap} cossim: {cos(ta, tb):.6f} max_abs_diff: {maxabs(ta, tb):.3e} shape_cpp: {ta.shape} shape_pt: {tb.shape}")

    ka, kb = pair("ref-audio-codes.bin")
    n_k = min(ka.size, kb.size)
    kai = ka.astype(np.int64).ravel()[:n_k]
    kbi = kb.astype(np.int64).ravel()[:n_k]
    kdiffs = np.where(kai != kbi)[0]
    print(f"[Cossim] RefAudioCodes exact: {100.0 * (1 - kdiffs.size / max(n_k, 1)):.2f}% diffs: {kdiffs.size} shape_cpp: {ka.shape} shape_pt: {kb.shape}")
    if ka.ndim == 2 and kb.ndim == 2 and ka.shape == kb.shape:
        per_k = []
        for k in range(ka.shape[0]):
            d = int(np.sum(ka[k].astype(np.int64) != kb[k].astype(np.int64)))
            per_k.append(f"k{k}={100.0 * (1 - d / ka.shape[1]):.1f}%")
        print(f"[Cossim] RefAudioCodes per-codebook: {' '.join(per_k)}")

    # Per layer LM hidden state drift bisect. C++ dumps cond and uncond at
    # 4 mid layers and the final norm. Pairs only print if both files exist.
    for layer_name, label in [("l0", "L0"),
                              ("l1-norm1", "L1.norm1"), ("l1-attn", "L1.attn"),
                              ("l1-norm2", "L1.norm2"), ("l1-mlp", "L1.mlp"),
                              ("l1", "L1"), ("l2", "L2"), ("l3", "L3"), ("l4", "L4"), ("l5", "L5"), ("l6", "L6"),
                              ("l13", "L13"), ("l14", "L14"), ("l15", "L15"), ("l16", "L16"), ("l17", "L17"), ("l18", "L18"),
                              ("l19", "L19"), ("l20", "L20"), ("", "Lf")]:
        suffix = f"-{layer_name}" if layer_name else ""
        cn = f"lm-hidden-step0-cond{suffix}.bin"
        un = f"lm-hidden-step0-uncond{suffix}.bin"
        try:
            ca, cb = pair(cn)
            ua, ub = pair(un)
        except FileNotFoundError:
            continue
        # Trim cpp shape (which is c_len) into the larger pt shape (max_c_len)
        nc = min(ca.size, cb.size)
        nu = min(ua.size, ub.size)
        cad = ca.ravel()[:nc].astype(np.float64)
        cbd = cb.ravel()[:nc].astype(np.float64)
        uad = ua.ravel()[:nu].astype(np.float64)
        ubd = ub.ravel()[:nu].astype(np.float64)
        max_c = float(np.max(np.abs(cad - cbd))) if nc else 0.0
        max_u = float(np.max(np.abs(uad - ubd))) if nu else 0.0
        cos_c = cos(cad, cbd)
        cos_u = cos(uad, ubd)
        print(f"[Cossim] {label} hidden cond: cos={cos_c:.6f} max_abs={max_c:.3e}  uncond: cos={cos_u:.6f} max_abs={max_u:.3e}")

    ca, cb = pair("lm-logits-step0-cond.bin")
    ua, ub = pair("lm-logits-step0-uncond.bin")
    print(f"[Cossim] Logits cond: {cos(ca, cb):.6f} uncond: {cos(ua, ub):.6f}")

    la, lb = pair("mg-log-probs-step0.bin")
    print(f"[Cossim] Step0 log_probs cossim: {cos(la, lb):.6f} max_abs_diff: {maxabs(la, lb):.3e}")

    pa, pb = pair("mg-pred-tokens-step0.bin")
    n_p = min(pa.size, pb.size)
    pai = pa.astype(np.int64).ravel()[:n_p]
    pbi = pb.astype(np.int64).ravel()[:n_p]
    pdiffs = np.where(pai != pbi)[0]
    print(f"[Cossim] Step0 pred_tokens exact: {100.0 * (1 - pdiffs.size / max(n_p, 1)):.2f}% diffs: {pdiffs.size}/{n_p}")

    sa, sb = pair("mg-scores-step0.bin")
    print(f"[Cossim] Step0 scores cossim: {cos(sa, sb):.6f} max_abs_diff: {maxabs(sa, sb):.3e}")

    ta, tb = pair("mg-tokens.bin")
    n = min(ta.size, tb.size)
    ai = ta.astype(np.int64).ravel()[:n]
    bi = tb.astype(np.int64).ravel()[:n]
    print(f"[Cossim] Tokens: {cos(ta, tb):.6f} exact: {100.0 * float(np.mean(ai == bi)):.2f}%")

    aa, ab = pair("output-audio.bin")
    print(f"[Cossim] Audio: {cos(aa, ab):.6f}")

    n = min(audio_cpp.size, audio_pt.size)
    print(f"[Cossim] WAV stft_cos: {stft_cos(audio_cpp[:n], audio_pt[:n]):.6f} samples: {n}")

if __name__ == "__main__":
    main()
