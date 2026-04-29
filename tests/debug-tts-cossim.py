#!/usr/bin/env python3
"""Cossim debug : C++ omnivoice-tts vs Python OmniVoice on voice design.

Inputs (relative to CWD = tests/) :
    prompt.txt       target text fed to both pipelines

Both sides run with :
    instruct=male, language=French, seed=42, F32 weights, no pre or post
    process. Defaults match : num_step=32, guidance_scale=2.0, t_shift=0.1,
    layer_penalty_factor=5.0, position_temperature=5.0, class_temperature=0.0.

Dumps land in cpp/ (C++) and python/ (Python). The script compares each
matching .bin pair via cosine similarity over the f32 payload, plus exact
match rate for tensors that originated as int (mg-tokens). All paths are
relative, no absolute paths anywhere.
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

# Belt and suspenders : disable PyTorch's own TF32 toggles too. Some code
# paths bypass NVIDIA_TF32_OVERRIDE through cudnn or torch internal flags.
torch.backends.cuda.matmul.allow_tf32                             = False
torch.backends.cudnn.allow_tf32                                   = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.set_float32_matmul_precision("highest")

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
    """Write a tensor in the C++ debug.h format :
        [ndim:i32] [shape:i32 x ndim] [data:f32 x numel]
    """
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
    """Inverse of save_dump : returns (data:f32 numpy, shape:tuple)."""
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
    # First call to _prepare_embed_inputs at step 0 returns the input embedding
    # right before layer 0 of the LLM, mirroring the C++ inputs_embeds dump.
    # Also captures the raw input_ids row k=0 for cond and uncond, so any
    # token-level divergence localizes upstream of the embed lookup.
    seen_embed = {"done": False}
    orig_prepare = model._prepare_embed_inputs
    def hooked_prepare(input_ids, audio_mask):
        out = orig_prepare(input_ids, audio_mask)
        if not seen_embed["done"] and out.dim() == 3 and out.shape[0] >= 2:
            cond   = out[0].detach().to(torch.float32).cpu().numpy()
            uncond = out[1].detach().to(torch.float32).cpu().numpy()
            save_dump(os.path.join(dump_dir, "lm-hidden-step0-cond-embed.bin"),   cond)
            save_dump(os.path.join(dump_dir, "lm-hidden-step0-uncond-embed.bin"), uncond)
            # Style and text tokens duplicate across all K codebooks so k=0
            # carries the full sequence for diagnostic. Cast to f32 keeps the
            # debug.h binary format identical on both sides.
            cond_ids   = input_ids[0, 0, :].detach().to(torch.float32).cpu().numpy()
            uncond_ids = input_ids[1, 0, :].detach().to(torch.float32).cpu().numpy()
            save_dump(os.path.join(dump_dir, "prompt-cond-ids.bin"),   cond_ids)
            save_dump(os.path.join(dump_dir, "prompt-uncond-ids.bin"), uncond_ids)
            seen_embed["done"] = True
        return out
    model._prepare_embed_inputs = hooked_prepare

    # Bisection : dump cond and uncond hidden states after a few layers so a
    # mismatch can be localized within the 28 layer Qwen3 stack.
    bisect_layers = [0, 6, 13, 20]
    seen_layers   = {idx: False for idx in bisect_layers}
    def make_layer_hook(layer_idx):
        def hook(module, inputs, output):
            if seen_layers[layer_idx]:
                return
            h = output[0] if isinstance(output, tuple) else output
            if h.dim() == 3 and h.shape[0] >= 2:
                cond   = h[0].detach().to(torch.float32).cpu().numpy()
                uncond = h[1].detach().to(torch.float32).cpu().numpy()
                save_dump(os.path.join(dump_dir, f"lm-hidden-step0-cond-l{layer_idx}.bin"),   cond)
                save_dump(os.path.join(dump_dir, f"lm-hidden-step0-uncond-l{layer_idx}.bin"), uncond)
                seen_layers[layer_idx] = True
        return hook
    for layer_idx in bisect_layers:
        model.llm.layers[layer_idx].register_forward_hook(make_layer_hook(layer_idx))

    # First call to audio_heads corresponds to step 0 of the MaskGIT loop.
    # The input to audio_heads is the final hidden state, shape [B, S, D],
    # mirroring what the C++ side reads back via dump_hidden_dir before the
    # lm_head matmul. We dump cond (b=0) and uncond (b=1) separately.
    seen_hidden = {"done": False}
    def pre_audio_heads(module, inputs):
        if not seen_hidden["done"]:
            h = inputs[0]
            if h.dim() == 3 and h.shape[0] >= 2:
                cond   = h[0].detach().to(torch.float32).cpu().numpy()
                uncond = h[1].detach().to(torch.float32).cpu().numpy()
                save_dump(os.path.join(dump_dir, "lm-hidden-step0-cond.bin"),   cond)
                save_dump(os.path.join(dump_dir, "lm-hidden-step0-uncond.bin"), uncond)
                seen_hidden["done"] = True
    model.audio_heads.register_forward_pre_hook(pre_audio_heads)

    # First call to _predict_tokens_with_scoring corresponds to step 0 of the
    # MaskGIT loop. Capture cond and uncond logits in [K, T, V] layout (squeeze
    # the batch axis to match the C++ dump shape). Replicate the predict math
    # locally so log_probs / pred_tokens / scores can all be dumped from a
    # single source of truth on the Python side.
    seen = {"step0": False, "mg_tokens": False, "audio": False}
    orig_pred = model._predict_tokens_with_scoring
    def hooked_pred(c_logits, u_logits, gen_config):
        pred_tokens, scores = orig_pred(c_logits, u_logits, gen_config)
        if not seen["step0"]:
            if gen_config.guidance_scale != 0:
                c_lp = torch.nn.functional.log_softmax(c_logits, dim=-1)
                u_lp = torch.nn.functional.log_softmax(u_logits, dim=-1)
                log_probs = torch.log_softmax(
                    c_lp + gen_config.guidance_scale * (c_lp - u_lp), dim=-1)
            else:
                log_probs = torch.nn.functional.log_softmax(c_logits, dim=-1)
            log_probs = log_probs.clone()
            log_probs[..., model.config.audio_mask_id] = float("-inf")

            c = c_logits.detach().to(torch.float32).cpu().numpy()
            u = u_logits.detach().to(torch.float32).cpu().numpy()
            if c.ndim == 4:
                c = c[0]
            if u.ndim == 4:
                u = u[0]
            save_dump(os.path.join(dump_dir, "lm-logits-step0-cond.bin"),   c)
            save_dump(os.path.join(dump_dir, "lm-logits-step0-uncond.bin"), u)

            lp_arr = log_probs.detach().to(torch.float32).cpu().numpy()
            if lp_arr.ndim == 4:
                lp_arr = lp_arr[0]
            save_dump(os.path.join(dump_dir, "mg-log-probs-step0.bin"), lp_arr)

            pt_arr = pred_tokens.detach().to(torch.float32).cpu().numpy()
            sc_arr = scores.detach().to(torch.float32).cpu().numpy()
            if pt_arr.ndim == 3:
                pt_arr = pt_arr[0]
            if sc_arr.ndim == 3:
                sc_arr = sc_arr[0]
            save_dump(os.path.join(dump_dir, "mg-pred-tokens-step0.bin"), pt_arr)
            save_dump(os.path.join(dump_dir, "mg-scores-step0.bin"),      sc_arr)
            seen["step0"] = True
        return pred_tokens, scores
    model._predict_tokens_with_scoring = hooked_pred

    orig_generate = model._generate_iterative
    def hooked_generate(task, gen_config):
        out = orig_generate(task, gen_config)
        # out is a list of (K, T_i) long tensors, one per batch item.
        # In chunked mode this hook fires once per chunk; only the first
        # call mirrors the chunk 0 dump on the C++ side.
        if not seen["mg_tokens"]:
            save_dump(os.path.join(dump_dir, "mg-tokens.bin"), out[0])
            seen["mg_tokens"] = True
        return out
    model._generate_iterative = hooked_generate

    orig_decode = model.audio_tokenizer.decode
    def hooked_decode(*args, **kwargs):
        out = orig_decode(*args, **kwargs)
        # The audio tokenizer returns either a tensor or a wrapper holding
        # audio_values shape [B, C, N]. Unwrap and dump the first item mono.
        # Same first-chunk guard as the mg-tokens hook above.
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt",   default="prompt.txt")
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--instruct", default="male, young adult, moderate pitch")
    ap.add_argument("--lang",     default="French")
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--out-cpp",  default="cpp/tts-cpp.wav")
    ap.add_argument("--out-pt",   default="python/tts-python.wav")
    args = ap.parse_args()

    ensure_dir(DUMP_CPP)
    ensure_dir(DUMP_PT)
    os.makedirs(os.path.dirname(args.out_cpp) or ".", exist_ok=True)

    with open(args.prompt, "r", encoding="utf-8") as f:
        text = f.read().strip()
    print(f"[Input] Prompt: {len(text)} chars: {text[:60]}{'...' if len(text) > 60 else ''}")
    print(f"[Input] Instruct: {args.instruct}")
    print(f"[Input] Language: {args.lang}")
    print(f"[Input] Seed: {args.seed}")

    # Python reference path : F32, voice design male, no pre or post process.
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
        instruct=args.instruct,
        duration=args.duration,
    )
    audios = model.generate(**gen_kwargs)
    audio_pt = np.asarray(audios[0], dtype=np.float32)
    sf.write(args.out_pt, audio_pt, 24000, subtype="FLOAT")
    print(f"[Python] Audio: {audio_pt.shape[0]} samples {audio_pt.shape[0] / 24000:.2f}s -> {args.out_pt}")

    # Free the GPU before launching the C++ binary so it has room to load
    # the F32 GGUFs without fighting for VRAM.
    del model
    torch.cuda.empty_cache()

    # C++ path : same text, same instruct, same seed, F32 GGUF weights,
    # dumps under cpp/.
    cmd = [
        BIN,
        "--model",       MODEL_LM,
        "--codec",       MODEL_CDC,
        "--seed",        str(args.seed),
        "--instruct",    args.instruct,
        "--lang",        args.lang,
        "--format",      "wav32",
        "--dump",        DUMP_CPP,
        "--no-fa",
        "-o",            args.out_cpp,
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

    # Cossim in pipeline order: prompt-ids -> embed -> l0 -> l6 -> l13 -> l20
    # -> final -> logits -> tokens -> audio. Cond and uncond on the same line
    # so a drift localizes immediately to the originating stage.
    def pair(name):
        a, _ = load_dump(os.path.join(DUMP_CPP, name))
        b, _ = load_dump(os.path.join(DUMP_PT,  name))
        return a, b

    def ids_exact(a, b):
        n = min(a.size, b.size)
        ai = a.astype(np.int64).ravel()[:n]
        bi = b.astype(np.int64).ravel()[:n]
        diffs = np.where(ai != bi)[0]
        return 100.0 * float(np.mean(ai == bi)), diffs, ai, bi

    ca, cb = pair("prompt-cond-ids.bin")
    ua, ub = pair("prompt-uncond-ids.bin")
    cm, cd, cai, cbi = ids_exact(ca, cb)
    um, ud, uai, ubi = ids_exact(ua, ub)
    print(f"[Cossim] PromptIDs cond exact: {cm:.2f}% uncond exact: {um:.2f}%")
    for s in cd[:20]:
        print(f"[Cossim] PromptIDs cond diff at s={s}: ggml={cai[s]} python={cbi[s]}")
    for s in ud[:20]:
        print(f"[Cossim] PromptIDs uncond diff at s={s}: ggml={uai[s]} python={ubi[s]}")

    stages = [
        ("Embed",  "lm-hidden-step0-{}-embed.bin"),
        ("L0",     "lm-hidden-step0-{}-l0.bin"),
        ("L6",     "lm-hidden-step0-{}-l6.bin"),
        ("L13",    "lm-hidden-step0-{}-l13.bin"),
        ("L20",    "lm-hidden-step0-{}-l20.bin"),
        ("Final",  "lm-hidden-step0-{}.bin"),
        ("Logits", "lm-logits-step0-{}.bin"),
    ]
    def metric(a, b):
        n = min(a.size, b.size)
        af = a.astype(np.float64).ravel()[:n]
        bf = b.astype(np.float64).ravel()[:n]
        d  = np.abs(af - bf)
        nrm_a = float(np.linalg.norm(af))
        nrm_b = float(np.linalg.norm(bf))
        c = float(np.dot(af, bf) / (nrm_a * nrm_b)) if nrm_a > 1e-10 and nrm_b > 1e-10 else 0.0
        return c, float(d.max()), float(d.mean())

    for label, fmt in stages:
        ca, cb = pair(fmt.format("cond"))
        ua, ub = pair(fmt.format("uncond"))
        cc, cmax, cmean = metric(ca, cb)
        uc, umax, umean = metric(ua, ub)
        print(f"[Cossim] {label} cond cos: {cc:.6f} max: {cmax:.4e} mean: {cmean:.4e} uncond cos: {uc:.6f} max: {umax:.4e} mean: {umean:.4e}")

    pa, pb = pair("mg-pred-tokens-step0.bin")
    n = min(pa.size, pb.size)
    ai = pa.astype(np.int64).ravel()[:n]
    bi = pb.astype(np.int64).ravel()[:n]
    diffs = np.where(ai != bi)[0]
    print(f"[Cossim] Step0Tokens exact: {100.0 * float((ai == bi).mean()):.2f}% diffs: {diffs.size}")

    sa, sb = pair("mg-scores-step0.bin")
    sd = np.abs(sa - sb)
    print(f"[Cossim] Step0Scores cos: {cos(sa, sb):.6f} max_abs_diff: {sd.max():.6f} mean_abs_diff: {sd.mean():.6f}")

    la, lb = pair("mg-log-probs-step0.bin")
    finite = np.isfinite(la) & np.isfinite(lb)
    laf = la[finite]; lbf = lb[finite]
    ld = np.abs(laf - lbf)
    print(f"[Cossim] Step0LogProbs cos: {cos(laf, lbf):.6f} max_abs_diff: {ld.max():.6f} mean_abs_diff: {ld.mean():.6f}")

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
