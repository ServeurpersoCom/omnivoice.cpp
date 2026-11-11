# Experimental Intel Mac Radeon recipe

This is an explicit source-build workaround, not a claim that every Intel Mac
or Intel integrated GPU is supported. It was exercised on an AMD Radeon Pro
5500M with 8 GB VRAM, macOS 15.6.1, with Q8_0 model and tokenizer weights.
No recordings, model weights, credentials or generated audio are distributed.

## Build and use

1. Install Xcode command line tools, Git and CMake, then clone this repository
   with its submodules.
2. Run `./build-intel-metal.sh`. It applies the bundled GGML compatibility
   patch only when it applies cleanly (or is already applied), builds Metal,
   and runs a model-free unaligned-transfer regression test.
3. Use your own model files and a recording you have permission to clone:

```sh
printf '%s\n' 'Hello. This is a local speech test.' | \
  OV_CODEC_DEVICE=MTL0 OV_CODEC_SERIAL=1 \
  ./build-intel-metal/omnivoice-tts \
  --model /path/to/omnivoice-base-Q8_0.gguf \
  --codec /path/to/omnivoice-tokenizer-Q8_0.gguf \
  --device MTL0 --no-fa --steps 16 \
  --ref-wav /path/to/reference.wav --ref-text /path/to/reference.txt \
  -o result.wav
```

The transcript must match the reference audio. Model and tokenizer terms are
separate from the source-code licence; review those terms before downloading.
Check the initialization log to confirm `MTL0` identifies the intended GPU.
Do not set `GGML_METAL_CONCURRENCY_DISABLE` globally: the codec context alone
is serialized, preserving parallel execution within the speech-model graph.
The refinement steps themselves remain sequential.

## Why the patch is needed

On a discrete Metal device, upload/readback used
`newBufferWithBytesNoCopy` on caller-owned host pointers. Those pointers are
not required to be page-aligned. The upload path now creates an owned shared
buffer from the bytes. Readback uses an owned shared buffer and copies into
the caller's destination after GPU completion. Unified-memory transfers keep
their existing path.

Separately, concurrent Metal execution produced invalid codec output on the
tested Radeon. An optional independent codec backend and per-context
concurrency setter let the codec run serialized while the speech model keeps
its existing context. This is a workaround, not a root-cause fix for the
codec scheduling problem. The environment opt-in leaves default behaviour
unchanged, including on Apple Silicon, Linux and Windows.

`patches/ggml-intel-metal.patch` applies to the pinned GGML submodule
`210138933b22221ced02943b7b6e4308610ab530`. It is included as an explicit
prerequisite for this experimental build, rather than pointing the submodule
at an unpublished commit. It still needs separate upstream GGML review.
The concurrency setter must be called during initialization, before any graph
is submitted; it is not a runtime thread-safe toggle.

## Validation and limits

- The model-free transfer test exercises unaligned upload, readback and a
  nonzero tensor offset on the actual Metal device. The minimal test also
  passed before patching on this run, so it is a transfer correctness check,
  not a reliable reproduction of the earlier model-loading assertion.
- Earlier local model checks compared CPU and serialized Metal decoder output
  with waveform cosine similarity above 0.999999; non-serialized Metal output
  was rejected for corruption.
- Matched full-pipeline tests used Q8_0, 16 steps and a fixed seed. The
  all-Metal path repeated deterministically, produced finite unclipped audio,
  and local speech recognition recovered the intended English and Urdu text.
- These are single-device observations, not a cross-platform certification.
  Apple Silicon and other Radeon devices still need independent testing.
- Full GPU assignment is not a promise of real-time speech generation. Keep
  the codec on CPU with `OV_CODEC_DEVICE=CPU` if this workaround is unsuitable.

For rollback, use an ordinary build directory and unset `OV_CODEC_DEVICE` and
`OV_CODEC_SERIAL`. The experimental script does not replace an installed app.
