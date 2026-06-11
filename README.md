# omnivoice.cpp

Local AI text-to-speech with voice cloning and voice design, powered
by GGML. C++17 port of OmniVoice (k2-fsa/OmniVoice). 646 languages,
24 kHz mono output, runs on CPU, CUDA, ROCm, Metal, Vulkan.

## Features

- Voice cloning from a reference WAV plus its transcript
- Voice design via attribute keywords (gender, age, pitch, style,
  volume, emotion)
- Auto voice with consistent speaker identity across long inputs
- Long-form synthesis with punctuation-aware text chunking, voice
  prompt promotion, cross-fade and pydub-strict silence removal
- Bit deterministic generation in greedy mode, seedable Philox PRNG
  for stochastic sampling
- Q8_0 quantisation of the 612 M parameter Qwen3 backbone
- Two CLI tools : `omnivoice-tts` (text -> WAV) and `omnivoice-codec`
  (WAV <-> RVQ codes)

## Build

```
git clone --recurse-submodules https://github.com/ServeurpersoCom/omnivoice.cpp.git
cd omnivoice.cpp
./buildcuda.sh                   # NVIDIA GPU
./buildvulkan.sh                 # AMD/Intel GPU (Vulkan)
./buildcpu.sh                    # CPU only
./buildall.sh                    # all backends, runtime DL loading
NVCC_CCBIN=g++-13 ./buildcuda.sh # rolling release distros (Arch w/ GCC 16, etc.)
```

## Model conversion

Pre-converted GGUFs are available on Hugging Face :

  https://huggingface.co/Serveurperso/OmniVoice-GGUF

Drop them in `models/` and skip to the quick start. To convert from
the original checkpoint :

```
./checkpoints.sh      # hf download k2-fsa/OmniVoice -> checkpoints/
./convert.py          # 2 GGUFs in BF16 -> models/
./quantize.sh         # base LM Q8_0 (tokenizer stays at native dtype)
```

## Quick start

```
echo "Hello world." | ./build/omnivoice-tts \
    --model models/omnivoice-base-Q8_0.gguf \
    --codec models/omnivoice-tokenizer-F32.gguf \
    --lang English -o hello.wav
```

Voice cloning :

```
./build/omnivoice-tts \
    --model models/omnivoice-base-Q8_0.gguf \
    --codec models/omnivoice-tokenizer-F32.gguf \
    --ref-wav ref.wav --ref-text ref.txt \
    --lang English -o out.wav < prompt.txt
```

Pre-encoded reference (`clone.sh`): `omnivoice-codec` encodes a reference WAV
into a compact `.rvq` latent (it applies the exact TTS reference
preprocessing, so the codes are bit-identical to the `--ref-wav` path).
Passing it via `--ref-rvq` skips the codec encode on every synthesis:

```
build/omnivoice-codec --model models/omnivoice-tokenizer-F32.gguf -i ref.wav
build/omnivoice-tts \
    --model models/omnivoice-base-Q8_0.gguf \
    --codec models/omnivoice-tokenizer-F32.gguf \
    --ref-rvq ref.rvq --ref-text ref.txt \
    --lang English -o out.wav < prompt.txt
```

## Embedding the library

The CLI tools are thin wrappers over a public ABI. Single-header,
single-name-prefix, plain C linkage so that C, C++, Python ctypes,
Rust bindgen and Go cgo all consume it the same way.

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
ov_synthesize(ov, &params, &audio);
/* audio.samples, audio.n_samples, audio.sample_rate, audio.channels */
ov_audio_free(&audio);
ov_free(ov);
```

`tests/abi-c.c` is built with `-std=c99 -Wall -Werror -pedantic` on
every build, so any regression that breaks plain C consumability fails
the build, not just an opt-in target.

For a binding-friendly shared library (libomnivoice.so / .dll / .dylib),
configure with `cmake -DOMNIVOICE_SHARED=ON ...`. The shared target
exports only the `ov_*` symbols ; every internal `pipeline_*` and
`backend_*` stays hidden inside the .so.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the model, the
GGUF layout, the inference pipeline, every CLI flag, the public API
reference and the validation results.

## License

MIT. See [LICENSE](LICENSE).

Upstream model : OmniVoice by Xiaomi / k2-fsa, Apache 2.0.
Audio codec : Higgs Audio v2 (`bosonai/higgs-audio-v2-tokenizer`),
Apache 2.0.
