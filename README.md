# omnivoice.cpp — Pascal (GTX 1080 Ti) fork

Fork of [ServeurpersoCom/omnivoice.cpp](https://github.com/ServeurpersoCom/omnivoice.cpp),
tuned to build and run the OpenAI-compatible `tts-server` on a **Pascal GTX 1080 Ti
(compute capability 6.1 / sm_61)**.

Upstream targets a wide GPU matrix; Pascal is the one architecture the default
CMake CUDA arch list omits, and CUDA 13 dropped it entirely. This fork pins the
build to `sm_61` and ships a Docker build + compose file for that target.

## What this fork changes (vs. upstream)

1. **`ggml` submodule → this fork's own `ggml`** (`.gitmodules` points at
   `herr-schreiber/ggml`). Keeps the whole build in one place and lets the
   ggml side carry the same Pascal-oriented build settings.
2. **Pinned CUDA arch `61`** — the Docker build sets
   `-DCMAKE_CUDA_ARCHITECTURES=61` (no fat binary for newer GPUs).
3. **Short-unprompted-duration patch** in `src/duration-estimator.h` — a minimum
   frame floor so very short, unprompted utterances don't collapse to a DC
   blip / near-silence.
4. **`Dockerfile`** — two-stage (CUDA 12.9 *devel* build → CUDA 12.9 *runtime*
   slim) producing `tts-server` + the ggml backend `.so` files.
5. **`compose.yaml`** — deploys the server on the GPU host with the 1080 Ti
   reserved as **GPU 0** and the GGUFs mounted read-only from the host.

Everything else (the C++ core, the OpenAI endpoints, the voice registry) is
unchanged from upstream.

## The target

| | |
|---|---|
| Host | `gputainer` (`ollama.ugin.herr-schreiber.com`) |
| GPU | NVIDIA GTX 1080 Ti = **GPU 0** (`device_ids: ['0']`) |
| Models | already on the host at **`/opt/1080ki/models`** (mounted read-only as `/opt/models`) |
| Model / codec | `omnivoice-base-Q8_0.gguf` / `omnivoice-tokenizer-Q8_0.gguf` |
| Port | `8015` on the host → `8080` in the container |

## Build

The `ggml` submodule must be checked out **before** `docker build` (the
Dockerfile does `COPY . /src` and does *not* clone submodules itself).

```
git clone --recurse-submodules https://github.com/herr-schreiber/omnivoice.cpp
cd omnivoice.cpp
docker build -t omnivoice-pascal:1080ti .
```

Verify the build compiled for Pascal — the configure line must show
`CMAKE_CUDA_ARCHITECTURES=61`:

```
docker build --progress=plain -t omnivoice-pascal:1080ti . 2>&1 | grep -i 'architectures'
```

The image is self-contained: `tts-server` plus every ggml backend `.so` it
`dlopen`s (they sit beside the binary, resolved via `CMAKE_BUILD_RPATH=$ORIGIN`).
Only the models are supplied at runtime, as a host bind mount.

## Run (compose)

```
docker compose -f compose.yaml up -d
docker compose -f compose.yaml logs -f
```

`compose.yaml` sets, for the `omnivoice` service:

- `--model /opt/models/omnivoice-base-Q8_0.gguf`
- `--codec /opt/models/omnivoice-tokenizer-Q8_0.gguf`
- `--device CUDA0`            (the 1080 Ti, explicitly)
- `--host 0.0.0.0 --port 8080`
- `volumes: /opt/1080ki/models:/opt/models:ro`
- `deploy.resources.reservations.devices: driver nvidia, device_ids ['0']`
- a `curl /health` liveness probe (`start_period: 60s` for model load)

If your GGUFs on the host have different filenames, only the two `--model` /
`--codec` paths in `compose.yaml` need changing.

## API

OpenAI-compatible, from `src/tts-server.h`:

- `POST /v1/audio/speech` — `{"input": "...", "voice": "...", "response_format": "wav"}`
- `POST /v1/audio/voices` — register a cloned voice (WAV or `.rvq`), selected by name
- `GET  /v1/models`, `GET /health`

Smoke test once it is up:

```
curl http://<gputainer>:8015/health
curl -X POST http://<gputainer>:8015/v1/audio/speech \
     -H "Content-Type: application/json" \
     -d '{"model":"omnivoice","input":"Hallo Welt.","voice":"default","format":"wav"}' \
     -o out.wav
```

## Why CUDA 12.9

Newest CUDA that still ships a Pascal (`sm_61`) target. CUDA 13 dropped the
6.x / 7.0 architectures, so 12.9 is the ceiling for a 1080 Ti.

## License

MIT (code); models OmniVoice (Xiaomi / k2-fsa, Apache-2.0) + Higgs Audio v2
tokenizer (Apache-2.0). See upstream `docs/ARCHITECTURE.md` for the full
model / pipeline / API reference.
