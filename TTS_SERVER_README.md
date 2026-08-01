# TTS Server Documentation

The `tts-server` is an OpenAI-compatible HTTP server that provides text-to-speech capabilities using the OmniVoice ABI.

## API Endpoints

### `POST /v1/audio/speech`
Converts text to speech.

**Request Body (JSON):**

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `input` | `string` | Yes | The text to be synthesized into speech. |
| `voice` | `string` | No | The requested voice. Note: This implementation ignores the voice name and uses the model's default/instructions. |
| `instructions` | `string` | No | Instructions for the voice design/style. |
| `response_format` | `string` | No | The output audio format. Options: `pcm` (default) or `wav`. |
| `speed` | `float` | No | Speed of the speech (currently parsed but not applied). |
| `seed` | `int64` | No | Seed for reproducible output. |

**Response Formats:**

1. **`wav`**:
   - **Content-Type**: `audio/wav`
   - **Description**: A complete RIFF/WAV file containing the full utterance.
   - **Best for**: Saving to a file or playing a single clip.

2. **`pcm`**:
   - **Content-Type**: `audio/pcm`
   - **Description**: A raw stream of `s16le` (Signed 16-bit Little Endian) audio.
   - **Audio Specs**: 24 kHz, Mono.
   - **Best for**: Real-time streaming.
   - **Note**: If saved as a file, you must use a tool like `ffmpeg` to add a header:
     `ffmpeg -f s16le -ar 24000 -ac 1 -i input.pcm output.wav`

---

### `GET /v1/models`
Returns a list of currently loaded models.

**Response Body (JSON):**
```json
{
  "object": "list",
  "data": [
    {
      "id": "model_name",
      "object": "model",
      "owned_by": "local"
    }
  ]
}
```

---

### `GET /v1/voices`
Returns a list of available voices.

**Response Body (JSON):**
```json
{
  "object": "list",
  "voices": []
}
```
*Note: This implementation currently returns an empty list as OmniVoice uses instructions instead of a named voice table.*

---

### `GET /health`
Liveness probe for the server.

**Response Body:**
`{"status":"ok"}`

---

## Command Line Interface (CLI)

The server can be started from the command line with the following arguments:

| Argument | Required | Description |
| :--- | :--- | :--- |
| `--model <path>` | Yes | Path to the LLM GGUF file. |
| `--codec <path>` | Yes | Path to the Codec GGUF file (omnivoice-tokenizer-*.gguf). |
| `--host <ip>` | No | Listen address (default: `127.0.0.1`). |
| `--port <n>` | No | Listen port (default: `8080`). |
| `--lang <str>` | No | Language label (default: `None`). |
| `--no-fa` | No | Disable flash attention. |
| `--clamp-fp16` | No | Clamp hidden states to FP16 range. |

---

## Troubleshooting

### "Invalid Format" in Audio Players
If you receive a file from the `pcm` format and your player reports an "incorrect format" or "invalid file", it is because `pcm` is raw audio without a header.

**Solution:**
1. Use `"response_format": "wav"` in your request.
2. **OR** convert the raw PCM file using `ffmpeg`:
   `ffmpeg -f s16le -ar 24000 -ac 1 -i <input_file> <output_file>.wav`
