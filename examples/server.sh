#!/bin/bash
# Start the omnivoice OpenAI-compatible TTS server.

./build/tts-server \
    --model models/omnivoice-base-Q8_0.gguf \
    --codec models/omnivoice-tokenizer-F32.gguf \
    --host 127.0.0.1 --port 8080 --lang English
