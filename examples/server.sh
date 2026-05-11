#!/bin/bash

set -eu

exec ../build/omnivoice-tts-server \
    --listen-ip 127.0.0.1 \
    --model ../models/omnivoice-base-Q8_0.gguf \
    --codec ../models/omnivoice-tokenizer-Q8_0.gguf \
    --instruct "male, young adult, moderate pitch" \
    --lang English "$@"
