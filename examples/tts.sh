#!/bin/bash

set -eu

../build/omnivoice-tts \
    --model ../models/omnivoice-base-Q8_0.gguf \
    --codec ../models/omnivoice-tokenizer-BF16.gguf \
    --instruct "male, young adult, moderate pitch" \
    --lang English \
    -o tts.wav < prompt.txt
