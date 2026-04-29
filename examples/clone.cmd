@echo off

set PATH=%~dp0..\build\Release;%PATH%

omnivoice-tts.exe ^
    --model ..\models\omnivoice-base-Q8_0.gguf ^
    --codec ..\models\omnivoice-tokenizer-BF16.gguf ^
    --ref-wav freeman.wav ^
    --ref-text freeman.txt ^
    --lang English ^
    -o clone.wav < prompt.txt

pause
