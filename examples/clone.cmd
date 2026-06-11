@echo off

set PATH=%~dp0..\build\Release;%PATH%

omnivoice-tts.exe ^
    --model ..\models\omnivoice-base-Q8_0.gguf ^
    --codec ..\models\omnivoice-tokenizer-Q8_0.gguf ^
    --ref-rvq freeman.rvq ^
    --ref-text freeman.txt ^
    --lang English ^
    -o clone.wav < prompt.txt

pause
