#!/bin/bash

set -eu

if ! command -v jq; then
  echo "This script requires 'jq', please install it before you proceed"
  exit 1
fi

if [ $# -lt 1 ]; then
  echo "Usage $0 input [lang format instructions]"
  echo "Environment variables:"
  echo
  echo -e "OMNIVOICE_SERVER\tHTTP server URL (default: http://127.0.0.1:1234)"
  echo -e "OMNIVOICE_OUTPUT\tOutput path (default: output.wav)"
  echo -e "OV_TIMEOUT\tClient request timeout in seconds (default: 300)"
  exit 255
fi

: "${OV_SERVER:=http://127.0.0.1:1234}"
: "${OV_OUTPUT:=output.wav}"
: "${OV_TIMEOUT:=300}"
PAYLOAD=$(jq -cn --arg input "$1" --arg lang "${2:-}" --arg format "${3:-}" --arg instructions "${4:-}" \
  'if $input != "" then {input: $input} else . end +
   if $lang != "" then {lang: $lang} else . end +
   if $format != "" then {format: $format} else . end +
   if $instructions != "" then {instructions: $instructions} else . end')

exec curl -v --max-time $OV_TIMEOUT --request POST \
  --header "Content-Type: application/json" \
  --data "$PAYLOAD" "$OV_SERVER"/v1/audio/speech \
  --output "$OV_OUTPUT"
