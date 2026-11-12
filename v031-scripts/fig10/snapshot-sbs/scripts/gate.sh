#!/bin/bash
# Only load gate: one successful completion through the frontend.
timeout 300 kubectl -n <namespace> exec rp-snap-sbs-frontend -c frontend -- bash -c \
 'date +%s.%N; curl -s -m 300 localhost:8000/v1/completions -H "Content-Type: application/json" -d "{\"model\":\"glm-5.2-nvfp4\",\"prompt\":\"The capital of France is\",\"max_tokens\":16,\"temperature\":0}"; echo; date +%s.%N'
