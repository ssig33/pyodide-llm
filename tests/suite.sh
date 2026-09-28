#!/usr/bin/env bash
# tests/suite.sh (T193): the tests of deploy.yml and tests.yml, in one place so that a branch's tests.yml runs the very
# steps the deploy runs (the deploy runs on main only).
#
#   bash tests/suite.sh light   what keeps a broken page off the site: the engine's pytest, the page's modules in Node,
#                               the smoke test on the latest Pyodide (the deploy runs this one, and tests.yml by default)
#   bash tests/suite.sh full    and the heavier checks: forward.js against NumPy (shared, --plain, --wide), the default
#                               choice of the GPU or the CPU (gpu-default-check) and the software threads (tests.yml
#                               every night, and with full=true)
#
# The owner, 2026-09-27: "たまーにフルテストする感じでコミットたくさんあるときは軽くしたい". Needs what `make models
# kernels` makes and the Python packages of the workflows; the build is the workflow's own step.
set -euo pipefail
suite=${1:-}
case "$suite" in
  light | full) ;;
  *) echo "usage: bash tests/suite.sh light|full (not '$suite')" >&2; exit 2 ;;
esac
echo "suite: $suite, starting"
# each part's name and seconds, for the log (tests/ci.mjs reads the lines that start with "suite:" and "--- ")
part() {
  local name=$1 began=$SECONDS
  shift
  echo "--- $name"
  "$@"
  echo "--- $name: $((SECONDS - began)) s"
}

part "unit tests of the engine" python -m pytest tests -q

# Node alone: the benchmark's tables (T45), the table CI writes (T82), the list, the kept models, the service worker...
page_modules() {
  node tests/bench.mjs
  node tests/page-memory.mjs
  node tests/wake.mjs
  node tests/summary-check.mjs
  node tests/models-check.mjs
  node tests/ladder-check.mjs
  node tests/kept-check.mjs
  node tests/coi-js-check.mjs
  node tests/gpu-choice-check.mjs
  node tests/worker-sink-check.mjs
}
part "unit tests of the page's modules" page_modules

# the page resolves the latest Pyodide release when it is opened, so test against that one
part "the latest Pyodide" npm install --no-save pyodide@latest
part "smoke test" node tests/smoke.mjs
# T217 (the review of T201): attention's softmax where its largest score decides something (two positions far above
# the rest): a largest that leaves positions out, which forward-check's line cannot see (under a second)
part "attention's largest score" node tests/attention-check.mjs

if [ "$suite" = full ]; then
  # T93: the forward pass of public/forward.js against NumPy's, on the site's models (the line is for 128 positions)
  part "forward.js against NumPy" node tests/forward-check.mjs --rounds 1 --positions 128
  part "forward.js against NumPy, not shared" node tests/forward-check.mjs stories260K tiny-lm --rounds 1 --positions 128 --plain
  # T101: the 64-bit memory and its kernels
  part "forward.js against NumPy, 64-bit" node tests/forward-check.mjs stories260K tiny-lm --rounds 1 --positions 128 --wide
  # T229: a small Qwen3.5 (attention and Gated DeltaNet), in float32, int8, and stored in Prism's rotation (Bonsai 2)
  part "a Qwen3.5's files" env PYTHONPATH=tests:public python tests/qwen35_prepare.py .tmp/qwen35
  part "forward.js against NumPy, a Qwen3.5" node tests/forward-check.mjs .tmp/qwen35/qwen35-float32 \
    .tmp/qwen35/qwen35-int8 .tmp/qwen35/qwen35-rotated-int8 --rounds 1 --positions 64
  # T148: the default choice of the GPU or the CPU for a prompt's blocks, with a made-up GPU's worker
  part "the GPU or the CPU by default" node tests/gpu-default-check.mjs
  part "the software threads" node tests/threads-check.mjs
  # T206: the pre-tokenizers against the real ones at every code point (about 90 s, too long for the deploy)
  part "the pre-tokenizers at every code point" env EVERY_CODE_POINT=1 python -m pytest tests/test_bytebpe.py -q -k every_character
fi
echo "suite: $suite passed in $SECONDS s"
