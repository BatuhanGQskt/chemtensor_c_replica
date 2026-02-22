#!/usr/bin/env bash
# Build C project, run main, and copy generated JSON reference data to Mojo test_data.
# Touch main.c so that included manual_tests/mpo.c changes are picked up on rebuild.
# Output: generated/mpo/*.json, generated/dmrg/*.json, generated/mps/*.json (created under build/ when run from build).

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo "==> Working directory: $SCRIPT_DIR"

# Force recompile of main so that #included mpo.c (and its real-only Heisenberg) is used
touch main.c

echo "==> Removing old build..."
rm -rf build
mkdir -p build
cd build || exit 1

echo "==> Running cmake..."
if ! cmake ..; then
    echo "ERROR: cmake failed (exit $?)"
    exit 1
fi

echo "==> Running make..."
if ! make; then
    echo "ERROR: make failed (exit $?)"
    exit 1
fi

# Create output dirs so C can write generated/mpo/, generated/dmrg/, generated/mps/, generated/perf/ (relative to build/)
mkdir -p generated/mpo generated/dmrg generated/mps generated/perf

echo "==> Running ./main..."
set +e
./main
MAIN_EXIT=$?
set -e
if [ "$MAIN_EXIT" -ne 0 ]; then
    echo "Warning: ./main exited with code $MAIN_EXIT (continuing to copy any generated JSONs)"
fi

# Copy generated JSONs to Mojo test_data (sibling repo) - optional, do not exit on failure
MOJO_TEST_DATA="${SCRIPT_DIR}/../chemtensor_mojo/chemtensor_mojo/test_data"
if [ -d "$MOJO_TEST_DATA" ]; then
    echo "==> Copying generated JSONs to Mojo test_data..."
    for f in generated/mpo/*.json generated/dmrg/*.json generated/mps/*.json; do
        [ -f "$f" ] || continue
        cp "$f" "$MOJO_TEST_DATA/" && echo "  Copied $f"
    done
else
    echo "Warning: Mojo test_data not found at $MOJO_TEST_DATA (skipping copy)"
fi

# Run perf_contractions (writes to generated/perf/contraction_timings.jsonl) and copy to Mojo results/perf/
if [ -x ./perf_contractions ]; then
    echo "==> Running ./perf_contractions..."
    ./perf_contractions 6 2 16 generated/perf/contraction_timings.jsonl || true
    MOJO_RESULTS_PERF="${SCRIPT_DIR}/../chemtensor_mojo/chemtensor_mojo/results/perf"
    if [ -d "$(dirname "$MOJO_RESULTS_PERF")" ]; then
        mkdir -p "$MOJO_RESULTS_PERF"
        for f in generated/perf/*.jsonl; do
            [ -f "$f" ] || continue
            cp "$f" "$MOJO_RESULTS_PERF/" && echo "  Copied $f -> Mojo results/perf/"
        done
    fi
fi

cd .. || true
echo "==> Done."
