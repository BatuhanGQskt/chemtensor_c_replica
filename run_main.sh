#!/usr/bin/env bash
# Build C project, run main, and copy generated JSON reference data to Mojo test_data.
# Touch main.c so that included manual_tests/mpo.c changes are picked up on rebuild.
# Output: generated/mpo/*.json, generated/dmrg/*.json, generated/mps/*.json (created under build/ when run from build).
#
# Usage: run_main.sh [--build] [--mpo-mpo] [--dmrg-only]
#   --build       run cmake and make before executing (default: skip build, run only)
#   --mpo-mpo     pass through to perf_contractions (full MPO→matrix benchmark; huge memory)
#   --dmrg-only   skip perf_contractions (./main still runs; DMRG timings written from dmrg tests)

DO_BUILD=0
DMRG_ONLY=0
PERF_CONTRACTION_FLAGS=()
for arg in "$@"; do
    case "$arg" in
        --build) DO_BUILD=1 ;;
        --mpo-mpo) PERF_CONTRACTION_FLAGS+=(--mpo-mpo) ;;
        --dmrg-only) DMRG_ONLY=1 ;;
    esac
done

# When sourced, $0 may not refer to this script; use BASH_SOURCE[0] instead.
SOURCE_PATH="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="$(cd "$(dirname "$SOURCE_PATH")" && pwd)"

# Restore caller working directory even if the script is sourced.
ORIG_PWD="$(pwd)"
cd "$SCRIPT_DIR" || exit 1

echo "==> Working directory: $SCRIPT_DIR"

# Use incremental build so bench_config.json changes don't require rebuild (config is read at runtime)
mkdir -p build
cd build || exit 1

if [ "$DO_BUILD" -eq 1 ]; then
    # Force recompile of main so that #included mpo.c (and its real-only Heisenberg) is used
    touch "$SCRIPT_DIR/main.c"

    # Avoid re-running full CMake configuration when build system already exists.
    # (Rerunning `cmake ..` can hang inside this environment.)
    if [ ! -f "CMakeCache.txt" ]; then
        echo "==> Running cmake (first-time configure)..."
        if ! cmake ..; then
            echo "ERROR: cmake failed (exit $?)"
            exit 1
        fi
    fi

    echo "==> Running make..."
    if ! make; then
        echo "ERROR: make failed (exit $?)"
        exit 1
    fi
fi

# Create output dirs so C can write generated/mpo/, generated/dmrg/, generated/mps/, generated/perf/ (relative to build/)
mkdir -p generated/mpo generated/dmrg generated/mps generated/perf

# Drop *.jsonl from old bench configs (append-only benchmarks never delete stale filenames)
BENCH_JSON="${SCRIPT_DIR}/../bench_config.json"
PERF_JSONL_CLEANER="${SCRIPT_DIR}/perf/clean_stale_perf_jsonl.py"
if [ -f "$BENCH_JSON" ] && [ -f "$PERF_JSONL_CLEANER" ]; then
    echo "==> Pruning stale generated/perf/*.jsonl (not in $BENCH_JSON)..."
    python3 "$PERF_JSONL_CLEANER" "$SCRIPT_DIR/build/generated/perf" "$BENCH_JSON" || true
fi

echo "==> Running ./main..."
set +e
./main
MAIN_EXIT=$?
set -e
if [ "$MAIN_EXIT" -ne 0 ]; then
    echo "Warning: ./main exited with code $MAIN_EXIT (continuing to copy any generated JSONs)"
fi

# Export RNG sequence for Mojo rng_c_compat comparison test
if [ -x ./rng_export ]; then
    echo "==> Running ./rng_export 42 64..."
    ./rng_export 42 64 > generated/rng_reference.txt || true
fi

# Copy generated JSONs to Mojo test_data (sibling repo) - optional, do not exit on failure
MOJO_TEST_DATA="${SCRIPT_DIR}/../chemtensor_mojo/chemtensor_mojo/test_data"
if [ -d "$MOJO_TEST_DATA" ]; then
    echo "==> Copying generated JSONs to Mojo test_data..."
    for f in generated/mpo/*.json generated/dmrg/*.json generated/mps/*.json; do
        [ -f "$f" ] || continue
        cp "$f" "$MOJO_TEST_DATA/" && echo "  Copied $f"
    done
    [ -f generated/rng_reference.txt ] && cp generated/rng_reference.txt "$MOJO_TEST_DATA/" && echo "  Copied generated/rng_reference.txt"
else
    echo "Warning: Mojo test_data not found at $MOJO_TEST_DATA (skipping copy)"
fi

# Run perf_contractions (writes generated/perf/contraction_timings_{ns}_{d}_{chi}.jsonl) and copy to Mojo results/perf/
MOJO_RESULTS_PERF="${SCRIPT_DIR}/../chemtensor_mojo/chemtensor_mojo/results/perf"
if [ "$DMRG_ONLY" -eq 1 ]; then
    echo "==> Skipping ./perf_contractions (--dmrg-only)"
else
    if [ -x ./perf_contractions ]; then
        echo "==> Running ./perf_contractions..."
        mkdir -p generated/perf
        ./perf_contractions "${PERF_CONTRACTION_FLAGS[@]}" || true
    fi
fi

if [ -d "$(dirname "$MOJO_RESULTS_PERF")" ]; then
    mkdir -p generated/perf
    mkdir -p "$MOJO_RESULTS_PERF"
    if [ -f "$BENCH_JSON" ] && [ -f "$PERF_JSONL_CLEANER" ]; then
        echo "==> Pruning stale Mojo results/perf/*.jsonl (not in $BENCH_JSON)..."
        python3 "$PERF_JSONL_CLEANER" "$MOJO_RESULTS_PERF" "$BENCH_JSON" || true
    fi
    for f in generated/perf/*.jsonl generated/perf/random_mps_psi.json generated/perf/random_mps_chi.json; do
        [ -f "$f" ] || continue
        dest="$MOJO_RESULTS_PERF/$(basename "$f")"
        if [[ "$f" == *.jsonl && -f "$dest" ]]; then
            cat "$f" >> "$dest" && echo "  Appended $f -> Mojo results/perf/"
        else
            cp "$f" "$MOJO_RESULTS_PERF/" && echo "  Copied $f -> Mojo results/perf/"
        fi
    done
fi

cd .. || true
cd "$ORIG_PWD" || true
echo "==> Done."
