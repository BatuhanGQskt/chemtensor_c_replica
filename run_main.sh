#!/usr/bin/env bash
# Build C project, run main, and copy generated JSON reference data to Mojo test_data.
# Touch main.c so that included manual_tests/mpo.c changes are picked up on rebuild.
# Output: generated/mpo/*.json and generated/dmrg/*.json (created under build/ when run from build).

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

# Create output dirs so C can write generated/mpo/ and generated/dmrg/ (relative to build/)
mkdir -p generated/mpo generated/dmrg

echo "==> Running ./main..."
if ! ./main; then
    echo "ERROR: ./main failed (exit $?)"
    exit 1
fi

# Copy generated JSONs to Mojo test_data (sibling repo) - optional, do not exit on failure
MOJO_TEST_DATA="${SCRIPT_DIR}/../chemtensor_mojo/chemtensor_mojo/test_data"
if [ -d "$MOJO_TEST_DATA" ]; then
    echo "==> Copying generated JSONs to Mojo test_data..."
    for f in generated/mpo/*.json generated/dmrg/*.json; do
        [ -f "$f" ] || continue
        cp "$f" "$MOJO_TEST_DATA/" && echo "  Copied $f"
    done
    # MPS reference files (written to build/ by main)
    for f in mps_product_*.json; do
        [ -f "$f" ] || continue
        cp "$f" "$MOJO_TEST_DATA/" && echo "  Copied $f"
    done
else
    echo "Warning: Mojo test_data not found at $MOJO_TEST_DATA (skipping copy)"
fi

cd .. || true
echo "==> Done."
