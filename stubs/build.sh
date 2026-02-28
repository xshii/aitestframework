#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TOOLS_DIR="${SCRIPT_DIR}/tools"
BUILD_ROOT="${SCRIPT_DIR}/../build"

# Hardcoded entry/exit symbols (-E / -X)
ENTRY_FN="stub_entry"
EXIT_FN="stub_exit"

# Defaults
PLATFORM="func_sim"
MODE="all"
MODELS=""
SWAP_ENDIAN=0
PAD_ALIGN=0
EMBED_MANIFEST=""
EMBED_DIR=""

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Options:
  -p <platform>    Hook platform (default: func_sim)
  -m <mode>        Build mode: app | ut | all (default: all)
  -s <models>      Default models to run, comma-separated (default: all)
  -e               Enable endian swap on output binary
  -a <bytes>       Pad output binary to N-byte alignment (e.g. 8)
  -w <manifest>    Embed weights: path to weight manifest file
  -d <dir>         Embed weights: base directory for bin files
  -h               Show this help

Examples:
  $(basename "$0") -m app -s tdd
  $(basename "$0") -m app -s tdd,fdd -e -a 8
  $(basename "$0") -m app -w weights/manifest.txt -d weights/
EOF
    exit 0
}

while getopts "p:m:s:ea:w:d:h" opt; do
    case $opt in
        p) PLATFORM="$OPTARG" ;;
        m) MODE="$OPTARG" ;;
        s) MODELS="${OPTARG//,/;}" ;;  # comma -> semicolon for CMake list
        e) SWAP_ENDIAN=1 ;;
        a) PAD_ALIGN="$OPTARG" ;;
        w) EMBED_MANIFEST="$(cd "$(dirname "$OPTARG")" && pwd)/$(basename "$OPTARG")" ;;
        d) EMBED_DIR="$(cd "$OPTARG" && pwd)" ;;
        h) usage ;;
        *) usage ;;
    esac
done

BUILD_DIR="${BUILD_ROOT}/${PLATFORM}"

echo "=== Build: platform=${PLATFORM} mode=${MODE} models=${MODELS:-all} ==="

CMAKE_EXTRA_ARGS=""
if [ -n "${EMBED_MANIFEST}" ]; then
    CMAKE_EXTRA_ARGS="-DEMBED_WEIGHTS=ON -DEMBED_WEIGHT_MANIFEST=${EMBED_MANIFEST} -DEMBED_WEIGHT_DIR=${EMBED_DIR}"
fi

cmake -B "${BUILD_DIR}" -S "${SCRIPT_DIR}" \
    -DHOOK_PLATFORM="${PLATFORM}" \
    -DSTUB_BUILD_TARGET="${MODE}" \
    -DSTUB_MODELS="${MODELS}" \
    ${CMAKE_EXTRA_ARGS}
cmake --build "${BUILD_DIR}"

if [ "${MODE}" = "ut" ] || [ "${MODE}" = "all" ]; then
    ctest --test-dir "${BUILD_DIR}" --output-on-failure
fi

if [ "${MODE}" = "app" ] || [ "${MODE}" = "all" ]; then
    BIN=$(find "${BUILD_DIR}" -maxdepth 1 -name 'libstub_runner.*' | head -1)
    [ -n "${BIN}" ] || { echo "WARN: libstub_runner not found"; exit 0; }

    [ "${SWAP_ENDIAN}" -eq 1 ] && python3 "${TOOLS_DIR}/swap_endian.py" "${BIN}"
    [ "${PAD_ALIGN}" -gt 0 ]   && python3 "${TOOLS_DIR}/pad_align.py" "${BIN}" -a "${PAD_ALIGN}"

    python3 "${TOOLS_DIR}/print_info.py" "${BIN}"
fi
