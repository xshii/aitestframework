#!/bin/bash
# ci/scripts/archive_artifacts.sh — Package build artifacts with version tags.
#
# Usage: archive_artifacts.sh [platform]
#   platform  Hook platform name (default: func_sim)
set -euo pipefail

PLATFORM="${1:-func_sim}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/../.."
BUILD_DIR="${PROJECT_ROOT}/build/${PLATFORM}"
ARTIFACT_DIR="${PROJECT_ROOT}/build/artifacts"

# Generate version string from git
if git -C "${PROJECT_ROOT}" describe --tags --always > /dev/null 2>&1; then
    VERSION=$(git -C "${PROJECT_ROOT}" describe --tags --always --dirty)
else
    VERSION="dev-$(git -C "${PROJECT_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
fi

echo "=== Archiving artifacts: version=${VERSION} platform=${PLATFORM} ==="

mkdir -p "${ARTIFACT_DIR}"

# Copy shared library with version-tagged name
for ext in so dylib; do
    SRC="${BUILD_DIR}/libstub_runner.${ext}"
    if [ -f "${SRC}" ]; then
        DEST="${ARTIFACT_DIR}/libstub_runner-${VERSION}.${ext}"
        cp "${SRC}" "${DEST}"
        echo "Archived: ${DEST}"
        # Also keep an unversioned copy for convenience
        cp "${SRC}" "${ARTIFACT_DIR}/libstub_runner.${ext}"
    fi
done

# Copy ctest results if present
for xml in "${BUILD_DIR}"/Testing/*/Test.xml; do
    [ -f "${xml}" ] || continue
    cp "${xml}" "${ARTIFACT_DIR}/"
    echo "Archived: ${xml}"
done

# Write version metadata
cat > "${ARTIFACT_DIR}/version.txt" <<EOF
version: ${VERSION}
platform: ${PLATFORM}
build_date: $(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

echo "=== Archive complete ==="
