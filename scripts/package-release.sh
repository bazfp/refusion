#!/bin/sh
# Package output from the Docker export stage.
set -eu
cd "$(dirname "$0")/.."
VERSION="${1:?usage: scripts/package-release.sh VERSION [OUTPUT_DIR]}"
case "$VERSION" in
  [0-9]*.[0-9]*.[0-9]*) ;;
  *) echo "Expected a numeric release version" >&2; exit 2 ;;
esac
OUTPUT="${2:-dist}"
NAME="refusion-v${VERSION}-linux-x86_64"
test -x "$OUTPUT/refusion"
test -f "$OUTPUT/licenses/refusion-MIT.txt"
test -f "$OUTPUT/licenses/LICENSE_LGPL_21.txt"
test -f "$OUTPUT/licenses/OCCT_LGPL_EXCEPTION.txt"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT HUP INT TERM
mkdir -p "$STAGE/$NAME"
cp "$OUTPUT/refusion" "$STAGE/$NAME/"
cp -R "$OUTPUT/examples" "$OUTPUT/licenses" "$STAGE/$NAME/"
cp README.md "$STAGE/$NAME/"
tar -czf "$OUTPUT/$NAME.tar.gz" -C "$STAGE" "$NAME"
(cd "$OUTPUT" && sha256sum "$NAME.tar.gz" > "$NAME.tar.gz.sha256")
echo "Packaged $OUTPUT/$NAME.tar.gz"
