#!/bin/sh
# Build refusion in Docker and drop the binary into ./dist.
#   scripts/docker-build.sh          build → dist/refusion
#   scripts/docker-build.sh test     run the test suite in the container
set -e
cd "$(dirname "$0")/.."
case "${1:-build}" in
  build)
    docker buildx build --target export --output type=local,dest=dist .
    echo "Built dist/refusion — run: ./dist/refusion dist/examples/bracket.rfz" ;;
  test)
    docker buildx build --progress=plain --target test . ;;
  *) echo "usage: $0 [build|test]" >&2; exit 2 ;;
esac
