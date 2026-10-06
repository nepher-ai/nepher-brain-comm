#!/usr/bin/env bash
# Build the GR00T N1.7 brain image from the nepher-brain repository root and print its digest.
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
tag="${1:-nepher-brain-gr00t-n17}"

docker build -f "${root}/images/gr00t-n17/Dockerfile" -t "${tag}" "${root}"
docker push "${tag}"
docker image inspect --format '{{index .RepoDigests 0}}' "${tag}"
