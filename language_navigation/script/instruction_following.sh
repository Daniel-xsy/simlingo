#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

GENERATOR="${REPO_ROOT}/language_navigation/generate_language_xml_distance.py"
INPUT_DIR="${REPO_ROOT}/leaderboard/data/bench2drive_split"

for idx in $(seq -w 0 19); do
  input_xml="${INPUT_DIR}/bench2drive_${idx}.xml"
  seed_num=$((10#${idx}))

  echo "Generating ${input_xml} with seed ${seed_num}"
  python "${GENERATOR}" "${input_xml}" --seed "${seed_num}"
done

echo "Done generating instruction_following XML files for indices 00-19."
