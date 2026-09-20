#!/usr/bin/env bash
# Clone the upstream projects this harness benchmarks. Pinned to the commits used for results/.
set -euo pipefail
cd "$(dirname "$0")/../third_party"
clone() { [ -d "$2" ] || git clone "$1" "$2"; git -C "$2" checkout -q "$3" 2>/dev/null || git -C "$2" fetch -q --depth 50 origin && git -C "$2" checkout -q "$3"; }
clone https://github.com/bespokelabsai/nimble            nimble       "$(cat ../setup/pins/nimble 2>/dev/null || echo main)"
clone https://github.com/Heman10x-NGU/openJev-verdict-2.0 verdict      "$(cat ../setup/pins/verdict 2>/dev/null || echo main)"
clone https://github.com/wfzyx/von                        von          "$(cat ../setup/pins/von 2>/dev/null || echo main)"
clone https://github.com/pst2154/Nemotron_Jev             nemotron-jev "$(cat ../setup/pins/nemotron-jev 2>/dev/null || echo main)"
echo "upstream clones ready"
