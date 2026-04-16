#!/bin/bash

set -euo pipefail
source "$(dirname "$0")/api.sh"

# Run the magicbench.py script with the given model and API key.
python magicbench.py --model gpt-5.1 --provider openai --api-key "$OPENAI_API_KEY"