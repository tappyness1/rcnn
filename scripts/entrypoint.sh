#!/bin/bash --login
set -e

# activate conda environment and let the following process take over
source activate assist
exec "$@"
