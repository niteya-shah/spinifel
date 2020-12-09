#!/bin/bash

set -e
set -x

export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=https://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov,*.ncrc.gov'

jsrun -n1 -a1 -c42 -g6 --bind rs bash ./setup/build_from_scratch.sh
bash ./scripts/run_summit.sh -s -e
