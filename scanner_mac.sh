#!/bin/zsh

cat scanner.sh |
sed "s+/bin/bash+/bin/zsh+" |
sed "s/sed -i \"/sed -i .bak \"/" |
sed "s+n_cores=.*+alias nproc=\"sysctl -n hw.physicalcpu\"\nn_cores=\$(nproc)+" |
sh

