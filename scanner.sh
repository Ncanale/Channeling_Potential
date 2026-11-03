#!/bin/bash
# Run ciftxt_reader.py for one or more materials over several axes.
# Usage: ./scanner.sh [material ...]      (default: Ge)
#        PYTHON=python ./scanner.sh Si Ge C W PbWO4
# PDFs end up in output/<material>/<axis>/, depths are collected in output/results.txt
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python3}
[ $# -eq 0 ] && set -- Ge

axis_array=( "1 0 0" "1 1 0" "1 1 1" )
dir_axis=( "100" "110" "111" )
n_runs=${#axis_array[@]}
mkdir -p output

for material in "$@"
do
    for (( i=0; i<n_runs; i++ ))
    do
        echo "$material ${dir_axis[$i]}"
        outdir="output/${material}/${dir_axis[$i]}"
        rm -rf "$outdir"
        mkdir -p "$outdir"
        "$PYTHON" ciftxt_reader.py "$material" "${axis_array[$i]}" --outdir "$outdir" | tee "$outdir/log.txt" | grep RESULT \
            | tee -a output/results.txt
    done
done
