#!/bin/bash
# Run ciftxt_reader.py for one or more materials over their default axes and temperatures.
# Usage: ./scanner.sh [-T "77,150,293"] [material ...]     (default: Ge at 293 K)
#        PYTHON=python ./scanner.sh -T 77,293 Si Ge C W Ir PbWO4
# Output: output/<material>/<axis>/T<T>K/ (plots, npz), output/summary.csv (one row per
# material/axis/T), output/results.txt (RESULT lines). Axes/planes: channeling/data/materials.yaml
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python3}
TS=293
if [ "${1:-}" = "-T" ]; then TS=$2; shift 2; fi
[ $# -eq 0 ] && set -- Ge
mkdir -p output

for material in "$@"
do
    # tr strips the carriage returns that Python prints on Windows
    while IFS='|' read -r axis plane dir
    do
        echo "$material $dir (T = $TS K)"
        outdir="output/${material}/${dir}"
        rm -rf "$outdir"
        mkdir -p "$outdir"
        "$PYTHON" ciftxt_reader.py "$material" "$axis" --plane "$plane" --T-scan "$TS" \
            --outdir "$outdir" --summary output/summary.csv | tee "$outdir/log.txt" | grep RESULT \
            | tee -a output/results.txt
    done < <("$PYTHON" ciftxt_reader.py --list-axes "$material" | tr -d '\r')
done
