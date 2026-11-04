#!/bin/bash
# Run ciftxt_reader.py for one or more materials over their default axes and temperatures.
# Usage: ./scanner.sh [-T "77,150,293"] [material ...]     (default: Ge at 293 K)
#        PYTHON=python ./scanner.sh -T 77,293 Si Ge C W Ir PbWO4
#        FASTSIM=1 ./scanner.sh -T 77,293 Si              # also write the Geant4 FastSim data files
# Output: output/<material>/<axis>/T<T>K/ (plots, npz), output/summary.csv (one row per
# material/axis/T), output/results.txt (RESULT lines) and, with FASTSIM=1,
# output/fastsim/<material>/T<T>K/<name>_planes_<hkl>.dat, <name>_axes_<uvw>.dat (see FASTSIM.md).
# Axes/planes: channeling/data/materials.yaml
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python3}
TS=293
if [ "${1:-}" = "-T" ]; then TS=$2; shift 2; fi
[ $# -eq 0 ] && set -- Ge
mkdir -p output

for material in "$@"
do
    extra=""
    if [ -n "${FASTSIM:-}" ]; then extra="--fastsim --fastsim-dir output/fastsim/${material}"; fi
    # tr strips the carriage returns that Python prints on Windows
    while IFS='|' read -r axis plane dir
    do
        echo "$material $dir (T = $TS K)"
        outdir="output/${material}/${dir}"
        rm -rf "$outdir"
        mkdir -p "$outdir"
        # $extra is intentionally unquoted: it holds several words (empty if FASTSIM is unset)
        "$PYTHON" ciftxt_reader.py "$material" "$axis" --plane "$plane" --T-scan "$TS" \
            --outdir "$outdir" --summary output/summary.csv $extra | tee "$outdir/log.txt" | grep RESULT \
            | tee -a output/results.txt
    done < <("$PYTHON" ciftxt_reader.py --list-axes "$material" | tr -d '\r')
done
