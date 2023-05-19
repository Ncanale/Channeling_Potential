#!/bin/bash

n_cores=$(nproc)
n_runs=3

material=Si

rm -r output/*.bak
rm -r output/${material}*-*

#echo -e "\e[31mNumber of events: $n_events \e[39m..."
sed -i "s/^Material.*/Material = \"$material\"/" ciftxt_reader.py
axis_array=( "1 0 0" "1 1 0" "1 1 1")

for (( i=0; i<$n_runs; i++ ))
do
    echo "${axis_array[$i]}"
    sed -i "s/^asse = r.*/asse = r\"${axis_array[$i]}\"/" ciftxt_reader.py
    python3 ciftxt_reader.py
    mv -f ${material}*.pdf output/ 
    echo "done $(($i + 1))/$n_runs"
done

