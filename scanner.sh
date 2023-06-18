#!/bin/bash

n_cores=$(nproc)
n_runs=3

material=Ge

rm -r output/*.bak
rm -r output/${material}*-*


#echo -e "\e[31mNumber of events: $n_events \e[39m..."
sed -i "s/^Material.*/Material = \"$material\"/" ciftxt_reader.py
axis_array=( "1 0 0" "1 1 0" "1 1 1")
dir_axis=( "100" "110" "111")

for (( i=0; i<$n_runs; i++ ))
do
    echo "${dir_axis[$i]}"
    sed -i "s/^asse = r.*/asse = r\"${axis_array[$i]}\"/" ciftxt_reader.py
    mkdir output/${material}/${dir_axis[$i]}/
    python3 ciftxt_reader.py
    mv -f ${material}${dir_axis[$i]}*.pdf output/${material}/${dir_axis[$i]}/ 
    # cd output/${material}
    # mv -f ${material}_${dir_axis[$i]}*.pdf ${material}/${dir_axis[$i]}/  
    # cd ..
    echo "done $(($i + 1))/$n_runs"
done

