#!/bin/bash

turns=(3 4 5 6 7 8)
modes=("first" "middle" "last")
device=$1

for turn in "${turns[@]}"; do
    for mode in "${modes[@]}"; do
        data_dir="../datasets/data/multiwoz/multiwoz_grouped_${turn}_${mode}.json"
        answers_file="../results/${mode}/multiwoz/llama2_${turn}_${mode}"
        
        python ../eval/inference_llama2.py --data_dir "$data_dir" --answers_file "${answers_file}"  --device "${device}"
    done
done
