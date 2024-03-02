#!/bin/bash

turns=(2 3 4 5 6 7 8)
modes=("first" "middle" "last")
device="cuda:0"
for turn in "${turns[@]}"; do
    for mode in "${modes[@]}"; do
        data_dir="../datasets/data/natural_questions/natural_questions_grouped_${turn}_${mode}.json"
        answers_file="../results/${mode}/natural_questions/llama2_${turn}_${mode}"

        python ../eval/inference_llama2.py --data_dir "$data_dir" --answers_file "${answers_file}" --device "${device}"
    done
done

