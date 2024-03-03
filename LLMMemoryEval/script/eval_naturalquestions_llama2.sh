#!/bin/bash

turns=(4 5 6 7 8)
modes=("first" "middle" "last")
device=$1

for turn in "${turns[@]}"; do
    for mode in "${modes[@]}"; do
        context_file="../datasets/data/natural_questions/natural_questions_grouped_new_${turn}_${mode}.json"
        option_file="../datasets/data/natural_questions/labels/natural_questions_grouped_new_${turn}_${mode}-label.json"
        answers_file="../results/likelihood_exp/${mode}/natural_questions/llama2_${turn}_${mode}"
        python ../eval/inference_likelihood_llama2.py --data_dir "$data_dir" --answers_file "${answers_file}"  --device "${device}"
    done
done
