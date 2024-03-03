#!/bin/bash

turns=(4 5 6 7 8)
modes=("first" "middle" "last")
device=$1

for turn in "${turns[@]}"; do
    for mode in "${modes[@]}"; do
        context_file="../datasets/data/frames/frames_grouped_new_${turn}_${mode}.json"
        option_file="../datasets/data/frames/labels/frames_grouped_new_${turn}_${mode}-label.json"
        answers_file="../results/likelihood_exp/${mode}/frames/gemma_13b_${turn}_${mode}"
        python ../eval/inference_likelihood_gemma.py --context_file "$context_file" --option_file "${option_file}" --answers_file "$answers_file"  --device "${device}"
    done
done
