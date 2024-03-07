#!/bin/bash

turns=(60)
modes=("first" "middle" "last")
device="cuda:0"

for turn in "${turns[@]}"; do
    for mode in "${modes[@]}"; do
        context_file="../datasets/data/frames/frames_grouped_new_${turn}_${mode}.json"
        option_file="../datasets/data/frames/labels/frames_grouped_new_${turn}_${mode}-label.json"
        answers_file="../results/likelihood_exp/${mode}/frames/llama2_${turn}_${mode}"
        python ../eval/inference_likelihood_llama2.py --context_file "$context_file" --option_file "${option_file}" --answers_file "$answers_file"  --device "${device}"
    done
done

