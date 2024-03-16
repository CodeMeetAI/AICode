#!/bin/bash

turns=(2 3 4)
modes=(0)
device="cuda:0"
dataset=$1

for turn in "${turns[@]}"; do
    for mode in "${modes[@]}"; do
        context_file="../datasets/data/multiwoz/multiwoz-${turn}-${mode}${dataset}.json"
        option_file="../datasets/data/multiwoz/labels/multiwoz-${turn}-${mode}-label${dataset}.json"
        answers_file="../results/dialogue_based/multiwoz${dataset}/llama2-${turn}-${mode}"
        python ../eval/inference.py \
        --context_file "$context_file" \
        --option_file "${option_file}" \
        --answers_file "$answers_file"  \
        --device "${device}" \
        --model_path
    done
done

turns=(5)
modes=(0 1 2 3 4)
device="cuda:0"

for turn in "${turns[@]}"; do
    for mode in "${modes[@]}"; do
        context_file="../datasets/data/multiwoz/multiwoz-${turn}-${mode}${dataset}.json"
        option_file="../datasets/data/multiwoz/labels/multiwoz-${turn}-${mode}-label${dataset}.json"
        answers_file="../results/dialogue_based/${mode}/multiwoz${dataset}/llama2-${turn}-${mode}"
        python ../eval/inference_likelihood_newdataset_llama2.py --context_file "$context_file" --option_file "${option_file}" --answers_file "$answers_file"  --device "${device}"
    done
done
