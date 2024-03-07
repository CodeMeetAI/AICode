#!/bin/bash

turns=(2 3 4)
modes=(0)
device="cuda:0"

for turn in "${turns[@]}"; do
    for mode in "${modes[@]}"; do
        context_file="../datasets/data/multiwoz/multiwoz-${turn}-${mode}.json"
        option_file="../datasets/data/multiwoz/labels/multiwoz-${turn}-${mode}-label.json"
        answers_file="../results/dialogue_based/${mode}/multiwoz/gemma-${turn}-${mode}"
        python ../eval/inference_likelihood_newdataset_gemma.py --context_file "$context_file" --option_file "${option_file}" --answers_file "$answers_file"  --device "${device}"
    done
done

turns=(5)
modes=(0 1 2 3 4)
device="cuda:0"

for turn in "${turns[@]}"; do
    for mode in "${modes[@]}"; do
        context_file="../datasets/data/multiwoz/multiwoz-${turn}-${mode}.json"
        option_file="../datasets/data/multiwoz/labels/multiwoz-${turn}-${mode}-label.json"
        answers_file="../results/dialogue_based/${mode}/multiwoz/gemma-${turn}-${mode}"
        python ../eval/inference_likelihood_newdataset_gemma.py --context_file "$context_file" --option_file "${option_file}" --answers_file "$answers_file"  --device "${device}"
    done
done