#!/bin/bash

turns=(1 2 3 4)
modes=(0)
device="cuda:0"
dataset=$1

for turn in "${turns[@]}"; do
    for mode in "${modes[@]}"; do
        context_file="../datasets/data/multiwoz/multiwoz-${turn}-${mode}${dataset}.json"
        option_file="../datasets/data/multiwoz/labels/multiwoz-${turn}-${mode}-label${dataset}.json"
        answers_file="../results/dialogue_based/multiwoz${dataset}/qwen_7b-${turn}-${mode}"
        python ../eval/inference_qwen2.py \
            --context_file "$context_file" \
            --option_file "${option_file}" \
            --answers_file "$answers_file"  \
            --device "${device}" \
            --model_path "Qwen/Qwen1.5-7B-Chat"
    done
done

turns=(5)
modes=(0 1 2 3 4)
device="cuda:0"

for turn in "${turns[@]}"; do
    for mode in "${modes[@]}"; do
        context_file="../datasets/data/multiwoz/multiwoz-${turn}-${mode}${dataset}.json"
        option_file="../datasets/data/multiwoz/labels/multiwoz-${turn}-${mode}-label${dataset}.json"
        answers_file="../results/dialogue_based/multiwoz${dataset}/qwen_7b-${turn}-${mode}"
        python ../eval/inference_qwen2.py \
            --context_file "$context_file" \
            --option_file "${option_file}" \
            --answers_file "$answers_file"  \
            --device "${device}" \
            --model_path "Qwen/Qwen1.5-7B-Chat"
    done
done

turns=(1 2 3 4)
modes=(0)
device="cuda:0"
dataset=$1

for turn in "${turns[@]}"; do
    for mode in "${modes[@]}"; do
        context_file="../datasets/data/multiwoz/multiwoz-${turn}-${mode}${dataset}.json"
        option_file="../datasets/data/multiwoz/labels/multiwoz-${turn}-${mode}-label${dataset}.json"
        answers_file="../results/dialogue_based/multiwoz${dataset}/qwen_14b-${turn}-${mode}"
        python ../eval/inference_qwen2.py \
            --context_file "$context_file" \
            --option_file "${option_file}" \
            --answers_file "$answers_file"  \
            --device "${device}" \
            --model_path "Qwen/Qwen1.5-14B-Chat"
    done
done

turns=(5)
modes=(0 1 2 3 4)
device="cuda:0"

for turn in "${turns[@]}"; do
    for mode in "${modes[@]}"; do
        context_file="../datasets/data/multiwoz/multiwoz-${turn}-${mode}${dataset}.json"
        option_file="../datasets/data/multiwoz/labels/multiwoz-${turn}-${mode}-label${dataset}.json"
        answers_file="../results/dialogue_based/multiwoz${dataset}/qwen_14b-${turn}-${mode}"
        python ../eval/inference_qwen2.py \
            --context_file "$context_file" \
            --option_file "${option_file}" \
            --answers_file "$answers_file"  \
            --device "${device}" \
            --model_path "Qwen/Qwen1.5-14B-Chat"
    done
done
