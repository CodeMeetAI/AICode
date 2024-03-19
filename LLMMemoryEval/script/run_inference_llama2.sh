#!/bin/bash

turns=(1 2 3 4)
modes=(0)
device="cuda:0"
dataset=$1

for turn in "${turns[@]}"; do
    for mode in "${modes[@]}"; do
        context_file="../datasets/data/multiwoz/multiwoz-${turn}-${mode}${dataset}.json"
        option_file="../datasets/data/multiwoz/labels/multiwoz-${turn}-${mode}-label${dataset}.json"
        answers_file="../results/dialogue_based/multiwoz${dataset}/llama2_7b-${turn}-${mode}"
        python ../eval/inference_cat.py \
            --context_file "$context_file" \
            --option_file "${option_file}" \
            --answers_file "$answers_file"  \
            --device "${device}" \
            --model_path "meta-llama/Llama-2-7b-chat-hf"
    done
done

turns=(5)
modes=(0 1 2 3 4)
device="cuda:0"

for turn in "${turns[@]}"; do
    for mode in "${modes[@]}"; do
        context_file="../datasets/data/multiwoz/multiwoz-${turn}-${mode}${dataset}.json"
        option_file="../datasets/data/multiwoz/labels/multiwoz-${turn}-${mode}-label${dataset}.json"
        answers_file="../results/dialogue_based/multiwoz${dataset}/llama2_7b-${turn}-${mode}"
        python ../eval/inference_cat.py \
            --context_file "$context_file" \
            --option_file "${option_file}" \
            --answers_file "$answers_file"  \
            --device "${device}" \
            --model_path "meta-llama/Llama-2-7b-chat-hf"
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
        answers_file="../results/dialogue_based/multiwoz${dataset}/llama2_13b-${turn}-${mode}"
        python ../eval/inference_cat.py \
            --context_file "$context_file" \
            --option_file "${option_file}" \
            --answers_file "$answers_file"  \
            --device "${device}" \
            --model_path "meta-llama/Llama-2-13b-chat-hf"
    done
done

turns=(5)
modes=(0 1 2 3 4)
device="cuda:0"

for turn in "${turns[@]}"; do
    for mode in "${modes[@]}"; do
        context_file="../datasets/data/multiwoz/multiwoz-${turn}-${mode}${dataset}.json"
        option_file="../datasets/data/multiwoz/labels/multiwoz-${turn}-${mode}-label${dataset}.json"
        answers_file="../results/dialogue_based/multiwoz${dataset}/llama2_13b-${turn}-${mode}"
        python ../eval/inference_cat.py \
            --context_file "$context_file" \
            --option_file "${option_file}" \
            --answers_file "$answers_file"  \
            --device "${device}" \
            --model_path "meta-llama/Llama-2-13b-chat-hf"
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
        answers_file="../results/dialogue_based/multiwoz${dataset}/gemma_7b-${turn}-${mode}"
        python ../eval/inference_cat.py \
            --context_file "$context_file" \
            --option_file "${option_file}" \
            --answers_file "$answers_file"  \
            --device "${device}" \
            --model_path "google/gemma-7b-it"
    done
done

turns=(5)
modes=(0 1 2 3 4)
device="cuda:0"

for turn in "${turns[@]}"; do
    for mode in "${modes[@]}"; do
        context_file="../datasets/data/multiwoz/multiwoz-${turn}-${mode}${dataset}.json"
        option_file="../datasets/data/multiwoz/labels/multiwoz-${turn}-${mode}-label${dataset}.json"
        answers_file="../results/dialogue_based/multiwoz${dataset}/gemma_7b-${turn}-${mode}"
        python ../eval/inference_cat.py \
            --context_file "$context_file" \
            --option_file "${option_file}" \
            --answers_file "$answers_file"  \
            --device "${device}" \
            --model_path "google/gemma-7b-it"
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
        answers_file="../results/dialogue_based/multiwoz${dataset}/gemma_2b-${turn}-${mode}"
        python ../eval/inference_cat.py \
            --context_file "$context_file" \
            --option_file "${option_file}" \
            --answers_file "$answers_file"  \
            --device "${device}" \
            --model_path "google/gemma-2b-it"
    done
done

turns=(5)
modes=(0 1 2 3 4)
device="cuda:0"

for turn in "${turns[@]}"; do
    for mode in "${modes[@]}"; do
        context_file="../datasets/data/multiwoz/multiwoz-${turn}-${mode}${dataset}.json"
        option_file="../datasets/data/multiwoz/labels/multiwoz-${turn}-${mode}-label${dataset}.json"
        answers_file="../results/dialogue_based/multiwoz${dataset}/gemma_2b-${turn}-${mode}"
        python ../eval/inference_cat.py \
            --context_file "$context_file" \
            --option_file "${option_file}" \
            --answers_file "$answers_file"  \
            --device "${device}" \
            --model_path "google/gemma-2b-it"
    done
done