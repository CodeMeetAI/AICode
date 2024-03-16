#!/bin/bash

turns=(2)
modes=(0)
dataset=$1
project_path="/home/eidf018/eidf018/s2484588-epcc/MLP/LLMMemoryEval"

for turn in "${turns[@]}"; do
    for mode in "${modes[@]}"; do
        context_file="${project_path}/datasets/data/multiwoz/multiwoz-${turn}-${mode}${dataset}.json"
        option_file="${project_path}/datasets/data/multiwoz/labels/multiwoz-${turn}-${mode}-label${dataset}.json"
        answers_file="${project_path}/results/dialogue_based/multiwoz${dataset}/gpt3.5turbo-${turn}-${mode}"
        python ${project_path}/eval/inference_chatGPT.py \
            --context_file "$context_file" \
            --option_file "${option_file}" \
            --answers_file "$answers_file"  
    done
done

# turns=(5)
# modes=(0 1 2 3 4)

# for turn in "${turns[@]}"; do
#     for mode in "${modes[@]}"; do
#         context_file="${project_path}/datasets/data/multiwoz/multiwoz-${turn}-${mode}${dataset}.json"
#         option_file="${project_path}/datasets/data/multiwoz/labels/multiwoz-${turn}-${mode}-label${dataset}.json"
#         answers_file="${project_path}/results/dialogue_based/multiwoz${dataset}/gpt3.5turbo-${turn}-${mode}"
#         python ${project_path}/eval/inference_chatGPT.py \
#             --context_file "$context_file" \
#             --option_file "${option_file}" \
#             --answers_file "$answers_file"  
#     done
# done