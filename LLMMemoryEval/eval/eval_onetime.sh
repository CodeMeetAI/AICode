#!/bin/bash

metric_script="/home/eidf018/eidf018/s2484588-epcc/MLP/LLMMemoryEval/eval/metric_compute.py"


cd /home/eidf018/eidf018/s2484588-epcc/MLP/LLMMemoryEval/results/likelihood_exp

# 遍历 frames 和 multiwoz 文件夹中的所有 jsonl 文件
for position in first middle last; do
cd $position
echo "----------------Evaluating results in $position ----------------"
    for folder in frames multiwoz natural_questions; do
    # for folder in natural_questions

        echo "Evaluating results in $folder..."
        cd $folder
        for file in *.jsonl; do
        echo "Processing $file..."
        python3 "$metric_script" --result_path "$PWD/$file"
        done
        cd ..
    done
cd ..
done


