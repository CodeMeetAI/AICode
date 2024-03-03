#!/bin/bash

metric_script="/home/eidf018/eidf018/s2484588-epcc/MLP/LLMMemoryEval/eval/metric_compute.py"
results_base_path="/home/eidf018/eidf018/s2484588-epcc/MLP/LLMMemoryEval/results/conv_based_exp"
datasets_base_path="/home/eidf018/eidf018/s2484588-epcc/MLP/LLMMemoryEval/datasets/data"

datasets=("frames" "multiwoz" "natural_questions")
#datasets=("natural_questions")
for position in first middle last; do
#for position in first; do
    echo "---------------- Evaluating results in $position ----------------"

    for dataset in "${datasets[@]}"; do
                echo "Evaluating results in $dataset..."
        results_path="${results_base_path}/${position}/${dataset}"
        labels_path="${datasets_base_path}/${dataset}/labels"
        output_csv="${results_path}/${dataset}_${position}_metrics.csv"

        echo "Metric,Value" > "$output_csv"

        # 遍历jsonl
        files=($(ls "${results_path}"/*new*${position}*.jsonl))
        for file_path in "${files[@]}"; do
            file=$(basename "$file_path")
            # 提取文件名
            key_part=$(echo $file | grep -o -E 'new_[0-9]+')
            label_file="${labels_path}/${dataset}_grouped_${key_part}_${position}-label.json"

            if [ -f "$label_file" ]; then
                echo "Processing $file with label ${dataset}_grouped_${key_part}_${position}-label.json..."
                #python3 "$metric_script" --result_path "$file_path" --answer_path "$label_file"
                value=$(python3 "$metric_script" --result_path "$file_path" --answer_path "$label_file")
                if [ $? -eq 0 ]; then
                    # 将文件名和值写入CSV文件
                    echo "${file},${value}" >> "$output_csv"
                else
                    echo "Error processing $file_path"
                fi
            else
                echo "Label file $label_file does not exist."
            fi
        done
    done
done
