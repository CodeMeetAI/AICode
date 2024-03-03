#!/bin/bash

top_dir="/home/eidf018/eidf018/s2484588-epcc/MLP/LLMMemoryEval/results/conv_based_exp"


find "$top_dir" -type f -name '*.jsonl' | while read file; do

    if ! echo "$file" | grep -q 'new_[0-9]'; then

        dir=$(dirname "$file")
        base=$(basename "$file")

        new_base=$(echo "$base" | sed -r 's/(_)([0-9]+_)/\1new_\2/')

        mv "$file" "$dir/$new_base"
    fi
done
