#!/bin/bash

source_file=LLMMemoryEval/results
target_dir=/home/eidf018/eidf018/s2484588-epcc/MLP/LLMMemoryEval/results

kubectl cp mem-eval-pod:$source_file $target_dir 