#!/bin/bash

source_file=/home/eidf018/eidf018/s2484588-epcc/MLP/LLMMemoryEval
target_dir="."

kubectl cp $source_file mem-eval-pod:$target_dir 