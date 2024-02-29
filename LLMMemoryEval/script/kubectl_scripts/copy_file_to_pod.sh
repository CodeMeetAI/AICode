#!/bin/bash

type=$1

source_file=/home/eidf018/eidf018/s2484588-epcc/MLP/LLMMemoryEval
#source_file=/home/eidf018/eidf018/s2484588-epcc/MLP/Dataset
target_dir="."

kubectl cp $source_file mem-eval-pod$type:$target_dir 