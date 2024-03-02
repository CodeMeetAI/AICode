#!/bin/bash

type=$1

source_file1=/home/eidf018/eidf018/s2484588-epcc/MLP/LLMMemoryEval/eval
source_file2=/home/eidf018/eidf018/s2484588-epcc/MLP/LLMMemoryEval/script
#source_file=/home/eidf018/eidf018/s2484588-epcc/MLP/Dataset
target_dir1="./eval"
target_dir2="./script"

kubectl cp $source_file1 mem-eval-pod$type:$target_dir1
kubectl cp $source_file2 mem-eval-pod$type:$target_dir2