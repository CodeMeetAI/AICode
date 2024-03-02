pos = "middle"
device = "cuda:0"

python ../eval/inference_gemma.py --data_dir "../datasets/data/frames/frames_grouped_3_$pos.json" --answers_file "../results/frames/gemma_3_$pos" --device "$device"

python ../eval/inference_gemma.py --data_dir "../datasets/data/frames/frames_grouped_4_$pos.json" --answers_file "../results/frames/gemma_4_$pos" --device "$device"

python ../eval/inference_gemma.py --data_dir "../datasets/data/frames/frames_grouped_5_$pos.json" --answers_file "../results/frames/gemma_5_$pos" --device "$device"

python ../eval/inference_gemma.py --data_dir "../datasets/data/frames/frames_grouped_6_$pos.json" --answers_file "../results/frames/gemma_6_$pos" --device "$device"
