############## 40->300
#CUDA_VISIBLE_DEVICES=1 python llama_big2small_stage1_batch_decoding.py --model_name "meta-llama/Llama-2-70b-hf" --batch_size 10 --max_gen_len 40 --dataset GSM8K --few_shot 8

CUDA_VISIBLE_DEVICES=1 python llama_big2small_stage2_batch_decoding.py  --model_name "meta-llama/Llama-2-7b-hf" --dataset GSM8K --big_output_path "/data/yaoy/big2small_final/output/big2small/stage1/Llama-2-70b-hf/GSM8K/40_output_stage1.jsonlines" --max_gen_len 300 --few_shot 8 --out_path="output/big2small/llama70b_stage2"

CUDA_VISIBLE_DEVICES=1 python llama_big2small_stage2_batch_decoding.py  --model_name "meta-llama/Llama-2-13b-hf" --dataset GSM8K --big_output_path "/data/yaoy/big2small_final/output/big2small/stage1/Llama-2-70b-hf/GSM8K/40_output_stage1.jsonlines" --max_gen_len 300 --few_shot 8 --out_path="output/big2small/llama70b_stage2"
