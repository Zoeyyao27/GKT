############## 30->100
#CUDA_VISIBLE_DEVICES=0 python llama_big2small_stage1_batch_decoding.py --model_name "meta-llama/Llama-2-70b-hf" --batch_size 10 --max_gen_len 30 --dataset CSQA --few_shot 7

CUDA_VISIBLE_DEVICES=0 python llama_big2small_stage2_batch_decoding.py --model_name "meta-llama/Llama-2-7b-hf" --dataset CSQA --few_shot 7 --big_output_path "/data/yaoy/big2small_final/output/big2small/stage1/Llama-2-70b-hf/CSQA/30_output_stage1.jsonlines" --max_gen_len 100 --out_path="output/big2small/llama70b_stage2"

CUDA_VISIBLE_DEVICES=0 python llama_big2small_stage2_batch_decoding.py --model_name "meta-llama/Llama-2-13b-hf" --dataset CSQA --few_shot 7 --big_output_path "/data/yaoy/big2small_final/output/big2small/stage1/Llama-2-70b-hf/CSQA/30_output_stage1.jsonlines" --max_gen_len 100 --out_path="output/big2small/llama70b_stage2"
