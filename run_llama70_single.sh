############## GSM8K
CUDA_VISIBLE_DEVICES=4 python llama_singlemodel_batch_decoding.py --modelset_name "llama-big" --max_gen_len 100 --dataset CSQA --few_shot 7

############## CSQA
CUDA_VISIBLE_DEVICES=5 python llama_singlemodel_batch_decoding.py --modelset_name "llama-big" --max_gen_len 300 --dataset GSM8K --few_shot 8

