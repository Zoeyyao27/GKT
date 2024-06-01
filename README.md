# GKT: A Novel Guidance-Based Knowledge Transfer Framework For Efficient Cloud-edge Collaboration LLM Deployment

We introduce the GKT framework which excels in both efficiency and affordability. GKT uses larger models to batch generate guidance and then pass the guidance to smaller models, while ensuring user customization.

[paper link](https://arxiv.org/abs/2405.19635)

![image](https://github.com/Zoeyyao27/GKT/blob/main/figure/GKT.png)

GKT consists of two steps: guidance generation and response completion. In guidance generation, teacher model generates guidance prompts using batch generation to process concurrent user inputs. In response completion, student model receives guidance prompt and complete the response. Student model generates output with a batch size of 1 which allows customize generation settings by the user

## 🎉Get Started
### 🛠️Requirement
```
# Please install PyTorch according to your CUDA version.
pip3 install torch torchvision torchaudio
pip install transformers==4.35.0
```
### 👀GKT
👉🏻example for CSQA dataset
```
#guidance generation stage

CUDA_VISIBLE_DEVICES=0 python llama_big2small_stage1_batch_decoding.py \
--model_name "meta-llama/Llama-2-70b-hf" --batch_size 10 \
--max_gen_len 30 --dataset CSQA --few_shot 7


#response completion stage

CUDA_VISIBLE_DEVICES=0 python llama_big2small_stage2_batch_decoding.py \
--model_name "meta-llama/Llama-2-7b-hf" --dataset CSQA \
--few_shot 7 --max_gen_len 100 \
--out_path "output/big2small" \
--big_output_path "output/big2small/stage1/Llama-2-70b-hf/CSQA/30_output_stage1.jsonlines" 

```

👉🏻example for GSM8K dataset
```
#guidance generation stage

#CUDA_VISIBLE_DEVICES=0 python llama_big2small_stage1_batch_decoding.py \
--model_name "meta-llama/Llama-2-70b-hf" --batch_size 10 \
--max_gen_len 40 --dataset GSM8K --few_shot 8

#response completion stage

CUDA_VISIBLE_DEVICES=0 python llama_big2small_stage2_batch_decoding.py \ 
--model_name "meta-llama/Llama-2-7b-hf" --dataset GSM8K \
--max_gen_len 300 --few_shot 8 \
--out_path "output/big2small" \
--big_output_path "output/big2small/stage1/Llama-2-70b-hf/GSM8K/40_output_stage1.jsonlines"

```

👉🏻example for AQUA-RAT dataset
```
#guidance generation stage

CUDA_VISIBLE_DEVICES=0 python llama_big2small_stage1_batch_decoding.py \
--batch_size 10 --max_gen_len 20 --dataset AQuA \
--few_shot 4 --model_name "meta-llama/Llama-2-70b-hf"

#response completion stage

CUDA_VISIBLE_DEVICES=0 python llama_big2small_stage2_batch_decoding.py \
--dataset AQuA --max_gen_len 100 --few_shot 4 \
--model_name "meta-llama/Llama-2-7b-hf" \
--big_output_path "/data/yaoy/big2small_final/output/big2small/stage1/Llama-2-70b-hf/AQuA/20_output_stage1.jsonlines" \
--out_path "output/big2small"
```

### 👀Single mode generation

```
# GSM8K
CUDA_VISIBLE_DEVICES=0 python llama_singlemodel_batch_decoding.py \
--modelset_name "llama-big" --max_gen_len 100 \
--dataset CSQA --few_shot 7

# CSQA
CUDA_VISIBLE_DEVICES=0 python llama_singlemodel_batch_decoding.py \
--modelset_name "llama-big" --max_gen_len 300 \
--dataset GSM8K --few_shot 8

#AQUA
CUDA_VISIBLE_DEVICES=0 python llama_singlemodel_batch_decoding.py \
--modelset_name "llama-big" --max_gen_len 100 \
--dataset AQuA  --few_shot 4
```

## 📖Citing GKT
```
@article{yao2024gkt,
  title={GKT: A Novel Guidance-Based Knowledge Transfer Framework For Efficient Cloud-edge Collaboration LLM Deployment},
  author={Yao, Yao and Li, Zuchao and Zhao, Hai},
  journal={arXiv preprint arXiv:2405.19635},
  year={2024}
}
```
