import argparse
import torch
import os
import jsonlines
import json
import re
import random
import time
from tqdm import tqdm
from utils_data import Gsm8k_dataset,CSQA_dataset,CNNDM_dataset,AQuA_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModelForSeq2SeqLM
from thop import profile
from peft import PeftModel

def generate(
    model,
    input_data,args
):
    top_p= 0.9
    max_gen_len = args.max_gen_len
    for i in input_data:
        input_data[i]=input_data[i].squeeze(1)
    output_sequences = model.generate(**input_data, max_new_tokens=max_gen_len, temperature = 0.8,top_p = top_p)
    results=tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
    return results

def generation_loop(dataloader, model,args):
    results=[]
    for batch_data in tqdm(dataloader):
        batch_data = batch_data.to(model.device)
        with torch.no_grad():
            generated_tokens = generate(model,
                                        batch_data,args
                                        )
            #print(generated_tokens)
            for i in generated_tokens:
                #print(i)
                results.append(i.split("\nA:")[-1].strip())
        
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, choices=["meta-llama/Llama-2-7b-hf","meta-llama/Llama-2-70b-hf","meta-llama/Llama-2-13b-hf","google/flan-t5-xl","bigscience/bloom-7b1"], default="meta-llama/Llama-2-13b-hf")#default="meta-llama/Llama-2-13b-hf")
    parser.add_argument('--max_seq_len', type=int, default=1024)
    parser.add_argument('--max_batch_size', type=int, default=4)
    parser.add_argument('--dataset', choices=['GSM8K', 'CSQA',"CNNDM","AQuA"])
    parser.add_argument('--if_concise_prompt', choices=[False,"Provide the answer in a brief manner: ","Provide a brief hint for the question: "],default=False)
    parser.add_argument('--data_path', type=str, default="./data/")
    parser.add_argument('--out_path', type=str, default="output/big2small/")
    parser.add_argument('--max_gen_len', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--few_shot', type=int,help="GSM8K:8 CSQA:7 CNNDM:0")
    args = parser.parse_args()

    
    if args.if_concise_prompt:
        args_out_path=os.path.join(args.out_path,"stage1_concised")
    else:
        args_out_path=os.path.join(args.out_path,"stage1")

    model_temp=args.model_name.split("/")[-1]
    folder_name = os.path.join(args_out_path,model_temp,args.dataset)

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"folder '{folder_name}' created")

    out_path=os.path.join(folder_name,str(args.max_gen_len)+"_output_stage1.jsonlines")
    if os.path.isfile(out_path):
        assert False

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    if "t5" in args.model_name:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, device_map="auto")
    elif "70b" in args.model_name:
        model = AutoModelForCausalLM.from_pretrained(args.model_name,load_in_8bit=True, torch_dtype="auto", device_map="auto",cache_dir="./cache")
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, device_map="auto")
        original_named_parameters = dict(model.named_parameters())
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference
    if args.dataset =="GSM8K":
        dataset=Gsm8k_dataset(tokenizer,args,stage2=False)
    elif args.dataset =="CSQA":
        dataset=CSQA_dataset(tokenizer,args,stage2=False)
    elif args.dataset =="AQuA":
        dataset=AQuA_dataset(tokenizer,args,stage2=False)

    with jsonlines.open(out_path, mode='w') as writer:
        dataloader = DataLoader(dataset, batch_size=args.batch_size)
        # Measure FLOPs
        if "t5" not in args.model_name:
            for batch_data in dataloader:
                flops, params = profile(model, inputs=(batch_data["input_ids"].squeeze(1).to(model.device),))
                print(f"FLOPs: {flops / 1e9} G FLOPs")  # FLOPs in billions (GFLOPs)
                print(f"Number of parameters: {params / 1e6} M")  # Parameters in millions (M)
                break
        else:
            flops=0
            params=0
        # recode start time
        start_time = time.time()
        answers=generation_loop(dataloader, model,args)
        # end time
        end_time = time.time()
        execution_time = end_time - start_time
        print("!!!execution_time",execution_time)
        qid=0
        for qid, ans in enumerate(answers):
            item={}
            item[qid]=ans
            writer.write(item)
            qid+=1

    ####save args
    args_dict = vars(args)
    args_dict["time"]=execution_time
    args_dict["FLOPs:(G)"]=flops / 1e9
    args_dict["Number of parameters:(M)"]=params / 1e6
    json_filename = os.path.join(folder_name,str(args.max_gen_len)+"_args.json")
    with open(json_filename, "w") as json_file:
        json.dump(args_dict, json_file, indent=4)
