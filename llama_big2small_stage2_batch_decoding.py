import argparse
import torch
import os
import jsonlines
import json
import re
import random
from utils_data import Gsm8k_dataset,CSQA_dataset,CNNDM_dataset,AQuA_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModelForSeq2SeqLM
import time
from tqdm import tqdm
from thop import profile
import evaluate


def answer_cleansing_gsm8k(pred,few_shot):
    pred = pred.lower()
    direct_answer_trigger_for_fewshot = "The answer is".lower()
    preds = pred.split(direct_answer_trigger_for_fewshot)
    answer_flag = True if len(preds) > few_shot+2 else False 
    if answer_flag:
        # choose the first answer in list                                                                                                     ...
        pred = preds[few_shot+1]
    else:
        # choose the last answer in list ...
        pred = preds[-1]

    pred = pred.replace(",", "")
    pred = pred.replace(" ", "")
    pred = [s for s in re.findall(r'(\d+)', pred)]#re.findall(r'-?\d+\.?\d*', pred)]
    # If there is no candidate in list, null is set.
    if len(pred) == 0:
        return random.randint(0, 100)

    if answer_flag:
        # choose the first element in list ...
        pred = pred[0]
    else:
        # choose the last element in list ...
        pred = pred[-1]
    
    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]
    return int(pred)

def answer_cleansing_csqa(pred,few_shot):
    pattern = re.compile(r'The answer is ([A-Z])')
    res = pattern.findall(pred)
    if len(res) >= few_shot+1:
        return res[few_shot].upper()  # 'A', 'B', ...
    else:
        option=["A","B","C","D"]
        return option[random.randint(0,3)]

def answer_cleansing_aqua(pred,few_shot):
    pattern = re.compile(r'The answer is ([A-Z])')
    res = pattern.findall(pred)
    if len(res) >= few_shot+1:
        return res[few_shot].upper()  # 'A', 'B', ...
    else:
        option=["A","B","C","D","E"]
        return option[random.randint(0,4)]

def generate(
    model,
    input_data,args
):
    top_p= 0.95
    max_gen_len = args.max_gen_len
    for i in input_data:
        input_data[i]=input_data[i].squeeze(1)
    output_sequences = model.generate(**input_data, max_new_tokens=max_gen_len,temperature = 0.8,top_p = top_p) #do_sample=True)#, top_p=top_p)
    results=tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
    return results

def generation_loop(dataloader, model,args):
    # if args.dataset =="GSM8K":
    #     few_shot=8
    # elif args.dataset =="CSQA":
    #     few_shot=7
    few_shot=args.few_shot
    results=[]
    for batch_data in tqdm(dataloader):
        batch_data = batch_data.to(model.device)
        with torch.no_grad():
            generated_tokens = generate(model,
                                        batch_data,args)
            #print(generated_tokens)
            for i in generated_tokens:
                #print(i)
                if "t5" in args.model_name:
                    decoded_output=i
                else:
                    if args.dataset in ["GSM8K","CSQA","AQuA"]:
                        decoded_output=i.split("A:")[few_shot+1].strip()
                        if args.dataset =="GSM8K":
                            answer=answer_cleansing_gsm8k(i,few_shot)
                        elif args.dataset =="CSQA":
                            answer=answer_cleansing_csqa(i,few_shot)
                        elif args.dataset =="AQuA":
                            answer=answer_cleansing_aqua(i,few_shot)
                        item={"output":i,"cot":decoded_output,"ans":answer}
                results.append(item)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, choices=["meta-llama/Llama-2-7b-hf","meta-llama/Llama-2-70b-hf","google/flan-t5-large","bigscience/bloom-3b","bigscience/bloom-7b1","meta-llama/Llama-2-13b-hf"], default="meta-llama/Llama-2-7b-hf")
    parser.add_argument('--max_seq_len', type=int, default=1024)
    parser.add_argument('--data_path', type=str, default="./data/")
    parser.add_argument('--dataset', choices=['GSM8K', 'CSQA',"AQuA"])
    parser.add_argument('--out_path', type=str, default="output/big2small/")
    parser.add_argument('--max_gen_len', type=int, default=180)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--big_output_path', type=str, default="")
    parser.add_argument('--different_model', type=bool, default=False)
    parser.add_argument('--few_shot', type=int,help="GSM8K:8 CSQA:7")
    args = parser.parse_args()

    if args.different_model:
        different_model="diff"
    else:
        different_model=""
    if "concise" in args.big_output_path:
        args_out_path=os.path.join(args.out_path,"stage2_concised1"+different_model)  #stage1 is concised
    else:
        args_out_path=os.path.join(args.out_path,"stage2"+different_model)


    model_temp=args.model_name.split("/")[-1]
    folder_name = os.path.join(args_out_path,model_temp,args.dataset)

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"folder '{folder_name}' created")

    if "chatgpt" not in args.big_output_path:
        stage1_len=args.big_output_path.split("/")[-1].split("_")[0]
    else:
        stage1_len=args.big_output_path.split("/")[-2]
        
    out_path=os.path.join(folder_name,str(stage1_len)+"_"+str(args.max_gen_len)+"_output_stage2.jsonlines")
    
    if os.path.isfile(out_path):
        assert False

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    if "t5" in args.model_name:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, device_map="auto")
    elif "70b" in args.model_name:
        model = AutoModelForCausalLM.from_pretrained(args.model_name,load_in_8bit=True, torch_dtype="auto", device_map="auto",cache_dir="./cache")
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, device_map="auto")
    #dataloader = DataLoader(valid_data, batch_size=32, shuffle=False, collate_fn=collote_fn)
    
    # Define PAD Token = EOS Token
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    if args.dataset =="GSM8K":
        dataset=Gsm8k_dataset(tokenizer,args,stage2=True)
    elif args.dataset =="CSQA":
        dataset=CSQA_dataset(tokenizer,args,stage2=True)
    elif args.dataset =="AQuA":
        dataset=AQuA_dataset(tokenizer,args,stage2=True)

    total=len(dataset)
    right=0
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

        if args.dataset in ["GSM8K","CSQA","AQuA"]:
            qid=0
            for qid, ans in enumerate(answers):
                item={}
                item[qid]=ans
                writer.write(item)
                if str(ans["ans"])==str(dataset.data[qid]["answer"]):
                    right+=1
                qid+=1
        elif args.dataset in ["CNNDM"]:
            qid=0
            answer_list=[]
            label_list=[]
            for qid, ans in enumerate(answers):
                item={}
                item[qid]=ans
                answer_list.append(ans["ans"])
                label_list.append(dataset.data[qid]["highlights"])
                writer.write(item)
                

    ####save args
    args_dict = vars(args)
    args_dict["time"]=execution_time
    args_dict["right"]=right
    args_dict["total"]=total
    if args.dataset in ["GSM8K","CSQA","AQuA"]:
        args_dict["acc"]=right/total
        print("!!! right",right)

    args_dict["FLOPs:(G)"]=flops / 1e9
    args_dict["Number of parameters:(M)"]=params / 1e6
    
    json_filename = os.path.join(folder_name,str(stage1_len)+"_"+str(args.max_gen_len)+"_args.json")
    with open(json_filename, "w") as json_file:
        json.dump(args_dict, json_file, indent=4)
