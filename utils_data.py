import jsonlines
from torch.utils.data import Dataset
import json
import os
import random
from datasets import load_dataset



class Gsm8k_dataset(Dataset):
    def __init__(self, tokenizer,args,stage2=False,split="test",sample_idx=-1):
        
        self.args=args
        data_path=os.path.join(self.args.data_path,self.args.dataset,split+".jsonl")
        self.data = self.load_json(data_path)
        self.tokenizer=tokenizer
        self.stage2=stage2
        self.big_output_pre=[]
        self.sample_idx = sample_idx
        if sample_idx>=0:
            model_temp = f"{args.big_model_name.split('/')[-1]}2{args.small_model_name.split('/')[-1]}"
            folder_name = os.path.join(args.out_path,args.dataset,model_temp)
            big_out_path=os.path.join(folder_name,f"{split}_big.jsonlines")
            with jsonlines.open(big_out_path, "r") as big_file:
                for i in big_file:
                    #print(list(i.values()))
                    self.big_output_pre.append(list(i.values())[0][sample_idx]) 
        else:
            if self.stage2:
                with jsonlines.open(args.big_output_path, "r") as big_file:
                    for i in big_file:
                        self.big_output_pre.append(list(i.values())[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ins=self.data[idx]
        if self.stage2: 
            big_output_pre= self.big_output_pre[idx]
            tokenized_full_data=self.tokenize(ins,big_output_pre,self.tokenizer)
        else:
            tokenized_full_data=self.tokenize(ins,None,self.tokenizer)
        return tokenized_full_data
    
    def tokenize(self,test_dict,big_output_pre,tokenizer):

        # examplar = """Answer the question:
        
        # Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.
        # Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.
        # Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.
        
        # """
        examplar=self.create_demo_text()
        if "if_concise_prompt" in self.args and self.args.if_concise_prompt:
            system_prompt=self.args.if_concise_prompt
        else:
            system_prompt=""
        
        # if self.sample_idx >=0 :
        #     instruction=system_prompt+examplar + " Q: " + test_dict["question"] +"\nA: "+big_output_pre[self.sample_idx]
        # else:
        if self.stage2:
            instruction=system_prompt+examplar + " Q: " + test_dict["question"] +"\nA: "+big_output_pre
        else:
            instruction=system_prompt+examplar + " Q: " + test_dict["question"] +"\nA: "
        input_text=instruction
        #TEMPLATE.format_map({'instruction': instruction,"system_prompt":DEFAULT_SYSTEM_PROMPT})

        inputs = tokenizer(input_text, 
                           return_tensors="pt",            
                           pad_to_max_length=True,max_length=1024)
        #inputs["answer"]=test_dict["answer"]
        #print(len(inputs["input_ids"][0]))
        return inputs
    def load_json(self,data_path):
        data=[]
        with jsonlines.open(data_path, "r") as reader:
            for item in reader:
                temp={}
                temp["question"]=item["question"]
                temp["solution"]=item["answer"].split("\n#### ")[0]
                temp["answer"]=int(item["answer"].split("\n#### ")[1].replace(',', ''))
                data.append(temp)
        return data
    def create_demo_text(self):
        direct_answer_trigger_for_fewshot = "The answer is"
        x, z, y = [], [], []
            
        x.append("There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?")
        z.append("There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6.")
        y.append("6")

        x.append("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?")
        z.append("There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.")
        y.append("5")        

        x.append("Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?")
        z.append("Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.")
        y.append("39")        

        x.append("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?")
        z.append("Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8.")
        y.append("8")        

        x.append("Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?")
        z.append("Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9.")
        y.append("9")        

        x.append("There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?")
        z.append("There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29.")
        y.append("29")        

        x.append("Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?")
        z.append("Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls.")
        y.append("33")        

        x.append("Olivia has $23. She bought five bagels for $3 each. How much money does she have left?")
        z.append("Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8.")
        y.append("8")
        
        
        # randomize order of the examples ...
        #index_list = list(range(len(x)))
        index_list = list(range(self.args.few_shot))
        #random.shuffle(index_list)

        # Concatenate demonstration examples ...
        demo_text = ""
        for i in index_list:
            demo_text += "Q: " + x[i] + "\nA: " + z[i] + " " + \
                        direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
        
        return demo_text
    

class CSQA_dataset(Dataset):
    def __init__(self, tokenizer,args,stage2=False,split="test",sample_idx=-1):
        
        self.args=args
        if split=="test":
            file_name="dev_rand_split.jsonl"
        elif split=="train":
            file_name="train_rand_split.jsonl"
        data_path=os.path.join(self.args.data_path,self.args.dataset,file_name)
        self.data = self.load_json(data_path)
        self.tokenizer=tokenizer
        self.stage2=stage2
        self.big_output_pre=[]
        self.sample_idx = sample_idx
        if sample_idx>=0:
            model_temp = f"{args.big_model_name.split('/')[-1]}2{args.small_model_name.split('/')[-1]}"
            folder_name = os.path.join(args.out_path,args.dataset,model_temp)
            big_out_path=os.path.join(folder_name,f"{split}_big.jsonlines")
            with jsonlines.open(big_out_path, "r") as big_file:
                for i in big_file:
                    #print(list(i.values()))
                    self.big_output_pre.append(list(i.values())[0][sample_idx]) 
        else:
            if self.stage2:
                with jsonlines.open(args.big_output_path, "r") as big_file:
                    for i in big_file:
                        self.big_output_pre.append(list(i.values())[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ins=self.data[idx]
        if self.stage2: 
            big_output_pre= self.big_output_pre[idx]
            tokenized_full_data=self.tokenize(ins,big_output_pre,self.tokenizer)
        else:
            tokenized_full_data=self.tokenize(ins,None,self.tokenizer)
        return tokenized_full_data
    
    def tokenize(self,test_dict,big_output_pre,tokenizer):

        # examplar = """Answer the question:
        
        # Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.
        # Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.
        # Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.
        
        # """
        examplar=self.create_demo_text()
        if "if_concise_prompt" in self.args and self.args.if_concise_prompt:
            system_prompt=self.args.if_concise_prompt
        else:
            system_prompt=""
        if self.stage2:
            instruction=system_prompt+examplar + " Q: " + test_dict["question"] +"\nA: "+big_output_pre
        else:
            instruction=system_prompt+examplar + " Q: " + test_dict["question"] +"\nA: "

        # if self.stage2:
        #     instruction=examplar + " Q: " + test_dict["question"] +"\nA: "+big_output_pre
        # else:
        #     instruction=examplar + " Q: " + test_dict["question"] +"\nA: "
        input_text=instruction
        #TEMPLATE.format_map({'instruction': instruction,"system_prompt":DEFAULT_SYSTEM_PROMPT})

        inputs = tokenizer(input_text, 
                           return_tensors="pt",            
                           pad_to_max_length=True,max_length=1024)
        #inputs["answer"]=test_dict["answer"]
        #print(len(inputs["input_ids"][0]))
        return inputs
    def load_json(self,data_path):
        data=[]
        with jsonlines.open(data_path, "r") as reader:
            for item in reader:
                temp={}
                choice = "Answer Choices:"
                for c in item["question"]["choices"]:
                    choice += " ("
                    choice += c["label"]
                    choice += ") "
                    choice += c["text"]
                temp["question"]=item["question"]["stem"].strip() + " " + choice

                #answers.append(json_res["answerKey"])
                
                #temp["solution"]=item["answer"].split("\n#### ")[0]
                temp["answer"]=item["answerKey"]#int(item["answer"].split("\n#### ")[1].replace(',', ''))
                data.append(temp)
        return data
    def create_demo_text(self):
        #"adapted from few shot cot paper"
        direct_answer_trigger_for_fewshot = "The answer is"
        x, z, y = [], [], []
            
        x.append("What do people use to absorb extra ink from a fountain pen? Answer choices: (A) shirt pocket, (B) calligrapher\u2019s hand, (C) inkwell, (D) desk drawer, (E) blotter.")
        z.append("The answer must be an item that can absorb ink. Of the above choices, only blotters are used to absorb ink.")
        y.append("E")

        x.append("What home entertainment equipment requires cable? Answer choices: (A) radio shack, (B) substation, (C) television, (D) cabinet.")
        z.append("The answer must require cable. Of the above choices, only television requires cable.")
        y.append("C")        

        x.append("The fox walked from the city into the forest, what was it looking for? Answer choices: (A) pretty flowers, (B) hen house, (C) natural habitat, (D) storybook.")
        z.append("The answer must be something in the forest. Of the above choices, only natural habitat is in the forest.")
        y.append("B")        

        x.append("Sammy wanted to go to where the people were. Where might he go? Answer choices: (A) populated areas, (B) race track, (C) desert, (D) apartment, (E) roadblock.")
        z.append("The answer must be a place with a lot of people. Of the above choices, only populated areas have a lot of people. ")
        y.append("A")        

        x.append("Where do you put your grapes just before checking out? Answer choices: (A) mouth, (B) grocery cart, (C)super market, (D) fruit basket, (E) fruit market.")
        z.append("The answer should be the place where grocery items are placed before checking out. Of the above choices, grocery cart makes the most sense for holding grocery items.")
        y.append("B")        

        x.append("Google Maps and other highway and street GPS services have replaced what? Answer choices: (A) united states, (B) mexico, (C) countryside, (D) atlas.")
        z.append("The answer must be something that used to do what Google Maps and GPS services do, which is to give directions. Of the above choices, only atlases are used to give directions.")
        y.append("D")        

        x.append("Before getting a divorce, what did the wife feel who was doing all the work? Answer choices: (A) harder, (B) anguish, (C) bitterness, (D) tears, (E) sadness.")
        z.append("The answer should be the feeling of someone getting divorced who was doing all the work. Of the above choices, the closest feeling is bitterness.")
        y.append("C")        

            
        # randomize order of the examples ...
        index_list = list(range(self.args.few_shot))
        #index_list = list(range(len(x)))
        #random.shuffle(index_list)

        # Concatenate demonstration examples ...
        demo_text = ""
        for i in index_list:
            demo_text += "Q: " + x[i] + "\nA: " + z[i] + " " + \
                        direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
        
        return demo_text



class AQuA_dataset(Dataset):
    def __init__(self, tokenizer,args,stage2=False,split="test",sample_idx=-1):
        
        self.args=args
        if split=="test":
            file_name="test.json"
        elif split=="train":
            file_name="train.jsonl"
        data_path=os.path.join(self.args.data_path,self.args.dataset,file_name)
        self.data = self.load_json(data_path)
        self.tokenizer=tokenizer
        self.stage2=stage2
        self.big_output_pre=[]
        self.sample_idx = sample_idx
        if sample_idx>=0:
            model_temp = f"{args.big_model_name.split('/')[-1]}2{args.small_model_name.split('/')[-1]}"
            folder_name = os.path.join(args.out_path,args.dataset,model_temp)
            big_out_path=os.path.join(folder_name,f"{split}_big.jsonlines")
            with jsonlines.open(big_out_path, "r") as big_file:
                for i in big_file:
                    #print(list(i.values()))
                    self.big_output_pre.append(list(i.values())[0][sample_idx]) 
        else:
            if self.stage2:
                with jsonlines.open(args.big_output_path, "r") as big_file:
                    for i in big_file:
                        self.big_output_pre.append(list(i.values())[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ins=self.data[idx]
        if self.stage2: 
            big_output_pre= self.big_output_pre[idx]
            tokenized_full_data=self.tokenize(ins,big_output_pre,self.tokenizer)
        else:
            tokenized_full_data=self.tokenize(ins,None,self.tokenizer)
        return tokenized_full_data
    
    def tokenize(self,test_dict,big_output_pre,tokenizer):

        # examplar = """Answer the question:
        
        # Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.
        # Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.
        # Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.
        
        # """
        examplar=self.create_demo_text()
        if "if_concise_prompt" in self.args and self.args.if_concise_prompt:
            system_prompt=self.args.if_concise_prompt
        else:
            system_prompt=""
        if self.stage2:
            instruction=system_prompt+examplar + " Q: " + test_dict["question"] +"\nA: "+big_output_pre
        else:
            instruction=system_prompt+examplar + " Q: " + test_dict["question"] +"\nA: "

        # if self.stage2:
        #     instruction=examplar + " Q: " + test_dict["question"] +"\nA: "+big_output_pre
        # else:
        #     instruction=examplar + " Q: " + test_dict["question"] +"\nA: "
        input_text=instruction
        #TEMPLATE.format_map({'instruction': instruction,"system_prompt":DEFAULT_SYSTEM_PROMPT})

        inputs = tokenizer(input_text, 
                           return_tensors="pt",            
                           pad_to_max_length=True,max_length=1024)
        #inputs["answer"]=test_dict["answer"]
        #print(len(inputs["input_ids"][0]))
        return inputs
    def load_json(self,data_path):
        data=[]
        with open(data_path, 'r') as file:
            for line in file:
                # 解析每一行的JSON数据
                question_data = json.loads(line)
                temp={}
                choice = "Answer Choices:"
                for c in question_data["options"]:
                    choice += " ("
                    choice += c

                temp["question"]=question_data["question"].strip() + " " + choice

                #answers.append(json_res["answerKey"])
                
                #temp["solution"]=item["answer"].split("\n#### ")[0]
                temp["answer"]=question_data["correct"]#int(item["answer"].split("\n#### ")[1].replace(',', ''))
                data.append(temp)
        return data
    def create_demo_text(self):
        #"adapted from few shot cot paper"
        direct_answer_trigger_for_fewshot = "The answer is"
        x, z, y = [], [], []
            
        x.append("John found that the average of 15 numbers is 40. If 10 is added to each number then the mean of the numbers is? Answer choices: (A) 50 (B) 45 (C) 65 (D) 78 (E) 64")
        z.append("If 10 is added to each number, then the mean of the numbers also increases by 10. So the new mean would be 50.")
        y.append("A")

        x.append("If a / b = 3/4 and 8a + 5b = 22,then find the value of a. Answer choices:  (A) 1/2 (B) 3/2 (C) 5/2 (D) 4/2 (E) 7/2")
        z.append("If a / b = 3/4, then b = 4a / 3. So 8a + 5(4a / 3) = 22. This simplifies to 8a + 20a / 3 = 22, which means 44a / 3 = 22. So a is equal to 3/2.")
        y.append("B")        

        x.append("A person is traveling at 20 km/hr and reached his destiny in 2.5 hr then find the distance? Answer choices:  (A) 53 km (B) 55 km (C) 52 km (D) 60 km (E) 50 km")
        z.append("The distance that the person traveled would have been 20 km/hr * 2.5 hrs = 50 km.")
        y.append("E")        

        x.append("How many keystrokes are needed to type the numbers from 1 to 500? Answer choices:  (A) 1156 (B) 1392 (C) 1480 (D) 1562 (E) 1788")
        z.append("There are 9 one-digit numbers from 1 to 9. There are 90 two-digit numbers from 10 to 99. There are 401 three-digit numbers from 100 to 500. 9 + 90(2) + 401(3) = 1392. ")
        y.append("D")        
            
        # randomize order of the examples ...
        index_list = list(range(self.args.few_shot))
        #index_list = list(range(len(x)))
        #random.shuffle(index_list)

        # Concatenate demonstration examples ...
        demo_text = ""
        for i in index_list:
            demo_text += "Q: " + x[i] + "\nA: " + z[i] + " " + \
                        direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
        
        return demo_text
    