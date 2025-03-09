### INFO: This is a helper script to allow participants to confirm their model is working!
import torch
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.device_mesh import init_device_mesh
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from fsdp_utils import  bfSixteen_policy
from cycling_utils import TimestampedTimer
from utils import CodeData

import csv
import functools
import os
import warnings
import socket
import re
import numpy as np
warnings.filterwarnings('ignore')


class Inference: 
    SHARD_STRATEGY = ShardingStrategy.FULL_SHARD
    BATCH_SIZE = 1
    MODEL_WEIGHT_IDS = {
    "DeepSeek-R1-Distill-Llama-70B": "e4b2dc79-79af-4a80-be71-c509469449b4",
    "DeepSeek-R1-Distill-Llama-8B": "38b32289-7d34-4c72-9546-9d480f676840",
    "DeepSeek-R1-Distill-Qwen-1.5B": "6c796efa-7063-4a74-99b8-aab1c728ad98",
    "DeepSeek-R1-Distill-Qwen-14B": "39387beb-9824-4629-b19b-8f7b8f127150",
    "DeepSeek-R1-Distill-Qwen-32B": "84c2b2cb-95b4-4ce6-a2d4-6f210afad36b",
    "DeepSeek-R1-Distill-Qwen-7B": "a792646c-39f5-4971-a169-425324fec87b",
    }
    MODEL_NAME_SETME = "DeepSeek-R1-Distill-Llama-8B"
    DATA_PATH = f"/data/{MODEL_WEIGHT_IDS[MODEL_NAME_SETME]}"
    OUTPUT_DIR = f"/root/isc-demos/output"

    def __init__(self, 
                 data_path: str,
                 key: str
                 ):
        
        self.timer = TimestampedTimer("Start")
        dist.init_process_group("nccl")
        self.rank = int(os.environ["RANK"])  
        self.device_id = int(os.environ["LOCAL_RANK"])  
        self.is_master = self.rank == 0  
        self.world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(self.device_id)  # Enables calling 'cuda'

        if self.is_master and not os.path.isdir(self.OUTPUT_DIR):
            os.makedirs(self.OUTPUT_DIR)
        
        if self.device_id == 0: 
            report = ("World Size: %d | Host Name: %s") % (self.world_size, socket.gethostname())
            self.timer.report(report)
            device_report  = ('Number of Nodes: %d | Number of proc per Node: %d') % (int(os.environ['NNODES']), int(os.environ['N_PROC']))
            self.timer.report(device_report)
        
        # Print device name
        self.timer.report(f'Rank {self.rank}: Name: {torch.cuda.get_device_name(self.device_id)}')
        
        # Get the train-test split
        key = 'level_1'
        dataset = load_dataset(data_path)
        data = dataset[key].to_pandas()
        train_indices = np.random.choice(len(data), size = int(0.95 * len(data)))
        test_indices =  np.array([i for i in range(len(data)) if i not in train_indices])

        train_dataset, test_dataset = CodeData(data.iloc[train_indices]), CodeData(data.iloc[test_indices])

        # sampler = DistributedSampler(dataset)
        self.train_data_loader = DataLoader(train_dataset, self.BATCH_SIZE, shuffle = True)  
        self.test_data_loader = DataLoader(test_dataset, self.BATCH_SIZE, shuffle = True)

        self.R1_Input = """ 
            Task: You are an expert in CUDA optimization and PyTorch performance analysis. 
            Your goal is to analyze a given PyTorch implementation and its corresponding CUDA kernel, then explain why the CUDA version is the most optimized.

            Input:
            Operation Name: {}
            Kernel Name: {}

            PyTorch Code:
            ```python
            {}
            ```
            CUDA Kernel Code:
            ```cuda
            {}
            ```
            Expected Output:

            Performance Gains: Explain why the CUDA version is faster, referring to execution time, memory efficiency, and computational throughput.
            Optimization Techniques Used: Identify key optimizations, such as:
            Shared memory usage
            Global memory coalescing
            Warp efficiency and thread utilization
            Use of intrinsic functions
            Avoidance of unnecessary memory transfers
            Bottlenecks Avoided: Describe inefficiencies in the PyTorch version that the CUDA kernel optimizes (e.g., redundant memory accesses, synchronization overhead).
            Comparative Metrics: If available, compare speedup ratios (CUDA_Runtime vs. PyTorch_Native_Runtime), memory usage, and parallel execution efficiency.
            Further Improvements: If applicable, suggest potential refinements for even better performance.
            Note: Keep the explanation concise yet detailed, using GPU architecture knowledge and CUDA best practices. Give the explanation within <answer> </answer> tags
            """



    def create_csv(self, 
                   filename, 
                   fieldnames, 
                   data):
        
        with open(filename, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            for row in data:
                writer.writerow(row)

    def run(self):
        tokenizer = AutoTokenizer.from_pretrained(self.DATA_PATH)
        # if self.rank == 0:
        model = AutoModelForCausalLM.from_pretrained(
            self.DATA_PATH, 
            use_cache=False, 
            torch_dtype=torch.bfloat16
        ).to("cuda")

        #     print(f"Main rank {self.rank} model params on device: {set([p.data.device for p in model.parameters()])}")
        # else:
        #     with torch.device("meta"):
        #         model = AutoModelForCausalLM.from_pretrained(
        #             self.DATA_PATH, 
        #             use_cache=False, 
        #             torch_dtype=torch.bfloat16
        #         )
        #         print(f"Non-main rank {self.rank} model params on device: {set([p.data.device for p in model.parameters()])}")

        # wrap model in FSDP
        # my_auto_wrap_policy = functools.partial(
        #     size_based_auto_wrap_policy, min_num_params=1_000
        # )
        # device_mesh = init_device_mesh("cuda", (self.world_size,))
        # model = FSDP(model, 
        #     auto_wrap_policy=my_auto_wrap_policy,
        #     sharding_strategy=self.SHARD_STRATEGY,
        #     mixed_precision=bfSixteen_policy,
        #     cpu_offload=CPUOffload(offload_params=True),
        #     device_id=torch.cuda.current_device(),
        #     param_init_fn=lambda mod: mod.to_empty(device=torch.cuda.current_device(), recurse=False), # for init from 'meta' device
        #     sync_module_states=True, # broadcast model weights from main rank
        #     device_mesh=device_mesh
        # )
        # filename = os.path.join(self.OUTPUT_DIR, f"{self.device_id}__{torch.cuda.get_device_name(self.device_id)}.csv")


        filename = os.path.join(self.OUTPUT_DIR, "train.csv")
        fieldnames = ['Operation Name', 'Kernel Name', 'CUDA Code', 'Torch Code', 'Response Token', 'Answer Token']
        if os.path.isfile(filename):
            os.remove(filename)
        
        with open(filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
        
    
        for i, data in enumerate(self.train_data_loader):
            self.timer.report(f"Entered {i} row")
            deepseek_r1_input = self.R1_Input.format(data.get('operation')[0], 
                                                data.get('kernel')[0], 
                                                data.get('cuda')[0], 
                                                data.get('torch')[0])
            
            messages=[
                { 'role': 'user', 'content': deepseek_r1_input}
            ]
            inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
            outputs = model.generate(inputs, max_new_tokens=2048, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id, temperature = 0.01)
            answer = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

            # Extract content from think tokens
            response_token = re.search(r"<think>(.*?)</think>", answer, re.DOTALL) 
            answer_token =  re.search(r"<answer>(.*?)</answer>", answer, re.DOTALL)
        
            
            row = {'Operation Name': data.get('operation')[0],
                    'Kernel Name': data.get('kernel')[0],
                    'CUDA Code': data.get('cuda')[0],
                    'Torch Code':data.get('torch')[0],
                    'Response Token': response_token.group(1) if response_token else None,
                    'Answer Token': answer_token.group(1) if answer_token else None
                    }
     
            self.create_csv(filename, fieldnames, [row])
            # self.timer.report(f'Rank : {self.rank} | Created a file after {i} steps')

        self.timer.report("Training Done.")
        # dist.barrier()
        # dist.destroy_process_group()

        if os.path.isfile(filename):
            os.remove(filename)

        filename = os.path.join(self.OUTPUT_DIR, "test.csv")
        if os.path.isfile(filename):
            os.remove(filename)
        fieldnames = ['Operation Name', 'Kernel Name', 'CUDA Code', 'Torch Code', 'Response Token', 'Answer Token']
        
        with open(filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
        
    
        for i, data in enumerate(self.test_data_loader):
            deepseek_r1_input = self.R1_Input.format(data.get('operation')[0], 
                                                data.get('kernel')[0], 
                                                data.get('cuda')[0], 
                                                data.get('torch')[0])
            
            messages=[
                { 'role': 'user', 'content': deepseek_r1_input}
            ]
            inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
            outputs = model.generate(inputs, max_new_tokens=2048, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id, temperature = 0.01)
            answer = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

            # Extract content from think tokens
            response_token = re.search(r"<think>(.*?)</think>", answer, re.DOTALL) 
            answer_token =  re.search(r"<answer>(.*?)</answer>", answer, re.DOTALL)
        
            
            row = {'Operation Name': data.get('operation')[0],
                    'Kernel Name': data.get('kernel')[0],
                    'CUDA Code': data.get('cuda')[0],
                    'Torch Code':data.get('torch')[0],
                    'Response Token': response_token.group(1) if response_token else None,
                    'Answer Token': answer_token.group(1) if answer_token else None
                    }
        

            self.create_csv(filename, fieldnames, [row])
            # self.timer.report(f'Rank : {self.rank} | Created a file after {i} steps')

        self.timer.report("Testing Done.")
        
if __name__ == "__main__":
    inference = Inference("SakanaAI/AI-CUDA-Engineer-Archive", "level_1")
    inference.run()