import os
import functools
import logging
import warnings

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.utils.data import DataLoader, RandomSampler

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraModel, LoraConfig

from datasets import load_dataset

from cycling_utils import atomic_torch_save, AtomicDirectory, TimestampedTimer, InterruptableDistributedSampler
from fsdp_utils import bfSixteen_ready, bfSixteen_policy, count_trainable_parameters, AppState, get_args_parser

timer = TimestampedTimer("Start")

# suppressing warnings about missing modules in state_dict
logger = logging.getLogger("torch.distributed.fsdp._state_dict_utils")
logger.setLevel(logging.ERROR)
# suppress warnings about "UserWarning: `_get_pg_default_device` will be deprecated" while saving and loading
warnings.filterwarnings("ignore", category=UserWarning)

ADAPTER_NAME = "ExampleLora"
SHARD_STRATEGY = ShardingStrategy.FULL_SHARD

'''
For Hybrid shard we need to:
1. Initialize the device_mesh as an (NNODES, NPROC) array.
2. Set SHARD_STRATEGY = ShardingStrategy.HYBRID_SHARD
3. Enumerate processes within one model shard group to participate in saving - HOW??
4. Gate saving on being member of that group
5. Pass the saving process group to the dcp.save function
'''

print("Finished imports")

if __name__ == "__main__":
    MODEL_WEIGHT_IDS = {
    "DeepSeek-R1-Distill-Llama-70B": "e4b2dc79-79af-4a80-be71-c509469449b4",
    "DeepSeek-R1-Distill-Llama-8B": "38b32289-7d34-4c72-9546-9d480f676840",
    "DeepSeek-R1-Distill-Qwen-1.5B": "6c796efa-7063-4a74-99b8-aab1c728ad98",
    "DeepSeek-R1-Distill-Qwen-14B": "39387beb-9824-4629-b19b-8f7b8f127150",
    "DeepSeek-R1-Distill-Qwen-32B": "84c2b2cb-95b4-4ce6-a2d4-6f210afad36b",
    "DeepSeek-R1-Distill-Qwen-7B": "a792646c-39f5-4971-a169-425324fec87b",
    }
    MODEL_NAME_SETME = "DeepSeek-R1-Distill-Qwen-1.5B"
    
    model_path = os.path.join("/data", MODEL_WEIGHT_IDS[MODEL_NAME_SETME])
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        use_cache=False, 
        torch_dtype=torch.bfloat16
    ).to("cuda")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0, # set to zero to see identical loss on all ranks
    )

    model = LoraModel(model, lora_config, ADAPTER_NAME)

    timer.report(f"PEFT model: {count_trainable_parameters(model)}")
    timer.report("FSDP wrapped model and broadcast to GPUs")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # prepare dataset and utilities
    dataset = load_dataset('csv', data_files = "../output/sample.csv", split = "train")

    def preprocess_function(examples):
        # Combine question and answer into a single text
        input_text =  '''
                        Input:                         
                        ```python
                        {}
                        ```

                        Output:
                        <think>
                        {}
                        </think>
                        <answer>
                        ```cuda
                        {}
                        ```
                        </answer>
                    '''
        texts = [input_text.format(torch_code, think, cuda) for torch_code, think, cuda in zip(examples['Torch Code'], examples['Response Token'], examples['CUDA Code'])]
        
        # Tokenize with padding and truncation
        encodings = tokenizer(
            texts,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors=None
        )
        
        # Create labels for causal language modeling (shift input_ids right)
        encodings["labels"] = encodings["input_ids"].copy()
        
        return encodings

    # Process dataset
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing and preprocessing dataset",
    )

    # Create dataloader
    def collate_fn(batch):
        return {
            'input_ids': torch.stack([torch.tensor(x['input_ids']) for x in batch]),
            'attention_mask': torch.stack([torch.tensor(x['attention_mask']) for x in batch]),
            'labels': torch.stack([torch.tensor(x['labels']) for x in batch])
        }
    
    train_sampler = RandomSampler(tokenized_dataset)

    batch_size = 4  # Adjust based on your GPU memory
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        sampler=train_sampler
    )

    # load checkpoint if found
    filename = "./checkpoint/"
    if not os.path.isdir(filename):os.makedirs(filename)

    # training
    num_epochs = 5
    save_every = 2
    model.train()
    best_loss = float('inf')

    for epoch in range(num_epochs):
        for step, batch in enumerate(dataloader):

            # Move batch to device
            input_ids = batch["input_ids"].to(torch.cuda.current_device())
            attention_mask = batch["attention_mask"].to(torch.cuda.current_device())
            labels = batch["labels"].to(torch.cuda.current_device())

            # forward, backward, update
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            # dataloader.sampler.advance(len(input_ids))
            optimizer.zero_grad()

            timer.report(f"Step {step} Loss: {loss.item()}")

            if loss < best_loss: 
                best_loss = loss

                state_dict = { "app": AppState(model, optimizer) }
                # .save(state_dict=state_dict, checkpoint_id=checkpoint_directory, process_group=saving_group)
                torch.save({
                            "state_dict": 
                            state_dict['app']
                    
                            }, os.path.join(filename, "model_state.pth"))
                timer.report("Saved checkpoint")

    timer.report("Done.")


