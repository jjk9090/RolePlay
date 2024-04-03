from datasets import load_dataset
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Annotated, Any, Optional, Union
from torch import nn
import torch
import numpy as np
import ruamel.yaml as yaml
import dataclasses as dc
import functools
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    DataCollatorWithPadding,
    DataCollatorForSeq2Seq,
    AutoModelForCausalLM,
    get_scheduler,
    AdamW
)

from peft import (
    PeftConfig,
    get_peft_model,
    get_peft_config
)

def _resolve_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()

@functools.cache
def _get_yaml_parser() -> yaml.YAML:
    parser = yaml.YAML(typ='safe', pure=True)
    parser.indent(mapping=2, offset=2, sequence=4)
    parser.default_flow_style = False
    return parser

def load_tokenizer_and_model(
        model_dir: str,
        peft_config: Optional[PeftConfig] = None,
) -> tuple[PreTrainedTokenizer, nn.Module]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        empty_init=False,
        use_cache=False,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return tokenizer, model

model_dir = "model/THUDM/chatglm3-6b"
config_path = "configs/lora.yaml"
config_path = _resolve_path(config_path)
kwargs = _get_yaml_parser().load(config_path)
peft_config = kwargs.get('peft_config', None)
peft_config = get_peft_config(peft_config)
tokenizer, model = load_tokenizer_and_model(model_dir, peft_config)

data_dir = "dataset/weibo/like_retweet/G"
data_files = "train.json"
num_proc = 16
raw_datasets = load_dataset(
    "json",
    data_dir=data_dir,
    data_files=data_files,
    num_proc=num_proc,
)
data_files = "dev.json"
test_dataset = load_dataset(
    "json",
    data_dir=data_dir,
    data_files=data_files,
    num_proc=num_proc,
)
def tokenize_function(batch):
    max_input_length = 128 
    max_output_length = 256
    batched_conv = batch['conversations']
    batched_input_ids = []
    batched_labels = []

    for conv in batched_conv:
        input_ids, loss_masks = [
            tokenizer.get_command('[gMASK]'),
            tokenizer.get_command('sop'),
        ], [False, False]
        for message in conv:
            if message['role'] in ('system', 'user'):
                loss_mask_val = False
            else:
                loss_mask_val = True
            new_input_ids = tokenizer.build_single_message(
                message['role'], '', message['content']
            )
            new_loss_masks = [loss_mask_val] * len(new_input_ids)
            input_ids += new_input_ids
            loss_masks += new_loss_masks

        input_ids.append(tokenizer.eos_token_id)
        loss_masks = [False, *loss_masks]
        labels = []
        for input_id, mask in zip(input_ids, loss_masks):
            if mask:
                labels.append(input_id)
            else:
                labels.append(-100)
        max_length = max_input_length + max_output_length + 1
        batched_input_ids.append(input_ids[:max_length])
        batched_labels.append(labels[:max_length])
    return {'input_ids': batched_input_ids, 'labels': batched_labels}

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    max_length=128, 
    return_tensors='pt',
)

remove_columns = raw_datasets['train'].column_names
tokenized_datasets = raw_datasets['train'].map(tokenize_function, batched=True, remove_columns=remove_columns, num_proc=16)
tokenized_datasets.set_format("torch")
train_dataloader = DataLoader(
    tokenized_datasets, shuffle=True, batch_size=1, collate_fn=data_collator
)

optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 100
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(num_training_steps)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
progress_bar = tqdm(range(num_training_steps))
model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)


for epoch in range(num_epochs):
    for conv in test_dataset['train']['conversations']:
        prompt = conv[0]['content']
        print(prompt)
        profile = model.chat(tokenizer, prompt, history=[], max_new_tokens=512)
        print(profile)
        exit(0)