# -*- coding: utf-8 -*-
import os
import jieba
import dataclasses as dc
import functools
import numpy as np
import ruamel.yaml as yaml
import torch
import typer
import json
import nltk
import logging
from datetime import datetime
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Optional, Union
from tqdm.auto import tqdm
from datasets import Dataset, DatasetDict, NamedSplit, Split, load_dataset
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

from peft import (
    PeftConfig,
    PeftModelForCausalLM,
    get_peft_config,
    get_peft_model
)
from rouge_chinese import Rouge
from torch import nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EvalPrediction,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    Seq2SeqTrainingArguments, AutoConfig,
    BitsAndBytesConfig,
    get_scheduler,
    AdamW,
    BertTokenizer, BertModel
)
from transformers import DataCollatorForSeq2Seq as _DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer as _Seq2SeqTrainer

from utils.utils import set_logger, load_model_and_tokenizer_after
from torch.utils.data import DataLoader 
import wandb
wandb.init(project="explicit_weibo_random")

logging.getLogger('transformers').setLevel(logging.WARNING)
ModelType = Union[PreTrainedModel, PeftModelForCausalLM]
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
app = typer.Typer(pretty_exceptions_show_locals=False)

class DataCollatorForSeq2Seq(_DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        output_ids = (
            [feature['output_ids'] for feature in features]
            if 'output_ids' in features[0].keys()
            else None
        )
        if output_ids is not None:
            max_output_length = max(len(out) for out in output_ids)
            if self.pad_to_multiple_of is not None:
                max_output_length = (
                        (
                                max_output_length + self.pad_to_multiple_of - 1) //
                        self.pad_to_multiple_of * self.pad_to_multiple_of
                )
            for feature in features:
                remainder = [self.tokenizer.pad_token_id] * (
                        max_output_length - len(feature['output_ids'])
                )
                if isinstance(feature['output_ids'], list):
                    feature['output_ids'] = feature['output_ids'] + remainder
                else:
                    feature['output_ids'] = np.concatenate(
                        [feature['output_ids'], remainder]
                    ).astype(np.int64)
        return super().__call__(features, return_tensors)

def load_behavior(file_path1):
    s = []
    with open(file_path1, "r", newline="") as file1:
        reader1 = csv.reader(file1)
        next(reader1)
        i = 0
        for row1 in reader1:
            name1, user_id1, action1, response1 = row1
            match = re.search(r'\[(.*?)\]', response1)
            if match:
                content1 = match.group(1)
            else:
                content1 = response1
            if content1.startswith('不'):
                s[i] = 0
            else:
                s[i] = 1
            i += 1
    return s

class Behavior:
    s1 = None
    s2 = None
    @classmethod
    def initialize(cls):
        s1_path = "/data/wl/hxy/RolePlay/dataset/weibo/res/BG_behaviors_G.csv"
        s2_path = "/data/wl/hxy/RolePlay/dataset/weibo/res/Real_behaviors.csv"
        cls.s1 = load_behavior(s1_path)
        cls.s2 = load_behavior(s2_path)

def add_loss(behavior_s1, behavior_s2):
    target = torch.tensor(behavior_s1, dtype=torch.float32).to(device)
    loss_fn = torch.nn.BCELoss()
    loss = loss_fn(behavior_s2, target)
    return loss

class Seq2SeqTrainer(_Seq2SeqTrainer):
    def prediction_step(
            self,
            model: nn.Module,
            inputs: dict[str, Any],
            prediction_loss_only: bool,
            ignore_keys=None,
            **gen_kwargs,
    ) -> tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:

        if self.args.predict_with_generate:
            output_ids = inputs.pop('output_ids')
        input_ids = inputs['input_ids']
        loss, generated_tokens, labels = super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys, **gen_kwargs
        )
        # loss_behavior = add_loss(Behavior.s1, Behavior.s2)
        # loss += loss_behavior
        generated_tokens = generated_tokens[:, input_ids.size()[1]:]
        if self.args.predict_with_generate:
            labels = output_ids
        return loss, generated_tokens, labels

    # For P-Tuning a new save_model function is fine for the prefix_encoder model
    # but may cost problems for the whole model loading

    # def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
    #     if output_dir is None:
    #         output_dir = self.args.output_dir
    #     os.makedirs(output_dir, exist_ok=True)
    #     ptuning_params = {k: v for k, v in self.model.transformer.prefix_encoder.state_dict().items()}
    
    #     torch.save(ptuning_params, os.path.join(output_dir, 'pytorch_model.bin'))
    
    #     print(f"P-Tuning model weights saved in {output_dir}")
    
    #     if self.tokenizer is not None:
    #         self.tokenizer.save_pretrained(output_dir)


def _resolve_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()


def _sanity_check(
        input_ids: Sequence[int],
        output_ids: Sequence[int],
        tokenizer: PreTrainedTokenizer,
):
    print('--> Sanity check')
    for in_id, out_id in zip(input_ids, output_ids):
        if in_id == 0:
            continue
        if in_id in tokenizer.tokenizer.index_special_tokens:
            in_text = tokenizer.tokenizer.index_special_tokens[in_id]
        else:
            in_text = tokenizer.decode([in_id])
        print(f'{repr(in_text):>20}: {in_id} -> {out_id}')


@functools.cache
def _get_yaml_parser() -> yaml.YAML:
    parser = yaml.YAML(typ='safe', pure=True)
    parser.indent(mapping=2, offset=2, sequence=4)
    parser.default_flow_style = False
    return parser


@dc.dataclass
class DataConfig(object):
    train_file: str
    val_file: Optional[str] = None
    test_file: Optional[str] = None

    num_proc: Optional[int] = None

    @property
    def data_format(self) -> str:
        return Path(self.train_file).suffix

    @property
    def data_files(self) -> dict[NamedSplit, str]:
        return {
            split: data_file
            for split, data_file in zip(
                [Split.TRAIN, Split.VALIDATION, Split.TEST],
                [self.train_file, self.val_file, self.test_file],
            )
            if data_file is not None
        }


@dc.dataclass
class FinetuningConfig(object):
    data_config: DataConfig

    max_input_length: int
    max_output_length: int

    training_args: Seq2SeqTrainingArguments = dc.field(
        default_factory=lambda: Seq2SeqTrainingArguments(output_dir='./output')
    )
    peft_config: Optional[PeftConfig] = None

    def __post_init__(self):
        if not self.training_args.do_eval or self.data_config.val_file is None:
            # skips the evaluation stage when `do_eval` or `eval_file` is not provided
            self.training_args.do_eval = False
            self.training_args.evaluation_strategy = 'no'
            self.data_config.val_file = None
        else:
            self.training_args.per_device_eval_batch_size = (
                    self.training_args.per_device_eval_batch_size
                    or self.training_args.per_device_train_batch_size
            )

    @classmethod
    def from_dict(cls, **kwargs) -> 'FinetuningConfig':
        training_args = kwargs.get('training_args', None)
        if training_args is not None and not isinstance(
                training_args, Seq2SeqTrainingArguments
        ):
            gen_config = training_args.get('generation_config')
            # TODO: a bit hacky
            if not isinstance(gen_config, GenerationConfig):
                training_args['generation_config'] = GenerationConfig(
                    **gen_config
                )
            kwargs['training_args'] = Seq2SeqTrainingArguments(**training_args)

        data_config = kwargs.get('data_config')
        if not isinstance(data_config, DataConfig):
            kwargs['data_config'] = DataConfig(**data_config)

        peft_config = kwargs.get('peft_config', None)
        if peft_config is not None and not isinstance(peft_config, PeftConfig):
            kwargs['peft_config'] = get_peft_config(peft_config)
        return cls(**kwargs)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> 'FinetuningConfig':
        path = _resolve_path(path)
        kwargs = _get_yaml_parser().load(path)
        return cls.from_dict(**kwargs)


def _load_datasets(
        data_dir: Path,
        data_format: str,
        data_files: dict[NamedSplit, str],
        num_proc: Optional[int],
) -> DatasetDict:
    if data_format in ('.csv', '.json', '.jsonl'):
        dataset_dct = load_dataset(
            data_format[1:],
            data_dir=data_dir,
            data_files=data_files,
            num_proc=num_proc,
        )
    else:
        err_msg = f"Cannot load dataset in the '{data_format}' format."
        raise NotImplementedError(err_msg)

    return dataset_dct


class DataManager(object):
    def __init__(self, data_dir: str, data_config: DataConfig):
        self._num_proc = data_config.num_proc

        self._dataset_dct = _load_datasets(
            _resolve_path(data_dir),
            data_config.data_format,
            data_config.data_files,
            self._num_proc,
        )

    def _get_dataset(self, split: NamedSplit) -> Optional[Dataset]:
        return self._dataset_dct.get(split, None)

    def get_dataset(
            self,
            split: NamedSplit,
            process_fn: Callable[[dict[str, Any]], dict[str, Any]],
            batched: bool = True,
            remove_orig_columns: bool = True,
    ) -> Optional[Dataset]:
        orig_dataset = self._get_dataset(split)
        if orig_dataset is None:
            return

        if remove_orig_columns:
            remove_columns = orig_dataset.column_names
        else:
            remove_columns = None
        return orig_dataset.map(
            process_fn,
            batched=batched,
            remove_columns=remove_columns,
            num_proc=self._num_proc,
        )
        pass


def print_model_size(model: PreTrainedModel):
    print("--> Model")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n--> model has {total_params / 1e6}M params\n")


def process_batch(
        batch: Mapping[str, Sequence],
        tokenizer: PreTrainedTokenizer,
        max_input_length: int,
        max_output_length: int,
) -> dict[str, list]:
    batched_tools = batch.get('tools', None)
    batched_conv = batch['conversations']
    batched_input_ids = []
    batched_labels = []

    if batched_tools is None:
        batched_tools = [None] * len(batched_conv)

    for tools, conv in zip(batched_tools, batched_conv):
        input_ids, loss_masks = [
            tokenizer.get_command('[gMASK]'),
            tokenizer.get_command('sop'),
        ], [False, False]

        if tools is not None:
            raise NotImplementedError()

        for message in conv:
            if message['role'] in ('system', 'user'):
                loss_mask_val = False
            else:
                loss_mask_val = True

            if message['role'] == 'tool':
                raise NotImplementedError()
            else:
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


def process_batch_eval(
        batch: Mapping[str, Sequence],
        tokenizer: PreTrainedTokenizer,
        max_input_length: int,
        max_output_length: int,
) -> dict[str, list]:
    batched_tools = batch.get('tools', None)
    batched_conv = batch['conversations']
    batched_input_ids = []
    # To avoid computing loss, we do not provide the `labels` field in the input dictionary.
    batched_output_ids = []
    pass
    if batched_tools is None:
        batched_tools = [None] * len(batched_conv)

    for tools, conv in zip(batched_tools, batched_conv):
        input_ids = [
            tokenizer.get_command('[gMASK]'),
            tokenizer.get_command('sop'),
        ]

        if tools is not None:
            raise NotImplementedError()

        for message in conv:
            print(len(input_ids))
            if len(input_ids) >= max_input_length:
                break
            if message['role'] == 'tool':
                raise NotImplementedError()
            else:
                new_input_ids = tokenizer.build_single_message(
                    message['role'], '', message['content']
                )
                if message['role'] == 'assistant':
                    output_prompt, output_ids = (
                        new_input_ids[:1],
                        new_input_ids[1:],
                    )
                    output_ids.append(tokenizer.eos_token_id)
                    batched_input_ids.append(
                        input_ids[:max_input_length] + output_prompt[:1]
                    )
                    batched_output_ids.append(output_ids[:max_output_length])
                input_ids += new_input_ids
    return {'input_ids': batched_input_ids, 'output_ids': batched_output_ids}


# Not sure if this is necessary, can set it to half.
# If train with cpu, cast all params to fp32 instead of trainable ones.
def _prepare_model_for_training(model: nn.Module, use_cpu: bool):
    for param in model.parameters():
        if param.requires_grad or use_cpu:
            param.data = param.data.to(torch.float32)


def load_tokenizer_and_model(
        model_dir: str,
        peft_config: Optional[PeftConfig] = None,
) -> tuple[PreTrainedTokenizer, nn.Module]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if peft_config is not None:
        if peft_config.peft_type.name == "PREFIX_TUNING":
            config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
            config.pre_seq_len = peft_config.num_virtual_tokens
            config.use_cache = False
    
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                trust_remote_code=True,
                config=config,
            ).cuda().half().quantize(4)
        if peft_config.peft_type.name == "LORA":
            config_kwargs = {}
            config_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=False,
                # llm_int8_threshold=6.0
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                trust_remote_code=True,
                empty_init=False,
                use_cache=False,
                # **config_kwargs
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            empty_init=False,
            use_cache=False
        )
    print_model_size(model)
    return tokenizer, model


def compute_metrics(eval_preds: EvalPrediction, tokenizer: PreTrainedTokenizer):
    batched_pred_ids, batched_label_ids = eval_preds

    metrics_dct = {'rouge-1': [], 'rouge-2': [], 'rouge-l': [], 'bleu-4': [], 'loss': []}
    for pred_ids, label_ids in zip(batched_pred_ids, batched_label_ids):
        pred_txt = tokenizer.decode(pred_ids).strip()
        label_txt = tokenizer.decode(label_ids).strip()
        pred_tokens = list(jieba.cut(pred_txt))
        label_tokens = list(jieba.cut(label_txt))
        rouge = Rouge()
        scores = rouge.get_scores(' '.join(pred_tokens), ' '.join(label_tokens))
        for k, v in scores[0].items():
            metrics_dct[k].append(round(v['f'] * 100, 4))
        metrics_dct['bleu-4'].append(
            sentence_bleu(
                [label_tokens],
                pred_tokens,
                smoothing_function=SmoothingFunction().method3,
            )
        )
    return {k: np.mean(v) for k, v in metrics_dct.items()}

def initial_train(data_manager, ft_config, tokenizer, model, auto_resume_from_checkpoint):
    train_dataset = data_manager.get_dataset(
        Split.TRAIN,
        functools.partial(
            process_batch,
            tokenizer=tokenizer,
            max_input_length=ft_config.max_input_length,
            max_output_length=ft_config.max_output_length,
        ),
        batched=True,
    )
    print('train_dataset:', train_dataset)
    val_dataset = data_manager.get_dataset(
        Split.VALIDATION,
        functools.partial(
            process_batch_eval,
            tokenizer=tokenizer,
            max_input_length=ft_config.max_input_length,
            max_output_length=ft_config.max_output_length,
        ),
        batched=True,
    )
    if val_dataset is not None:
        print('val_dataset:', val_dataset)
    
    test_dataset = data_manager.get_dataset(
        Split.TEST,
        functools.partial(
            process_batch_eval,
            tokenizer=tokenizer,
            max_input_length=ft_config.max_input_length,
            max_output_length=ft_config.max_output_length,
        ),
        batched=True,
    )
    if test_dataset is not None:
        print('test_dataset:', test_dataset)
        print(len(test_dataset))
    # checks encoded dataset
    _sanity_check(
        train_dataset[0]["input_ids"], train_dataset[0]["labels"], tokenizer
    )

    # turn model to fp32
    _prepare_model_for_training(model, ft_config.training_args.use_cpu)

    ft_config.training_args.generation_config.pad_token_id = (
        tokenizer.pad_token_id
    )
    ft_config.training_args.generation_config.eos_token_id = [
        tokenizer.eos_token_id,
        tokenizer.get_command('<|user|>'),
        tokenizer.get_command('<|observation|>'),
    ]
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=ft_config.training_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding='longest',
            return_tensors='pt',
        ),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=functools.partial(compute_metrics, tokenizer=tokenizer),
    )
    if auto_resume_from_checkpoint.upper() == "" or auto_resume_from_checkpoint is None:
        trainer.train()
    else:
        output_dir = ft_config.training_args.output_dir
        dirlist = os.listdir(output_dir)
        checkpoint_sn = 0
        for checkpoint_str in dirlist:
            if checkpoint_str.find("eckpoint") > 0 and checkpoint_str.find("tmp") == -1:
                checkpoint = int(checkpoint_str.replace("checkpoint-", ""))
                if checkpoint > checkpoint_sn:
                    checkpoint_sn = checkpoint
        if auto_resume_from_checkpoint.upper() == "YES":
            if checkpoint_sn > 0:
                model.gradient_checkpointing_enable()
                model.enable_input_require_grads()
                checkpoint_directory = os.path.join(output_dir, "checkpoint-" + str(checkpoint_sn))
                print("resume checkpoint from  checkpoint-" + str(checkpoint_sn))
                trainer.train(resume_from_checkpoint=checkpoint_directory)
            else:
                trainer.train()
        else:
            if auto_resume_from_checkpoint.isdigit():
                if int(auto_resume_from_checkpoint) > 0:
                    checkpoint_sn = int(auto_resume_from_checkpoint)
                    model.gradient_checkpointing_enable()
                    model.enable_input_require_grads()
                    checkpoint_directory = os.path.join(output_dir, "checkpoint-" + str(checkpoint_sn))
                    print("resume checkpoint from  checkpoint-" + str(checkpoint_sn))
                    trainer.train(resume_from_checkpoint=checkpoint_directory)
            else:
                print(auto_resume_from_checkpoint,
                      "The specified checkpoint sn(" + auto_resume_from_checkpoint + ") has not been saved. Please search for the correct chkeckpoint in the model output directory")

    # test stage
    return test_dataset

def remove_substring(string, substring):
    parts = string.split(substring, 1)
    if len(parts) > 1:
        string = parts[0]
    return string

def criterion(text1, text2):
    model_name = 'model/google-bert/bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    max_length = 500

    text1_tokens = tokenizer.tokenize(text1)[:max_length-2]  # 截断并去掉特殊标记
    text1_input_ids = tokenizer.convert_tokens_to_ids(text1_tokens)

    text2_tokens = tokenizer.tokenize(text2)[:max_length-2]  # 截断并去掉特殊标记
    text2_input_ids = tokenizer.convert_tokens_to_ids(text2_tokens)


    text1_tensor = torch.tensor([text1_input_ids])
    text2_tensor = torch.tensor([text2_input_ids])

    model = BertModel.from_pretrained(model_name)

    with torch.no_grad():
        text1_outputs = model(text1_tensor)
        text2_outputs = model(text2_tensor)

    text1_hidden_states = text1_outputs[0]
    text2_hidden_states = text2_outputs[0]

    text1_avg_pool = torch.mean(text1_hidden_states, dim=1)
    text2_avg_pool = torch.mean(text2_hidden_states, dim=1)

    cos_sim = cosine_similarity(text1_avg_pool, text2_avg_pool)

    loss = 1 - cos_sim
    return loss

def _save(model, tokenizer, output_dir, state_dict=None):
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, state_dict=state_dict)
    tokenizer.save_pretrained(output_dir, state_dict=state_dict)

def tokenize_function(batch, tokenizer):
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

def tokenize_function_one(batch, tokenizer):
    max_input_length = 128 
    max_output_length = 256
    batched_conv = batch['conversations']
    batched_input_ids = []
    batched_labels = []

    conv = batched_conv
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

def create_batch_D(prompt_D_true, prompt_D_fake, tokenizer, tmp_file):
    infos = []
    batch_true = {}
    batch_true['conversations'] = []
    batch_true['conversations'].append({
        "role": "user",
        "content": prompt_D_true
    })
            
    batch_true['conversations'].append({
        "role": "assistant",
        "content": "是"
    })
    # batch_true = tokenize_function_one(batch_true, tokenizer)

    batch_fake = {}
    batch_fake['conversations'] = []
    batch_fake['conversations'].append({
        "role": "user",
        "content": prompt_D_fake
    })
            
    batch_fake['conversations'].append({
        "role": "assistant",
        "content": "否"
    })
    # batch_fake = tokenize_function_one(batch_fake, tokenizer)
    infos.append(batch_true)
    infos.append(batch_fake)
    with open(tmp_file, "w", encoding="utf-8") as f:
        for e in infos:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
            
class MyDataset(Dataset):
    def __init__(self, batch):
        self.batch = batch

    def __len__(self):
        return len(self.batch['input_ids'])  # 根据你的数据结构进行调整

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.batch['input_ids'][idx])
        labels = torch.tensor(self.batch['labels'][idx])
        return {'input_ids': input_ids, 'labels': labels}

def similarity(fake_profile_text, True_profile_text):
    tokenizer = BertTokenizer.from_pretrained("model/bert-base-uncased")
    model = BertModel.from_pretrained("model/bert-base-uncased")

    fake_tokens = tokenizer.encode(fake_profile_text, add_special_tokens=False, truncation=True, padding=True, return_tensors="pt")
    True_tokens = tokenizer.encode(True_profile_text, add_special_tokens=False, truncation=True, padding=True, return_tensors="pt")

    fake_embeddings = model(fake_tokens).last_hidden_state.mean(dim=1).detach().numpy()
    True_embeddings = model(True_tokens).last_hidden_state.mean(dim=1).detach().numpy()

    similarity = cosine_similarity(fake_embeddings, True_embeddings)
    loss = 1 - similarity
    return loss

@app.command()
def main1(
        data_dir: Annotated[str, typer.Argument(help='')],
        mask_type: Annotated[str, typer.Argument(help='mask type')],
        data_dir_d: Annotated[str, typer.Argument(help='')],
        model_dir: Annotated[
            str,
            typer.Argument(
                help='A string that specifies the model id of a pretrained model configuration hosted on huggingface.co, or a path to a directory containing a model configuration file.'
            ),
        ],
        config_file: Annotated[str, typer.Argument(help='')],
        config_file_d: Annotated[str, typer.Argument(help='')],
        log_file: Annotated[str, typer.Argument(help='')],
        output_dir: Annotated[str, typer.Argument(help='')],
        output_dir_d: Annotated[str, typer.Argument(help='')],
        auto_resume_from_checkpoint: str = typer.Argument(
            default='',
            help='If entered as yes, automatically use the latest save checkpoint. If it is a numerical example 12 15, use the corresponding save checkpoint. If the input is no, restart training'
        ),
):
    time_str = datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
    logger = set_logger(log_file, time_str)

    ft_config_G = FinetuningConfig.from_file(config_file)
    ft_config_G.training_args.output_dir = f"{output_dir}{time_str}/"
    tokenizer_G, model_G = load_tokenizer_and_model(model_dir, peft_config=ft_config_G.peft_config)
    data_manager_G = DataManager(data_dir + "all/", ft_config_G.data_config)
    test_dataset_G = initial_train(data_manager_G, ft_config_G, tokenizer_G, model_G, auto_resume_from_checkpoint)

    ft_config_D = FinetuningConfig.from_file(config_file_d)
    ft_config_D.training_args.output_dir = f"{output_dir_d}{time_str}/"
    tokenizer_D, model_D = load_tokenizer_and_model(model_dir, peft_config=ft_config_D.peft_config)
    data_manager_D = DataManager(data_dir_d, ft_config_D.data_config)
    test_dataset_D = initial_train(data_manager_D, ft_config_D, tokenizer_D, model_D, auto_resume_from_checkpoint)

    # criterion = torch.nn.BCELoss()
    optimizer_G = AdamW(model_G.parameters(), lr=5e-5)
    optimizer_D = AdamW(model_D.parameters(), lr=5e-5)

    val_dataset_G = load_dataset(
        "json",
        data_dir=data_dir + mask_type,
        data_files=ft_config_G.data_config.data_files.get(Split.VALIDATION),
        num_proc=ft_config_G.data_config.num_proc,
    )
    num_epochs = 300
    num_training_steps = num_epochs * len(val_dataset_G['train'])
    print(num_training_steps)
    lr_scheduler_G = get_scheduler(
        "linear",
        optimizer=optimizer_G,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    lr_scheduler_D = get_scheduler(
        "linear",
        optimizer=optimizer_D,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    progress_bar = tqdm(range(num_training_steps))
    data_collator_G = DataCollatorForSeq2Seq(
        tokenizer=tokenizer_G,
        padding=True,
        max_length=128, 
        return_tensors='pt',
    )
    data_collator_D = DataCollatorForSeq2Seq(
        tokenizer=tokenizer_D,
        padding=True,
        max_length=128, 
        return_tensors='pt',
    )
    remove_columns = val_dataset_G['train'].column_names
    tokenized_datasets_G = val_dataset_G['train'].map(
        tokenize_function, 
        batched=True, 
        remove_columns=remove_columns, 
        num_proc=16,
        fn_kwargs={"tokenizer": tokenizer_G}
    )
    tokenized_datasets_G.set_format("torch")
    val_dataloader_G = DataLoader(
        tokenized_datasets_G, shuffle=True, batch_size=1, collate_fn=data_collator_G
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_G.to(device)
    model_D.to(device)
    for epoch in range(num_epochs):
        idx = 0
        loss_G_total = 0.0
        loss_D_total = 0.0
        for conv, input1 in zip(val_dataset_G['train']['conversations'], val_dataloader_G):
            idx += 1
            batch1 = {k: v.to(device) for k, v in input1.items()}
            output_G = model_G(**batch1)
            loss_G0 = output_G.loss

            prompt_G = conv[0]['content']
            fake_profile, history = model_G.chat(tokenizer_G, prompt_G, history=[], max_new_tokens=512)
            
            # behavior data
            prompt_D = remove_substring(prompt_G, "请推断出")
            fake_profile = fake_profile.split('\n')[0]
            question_fake = (
                    f"请判断[{fake_profile}]这个profile是否属于我吗？请注意这只是你的判断，你事先不知道我的信息。请以下列格式清楚作答:\n"
                +   f"是/否\n"
            )
            true_profile = conv[1]['content']
            true_profile = true_profile.split('\n')[0]
            question_true = (
                    f"请判断[{true_profile}]这个profile是否属于我吗？请注意这只是你的判断，你事先不知道我的信息。请以下列格式清楚作答:\n"
                +   f"是/否\n"
            )

            prompt_D_true = prompt_D + question_true
            prompt_D_fake = prompt_D + question_fake
            tmp_file = f"tmp_{time_str}.json"
            tmp_dir = 'output/tmp/'
            tmp_file_path = os.path.join(tmp_dir, tmp_file)
            create_batch_D(prompt_D_true, prompt_D_fake, tokenizer_D, tmp_file_path)
            tmp_dataset = load_dataset(
                "json",
                data_dir=tmp_dir,
                data_files=tmp_file,
                num_proc=ft_config_D.data_config.num_proc,
            )
            tokenized_tmp_datasets = tmp_dataset['train'].map(
                tokenize_function, 
                batched=True, 
                remove_columns=remove_columns, 
                num_proc=16,
                fn_kwargs={"tokenizer": tokenizer_G}
            )
            tokenized_tmp_datasets.set_format("torch")
            tmp_dataloader = DataLoader(
                tokenized_tmp_datasets, shuffle=True, batch_size=1, collate_fn=data_collator_G
            )
            tmp_idx = 0
            output_D = []
            j = 0
            # 创建输入给D的数据
            for tmp_batch in tmp_dataloader:
                tmp_batch = {k: v.to(device) for k, v in tmp_batch.items()}
                tmp_output = model_D(**tmp_batch)
                prediction_D = tmp_output.logits
                shape_D = prediction_D.shape 
                # judge = ''
                # for i in range(shape_D[1]):
                #     if i == 0:
                #         continue
                #     predicted_token_id_D = prediction_D[0, i].argmax(axis=-1)
                #     judge += tokenizer_D.decode(predicted_token_id_D)
                # if j == 0:
                #     label = '是'
                # else:
                #     label = '否'
                if j == 0:
                    label = torch.ones_like(prediction_D).cuda()
                else:
                    label = torch.zeros_like(prediction_D).cuda()
                j += 1
                output_D.append([prediction_D, label])
            
            predictions_D_true =  output_D[0][0]
            targets_D_true = output_D[0][1]

            predictions_D_fake =  output_D[1][0]
            targets_D_fake = output_D[1][1]

            predictions_G = output_D[1][0].detach().clone()
            predictions_G.requires_grad_(True)
            targets_G = torch.ones_like(predictions_G).cuda()
            targets_G.requires_grad_(True)

            loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')

            loss_D_true = loss_fn(predictions_D_true, targets_D_true)
            loss_D_fake = loss_fn(predictions_D_fake, targets_D_fake)
            # get D's loss
            loss_D = loss_D_true + loss_D_fake
            loss_D = torch.tensor(loss_D, requires_grad=True)

            optimizer_D.zero_grad()                             
            loss_D.backward()                                   
            optimizer_D.step()
            lr_scheduler_D.step()

            # get G's loss
            loss_G = loss_fn(predictions_G, targets_G)
            # loss_G = similarity(predictions_D_fake, targets_D_true)
            loss_G = torch.tensor(loss_G, requires_grad=True)

            optimizer_G.zero_grad()                                             
            loss_G.backward()                                                   
            optimizer_G.step()  
            lr_scheduler_G.step()
            progress_bar.update(1)

            loss_G_total += loss_G.item()
            loss_D_total += loss_D.item()
            logger.info(f"{epoch}-{idx}: D's loss {loss_D} G's loss {loss_G}")
        
        avg_loss_G = loss_G_total / len(val_dataset_G['train']['conversations'])
        avg_loss_D = loss_D_total / len(val_dataset_G['train']['conversations'])
        logger.info(f"{epoch}: D's loss {avg_loss_D} G's loss {avg_loss_G}")
        wandb.log({'epoch': epoch, 'avg_loss_D': avg_loss_D, 'avg_loss_G': avg_loss_G})
        if (epoch + 1) % 10 == 0:
            _save(model_G, tokenizer_G, f"{ft_config_G.training_args.output_dir}{epoch}")
            _save(model_D, tokenizer_D, f"{ft_config_D.training_args.output_dir}{epoch}")
     
if __name__ == '__main__':
    app()
