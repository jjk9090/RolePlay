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
    get_peft_model,
    PrefixTuningConfig
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
    BertTokenizer, BertModel,
    PretrainedConfig
)
from transformers import DataCollatorForSeq2Seq as _DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer as _Seq2SeqTrainer

from utils.utils import set_logger, load_model_and_tokenizer_after
from torch.utils.data import DataLoader
import wandb
wandb.init(project="hidden_weibo_random")

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

from configuration_chatglm import ChatGLMConfig
class PrefixEncoder1(torch.nn.Module):
    """
    The torch.nn model to encode the prefix
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    """

    def __init__(self, config: ChatGLMConfig):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            kv_size = config.num_layers * config.kv_channels * config.multi_query_group_num * 2
            self.embedding = torch.nn.Embedding(128, kv_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(kv_size, config.hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(config.hidden_size, kv_size)
            )
        else:
            print(config.num_layers * config.kv_channels * config.multi_query_group_num * 2)
            y = config.num_layers * config.kv_channels * config.multi_query_group_num * 2
            self.embedding = torch.nn.Embedding(config.pre_seq_len,
                                                y)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            print(prefix.device, prefix.shape)
            past_key_values = self.embedding(prefix)
        return past_key_values 

class PrefixEncoder(torch.nn.Module):
    """
    The torch.nn model to encode the prefix
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    """

    def __init__(self, config: ChatGLMConfig):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            kv_size = config.num_layers * config.kv_channels * config.multi_query_group_num * 2
            self.embedding = torch.nn.Embedding(config.pre_seq_len, kv_size)
            prefix_size = config.token_dim 
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(prefix_size, config.hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(config.hidden_size, kv_size)
            )
        else:
            print(config.num_layers * config.kv_channels * config.multi_query_group_num * 2)
            y = config.num_layers * config.kv_channels * config.multi_query_group_num * 2
            self.embedding = torch.nn.Embedding(config.pre_seq_len, y)

    def forward(self, prefix: torch.Tensor, config):
        if self.prefix_projection:
            prefix = prefix.to(self.trans[0].weight.dtype) 
            past_key_values = self.trans(prefix)
        else:
            print(prefix.device, prefix.shape)
            past_key_values = self.embedding(prefix)
        return past_key_values 

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
                output_hidden_states=True
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

def create_dataset(data_manager, ft_config, tokenizer):
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
    return train_dataset, val_dataset, test_dataset

def initial_train(data_manager, ft_config, tokenizer, model, auto_resume_from_checkpoint):
    train_dataset, val_dataset, test_dataset = create_dataset(data_manager, ft_config, tokenizer)
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

def create_batch_D(prompt_D_True, prompt_D_fake, tokenizer):
    path = "tmp.json"
    infos = []
    batch_True = {}
    batch_True['conversations'] = []
    batch_True['conversations'].append({
        "role": "user",
        "content": prompt_D_True
    })
            
    batch_True['conversations'].append({
        "role": "assistant",
        "content": "是"
    })
    # batch_True = tokenize_function_one(batch_True, tokenizer)

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
    infos.append(batch_True)
    infos.append(batch_fake)
    with open(path, "w", encoding="utf-8") as f:
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

def similarity(fake_profile_text, True_profile_text, model, tokenizer):
    fake_tokens = tokenizer.encode(fake_profile_text, add_special_tokens=False, truncation=True, padding=True, return_tensors="pt")
    True_tokens = tokenizer.encode(True_profile_text, add_special_tokens=False, truncation=True, padding=True, return_tensors="pt")

    fake_embeddings = model(fake_tokens).last_hidden_state.mean(dim=1).detach().numpy()
    True_embeddings = model(True_tokens).last_hidden_state.mean(dim=1).detach().numpy()

    similarity = cosine_similarity(fake_embeddings, True_embeddings)
    loss = 1 - similarity
    return loss

def get_prompt1(config):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    prefix_tokens = torch.arange(128).long()
    prefix_tokens = prefix_tokens.unsqueeze(0).expand(1, -1)
    prefix_encoder = PrefixEncoder1(config)
    past_key_values = prefix_encoder(prefix_tokens).type(torch.half).to(device)
    past_key_values = past_key_values.view(
        1,
        128,
        28 * 2,
        2,
        128
    )
    dropout = torch.nn.Dropout(0.1)
    # seq_len, b, nh, hidden_size
    past_key_values = dropout(past_key_values)
    past_key_values = past_key_values.permute([2, 1, 0, 3, 4]).split(2)
    return past_key_values

def get_prompt(config, prefix):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dropout = torch.nn.Dropout(0.1)

    prefix_encoder = PrefixEncoder(config).to(device)
    past_key_values = prefix_encoder(prefix, config).type(torch.half).to(device)
    shape = list(past_key_values.shape) 
    shape[0] = config.pre_seq_len

    past_key_values = past_key_values.detach()
    past_key_values.requires_grad_(False)
    past_key_values = past_key_values.resize_(shape)
    past_key_values.requires_grad_(True)
    past_key_values = past_key_values.view(
        1,
        config.pre_seq_len,
        config.num_layers * 2,
        config.multi_query_group_num,
        config.kv_channels
    )
    past_key_values = past_key_values.to(torch.float16)
    past_key_values = dropout(past_key_values)
    past_key_values = past_key_values.permute([2, 1, 0, 3, 4]).split(2)
    return past_key_values

def create_dataloader_BG(tokenizer, val_dataset, pre_len, next_len):
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        max_length=128, 
        return_tensors='pt',
    )
    remove_columns = val_dataset['train'].column_names
    # val_dataset['train'] = val_dataset['train'][pre_len:next_len]
    # tokenized_datasets = val_dataset['train'].map(
    #     tokenize_function, 
    #     batched=True, 
    #     remove_columns=remove_columns, 
    #     num_proc=16,
    #     fn_kwargs={"tokenizer": tokenizer}
    # )
    part_range = range(pre_len, next_len)
    part_dataset = val_dataset['train'].select(part_range)
    tokenized_datasets = part_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=remove_columns, 
        num_proc=16,
        fn_kwargs={"tokenizer": tokenizer}
    )
    tokenized_datasets.set_format("torch")
    val_dataloader = DataLoader(
        tokenized_datasets, shuffle=True, batch_size=1, collate_fn=data_collator
    )
    return val_dataloader, part_dataset

def create_dataloader(tokenizer, val_dataset):
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        max_length=128, 
        return_tensors='pt',
    )
    remove_columns = val_dataset['train'].column_names
    val_dataset['train'] = val_dataset['train']
    tokenized_datasets = val_dataset['train'].map(
        tokenize_function, 
        batched=True, 
        remove_columns=remove_columns, 
        num_proc=16,
        fn_kwargs={"tokenizer": tokenizer}
    )
    tokenized_datasets.set_format("torch")
    val_dataloader = DataLoader(
        tokenized_datasets, shuffle=True, batch_size=1, collate_fn=data_collator
    )
    return val_dataloader

@app.command()
def main(
        data_dir: Annotated[str, typer.Argument(help='')],
        data_dir_bg: Annotated[str, typer.Argument(help='')],
        model_dir: Annotated[
            str,
            typer.Argument(
                help='A string that specifies the model id of a pretrained model configuration hosted on huggingface.co, or a path to a directory containing a model configuration file.'
            ),
        ],
        config_file: Annotated[str, typer.Argument(help='')],
        config_file_bg: Annotated[str, typer.Argument(help='')],
        log_file: Annotated[str, typer.Argument(help='')],
        action_num: Annotated[int, typer.Argument(help='')],
        output_dir_bg: Annotated[str, typer.Argument(help='')],
        output_dir: Annotated[str, typer.Argument(help='')],
        auto_resume_from_checkpoint: str = typer.Argument(
            default='',
            help='If entered as yes, automatically use the latest save checkpoint. If it is a numerical example 12 15, use the corresponding save checkpoint. If the input is no, restart training'
        ),
):
    time_str = datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
    logger = set_logger(log_file, time_str)

    ft_config_BG = FinetuningConfig.from_file(config_file_bg)
    ft_config_BG.training_args.output_dir = f"{output_dir_bg}{time_str}/"
    tokenizer_BG, model_BG = load_tokenizer_and_model(model_dir, peft_config=ft_config_BG.peft_config)
    data_manager_BG = DataManager(data_dir_bg, ft_config_BG.data_config)
    test_dataset_BG = initial_train(data_manager_BG, ft_config_BG, tokenizer_BG, model_BG, auto_resume_from_checkpoint)

    ft_config_G = FinetuningConfig.from_file(config_file)
    ft_config_G.training_args.output_dir = f"{output_dir}{time_str}/"
    tokenizer_G, model_G = load_tokenizer_and_model(model_dir, peft_config=ft_config_G.peft_config)
    data_manager_G = DataManager(data_dir, ft_config_G.data_config)
    test_dataset_G = initial_train(data_manager_G, ft_config_G, tokenizer_G, model_G, auto_resume_from_checkpoint)

    bert_tokenizer = BertTokenizer.from_pretrained("model/bert-base-uncased")
    bert_model = BertModel.from_pretrained("model/bert-base-uncased")

    val_dataset_G = load_dataset(
        "json",
        data_dir=data_dir,
        data_files=ft_config_G.data_config.data_files.get(Split.VALIDATION),
        num_proc=ft_config_G.data_config.num_proc,
    )

    val_dataset_BG = load_dataset(
        "json",
        data_dir=data_dir_bg,
        data_files=ft_config_BG.data_config.data_files.get(Split.VALIDATION),
        num_proc=ft_config_BG.data_config.num_proc,
    )
    num_epochs = 300
    num_training_steps = num_epochs * len(val_dataset_G['train'])
    print(num_training_steps)

    parameters = list(model_G.parameters()) + list(model_BG.parameters())
    optimizer = AdamW(parameters, lr=5e-5)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    progress_bar = tqdm(range(num_training_steps))
    val_dataloader_G = create_dataloader(tokenizer_G, val_dataset_G)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_G.to(device)
    model_BG.to(device)

    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    config.update({
        'pre_seq_len': 256,
        'use_cache': False,
        'prefix_projection': True,
        'num_attention_heads': 32,
        'num_layers': 28,
        'multi_query_attention': True,
        'multi_query_group_num': 2,
        'use_cache': False,
        'rmsnorm': True,
        'encoder_hidden_size': 12,
        'num_virtual_tokens': 256,
        'token_dim': 256,
        'prefix_projection': True,
    })
    for epoch in range(num_epochs):
        loss_total = 0.0
        idx = 0
        for input_G, conv_G in zip(val_dataloader_G, val_dataset_G['train']['conversations']):
            idx += 1
            batch = {k: v.to(device) for k, v in input_G.items()}
            output_G = model_G(**batch, output_hidden_states=True)
            
            logits = output_G.logits 
            shape = logits.shape
            fake_profile = ""
            # get G's profile
            for i in range(shape[1]):
                if i == 0:
                    continue
                predicted_token_id = logits[0, i].argmax(axis=-1)
                fake_profile += tokenizer_G.decode(predicted_token_id)
            true_profile = conv_G[1]['content']

            loss1 = similarity(fake_profile, true_profile, bert_model, bert_tokenizer)
            loss1 = torch.tensor(loss1, requires_grad=True)

            last_layer_embedding = output_G.hidden_states[-1]
            prefix = last_layer_embedding.squeeze(dim=1)
            
            pre_seq_len = prefix.shape[0]
            token_dim = prefix.shape[1]
            prefix = prefix.to(device)
            config.token_dim = token_dim
            # add the BG's embedding
            past_key_values = get_prompt(config, prefix)
            model_BG.pre_seq_len = pre_seq_len 
            pre_len = action_num * (idx - 1)
            next_len = action_num * (idx)
            val_dataloader_BG, part_dataset_BG = create_dataloader_BG(tokenizer_BG, val_dataset_BG, pre_len, next_len)
            for input_BG, conv_BG in zip(val_dataloader_BG, part_dataset_BG['conversations']):
                batch_BG = {k: v.to(device) for k, v in input_BG.items()} 
                output_BG = model_BG(**batch_BG, past_key_values=past_key_values, output_hidden_states=True)
                logits_BG = output_BG.logits 
                shape_BG = logits_BG.shape
                fake_behavior = ""
                # get BG's behavior
                for i in range(shape_BG[1]):
                    if i == 0:
                        continue
                    predicted_token_id_BG = logits_BG[0, i].argmax(axis=-1)
                    fake_behavior += tokenizer_BG.decode(predicted_token_id_BG)
                true_behavior = conv_BG[1]['content']

                if not fake_behavior or not true_behavior:
                    continue
                loss2 = similarity(fake_behavior, true_behavior, bert_model, bert_tokenizer)
                loss2 = torch.tensor(loss2, requires_grad=True)

                loss = loss1 + loss2
                optimizer.zero_grad()                             
                loss.backward()                                   
                optimizer.step()
                lr_scheduler.step()

                logger.info(f"{epoch}-{idx}: G + BG 's loss {loss}")
                loss_total += loss.item()
        
        avg_loss = loss_total / len(val_dataset_G['train']['conversations']) / len(part_dataset_BG['conversations'])
        wandb.log({'avg_epoch': epoch, 'avg_loss': avg_loss})
        logger.info(f"{epoch}: G + BG's loss {avg_loss}")
        if (epoch + 1) % 5 == 0:
            _save(model_G, tokenizer_G, f"{ft_config_G.training_args.output_dir}{epoch}")
            _save(model_BG, tokenizer_BG, f"{ft_config_BG.training_args.output_dir}{epoch}")
           
if __name__ == '__main__':
    app()
