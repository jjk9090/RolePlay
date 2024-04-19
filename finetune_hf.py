# -*- coding: utf-8 -*-
import os
import copy 
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
import pandas as pd
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

from utils.utils import set_logger, load_model_and_tokenizer_after, create_directory_if_not_exists
from torch.utils.data import DataLoader 
from chatglm_modeling_cls import ChatGLMforPolicyModel
import wandb
wandb.init(project="explicit_weibo_random")

logging.getLogger('transformers').setLevel(logging.WARNING)
ModelType = Union[PreTrainedModel, PeftModelForCausalLM]
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
app = typer.Typer(pretty_exceptions_show_locals=False)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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
        model_task: Optional[str] = None,
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
        if model_task == 'POLICY':
            model = ChatGLMforPolicyModel.from_pretrained(
                pretrained_model_name_or_path=model_dir,
                trust_remote_code=True,
                empty_init=False,
                use_cache=False,
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        elif peft_config.peft_type.name == "LORA":
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                trust_remote_code=True,
                empty_init=False,
                use_cache=False,
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

    print("start trainer")
    trainer.train()
    # test stage
    return test_dataset

def remove_substring(string, substring):
    parts = string.split(substring, 1)
    if len(parts) > 1:
        string = parts[0]
    return string

def _save(model, tokenizer, output_dir, state_dict=None):
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, state_dict=state_dict)
    tokenizer.save_pretrained(output_dir, state_dict=state_dict)

def tokenize_function(batch, tokenizer):
    max_input_length = 512 
    max_output_length = 512
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
            input_ids += new_input_ids[:400]
            loss_masks += new_loss_masks[:400]

        input_ids.append(tokenizer.eos_token_id)
        loss_masks = [False, *loss_masks]
        
        labels = []
        real_label = []
        for input_id, mask in zip(input_ids, loss_masks):
            if mask:
                labels.append(input_id)
            else:
                labels.append(-100)
    
        max_length = max_input_length + max_output_length + 1
        batched_input_ids.append(input_ids[:max_length])
        batched_labels.append(labels[:max_length])
    return {'input_ids': batched_input_ids, 'labels': batched_labels}

def get_one_info(user_content, assistant_content):
    info = {}
    info['conversations'] = []
    info['conversations'].append({
        "role": "user",
        "content": user_content
    })
            
    info['conversations'].append({
        "role": "assistant",
        "content": assistant_content
    })
    return info

def create_batch_D(prompt, fake_profile, conv):
    fake_profile = fake_profile.split('\n')[0]
    question_fake = (
            f"请判断[{fake_profile}]这些个人信息是否属于我吗？请注意这只是你的判断，你事先不知道我的信息。请以下列格式清楚作答:\n"
        +   f"是/否\n"
    )
    true_profile = conv[1]['content']
    true_profile = true_profile.split('\n')[0]
    question_true = (
            f"请判断[{true_profile}]这些个人信息是否属于我吗？请注意这只是你的判断，你事先不知道我的信息。请以下列格式清楚作答:\n"
        +   f"是/否\n"
    )

    prompt_true = prompt + question_true
    prompt_fake = prompt + question_fake
    infos = []
    
    infos.append(get_one_info(prompt_fake, '否'))
    infos.append(get_one_info(prompt_true, '是'))
    return pd.DataFrame(infos)

def create_batch_G(prompt, fake_profile, conv):
    infos = []
    fake_profile = fake_profile.split('\n')[0]
    question_fake = (
            f"请判断[{fake_profile}]这个profile是否属于我吗？请注意这只是你的判断，你事先不知道我的信息。请以下列格式清楚作答:\n"
        +   f"是/否\n"
    )

    prompt = prompt + question_fake
    infos.append(get_one_info(prompt, '是'))
    return pd.DataFrame(infos)

def pre_finetune(data_dir, model_dir, config_file, output_dir, time_str, auto_resume_from_checkpoint):
    ft_config = FinetuningConfig.from_file(config_file)
    ft_config.training_args.output_dir = f"{output_dir}{time_str}/"
    create_directory_if_not_exists(ft_config.training_args.output_dir)
    tokenizer, model = load_tokenizer_and_model(model_dir, peft_config=ft_config.peft_config)
    data_dir += 'all/'
    data_manager = DataManager(data_dir, ft_config.data_config)
    # initial_train(data_manager, ft_config, tokenizer, model, auto_resume_from_checkpoint)
    return model, tokenizer, ft_config

def get_lr_scheduler(num_training_steps, model):
    optimizer = AdamW(model.parameters(), lr=5e-5)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    return lr_scheduler, optimizer

def get_profile_dataloader(tokenizer, val_dataset_G):
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        max_length=128, 
        return_tensors='pt',
    )
    remove_columns = val_dataset_G['train'].column_names
    tokenized_datasets = val_dataset_G['train'].map(
        tokenize_function, 
        batched=True, 
        remove_columns=remove_columns, 
        num_proc=16,
        fn_kwargs={"tokenizer": tokenizer}
    )
    tokenized_datasets.set_format("torch")
    val_dataloader = DataLoader(
        tokenized_datasets, shuffle=True, batch_size=2, collate_fn=data_collator
    )
    return val_dataloader, data_collator

def decode_output(labels, output, tokenizer):
    shape = output.shape 
    decode_txt = ""
    # get G's profile
    shape_size = shape[0]
    i = 0
    for label, logit in zip(labels, output):
        # get masked target
        if label != torch.tensor(-100):
            predicted_token_id = output[i].argmax(axis=-1)
            decode_txt += tokenizer.decode(predicted_token_id)
        i += 1
    return decode_txt.replace("<0x0A>", "")

def create_tmp_dataloader2(data, tokenizer, data_collator):
    dataset = Dataset.from_pandas(data)
    remove_columns = dataset.column_names
    tokenized_tmp_datasets = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=remove_columns, 
        num_proc=16,
        fn_kwargs={"tokenizer": tokenizer}
    )
    tokenized_tmp_datasets.set_format("torch")
    tmp_dataloader = DataLoader(
        tokenized_tmp_datasets, shuffle=True, batch_size=1, collate_fn=data_collator
    )
    return tmp_dataloader

def train_discriminator(logger, model_G, model_D, labels, output_list, convs, tokenizer, optimizer, lr_scheduler, data_collator):
    batch_size = len(output_list)

    loss_fn = torch.nn.BCELoss(reduction='none')
    for output, conv, label in zip(output_list, convs, labels[:batch_size]):
        # Decode the output profile of model G 
        fake_profile = decode_output(label, output, tokenizer)
        prompt = conv[0]['content']
        prompt = remove_substring(prompt, "请推断出")
        # create two data to discriminator
        data = create_batch_D(prompt, fake_profile, conv)
        tmp_dataloader = create_tmp_dataloader2(data, tokenizer, data_collator)
        loss_true = 0
        loss_fake = 0
        j = 0
        true_label = torch.tensor([1]).to(device).to(torch.float32)
        fake_label = torch.tensor([0]).to(device).to(torch.float32)
        for tmp_batch in tmp_dataloader:
            tmp_batch = {k: v.to(device) for k, v in tmp_batch.items()}
            x = tmp_batch['labels']
            tmp_output = model_D(**tmp_batch)
            logits = tmp_output.logits
            if j == 0:
                loss_fake = loss_fn(logits[:, 0], fake_label)
            else:
                loss_true = loss_fn(logits[:, 1], true_label)
            j += 1
        loss = loss_true + loss_fake
        loss = torch.tensor(loss, requires_grad=True)
        logger.info(f"D 's loss {loss}")
        wandb.log({'loss_D': loss})

        optimizer.zero_grad()                             
        loss.backward()                                   
        optimizer.step()
        lr_scheduler.step()

def train_generator(logger, model_G, model_D, labels, batch, convs, tokenizer, optimizer, lr_scheduler, data_collator):
    batch_size = len(batch)   
    loss_fn = torch.nn.BCELoss(reduction='none')

    for input, conv, label in zip(batch, convs, labels[batch_size:]):
        output = model_G(**input)
        fake_profile = decode_output(label, output.logits, tokenizer)
        prompt = conv[0]['content']
        prompt = remove_substring(prompt, "请推断出")

        data = create_batch_G(prompt, fake_profile, conv)
        tmp_dataloader = create_tmp_dataloader(data, tokenizer, data_collator)
        tmp_batch = next(iter(tmp_dataloader))
        tmp_batch = {k: v.to(device) for k, v in tmp_batch.items()}

        tmp_output = model_D(**tmp_batch)
        logits = tmp_output.logits
        true_label = torch.tensor([1]).to(device).to(torch.float32)
        loss = torch.sub(0, loss_fn(logits[:, 1], true_label))
        logger.info(f"G 's loss {loss}")
        wandb.log({'loss_G': loss})

        optimizer.zero_grad()                             
        loss.backward()                                   
        optimizer.step()
        lr_scheduler.step()

def get_two_batch(batch):
    batch1 = {}
    tmp_batch = {}
    for key, value in batch.items():
        v = int(len(value) / 2)
        batch1[key] = value[:v]
        tmp_batch[key] = value[v:]
    
    batch2 = []
    for i in range(len(tmp_batch['input_ids'])):
        tmp = {}
        for key, value in tmp_batch.items():
            tmp[key] = value[i:i + 1]
        batch2.append(tmp)

    return batch1, batch2

@app.command()
def main(
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
    model_G, tokenizer_G, ft_config_G = pre_finetune(data_dir, model_dir, config_file, output_dir, time_str, auto_resume_from_checkpoint)
    
    ft_config_D = FinetuningConfig.from_file(config_file_d)
    ft_config_D.training_args.output_dir = f"{output_dir_d}{time_str}/"
    create_directory_if_not_exists(ft_config_D.training_args.output_dir)
    tokenizer_D, model_D = load_tokenizer_and_model(model_dir, peft_config=ft_config_D.peft_config, model_task='POLICY')

    val_dataset_G = load_dataset(
        "json",
        data_dir=data_dir + mask_type,
        data_files=ft_config_G.data_config.data_files.get(Split.VALIDATION),
        num_proc=ft_config_G.data_config.num_proc,
    )

    # Acquire data for reinforcement training
    val_dataloader_G, data_collator_G = get_profile_dataloader(tokenizer_G, val_dataset_G)

    num_epochs = 300
    num_training_steps = num_epochs * len(val_dataloader_G)
    lr_scheduler_G, optimizer_G = get_lr_scheduler(num_training_steps, model_G)
    lr_scheduler_D, optimizer_D = get_lr_scheduler(num_training_steps, model_D)

    progress_bar = tqdm(range(num_training_steps))
    
    model_G.to(device)
    model_D.to(device)
    for epoch in range(num_epochs):
        idx = 0
        for input in val_dataloader_G:
            idx += 1
            batch = {k: v.to(device) for k, v in input.items()}
            batch_copy = copy.deepcopy(batch)
            # Split the batch into halves
            batch1, batch2 = get_two_batch(batch)
            # Feed the first batch to G and obtain G's output
            output = model_G(**batch1)
            
            labels = batch_copy['labels']
            batch_size = len(labels)
            convs = val_dataset_G['train']['conversations'][(batch_size) * (idx - 1): batch_size * idx]

            # train D
            train_discriminator(logger, model_G, model_D, labels, output.logits, convs, tokenizer_G, optimizer_D, lr_scheduler_D, data_collator_G)

            # train G
            train_generator(logger, model_G, model_D, labels, batch2, convs, tokenizer_G, optimizer_G, lr_scheduler_G, data_collator_G)
            progress_bar.update(1)

        if epoch % 10 == 0:
            _save(model_G, tokenizer_G, f"{ft_config_G.training_args.output_dir}{epoch}")
            _save(model_D, tokenizer_D, f"{ft_config_D.training_args.output_dir}{epoch}")
     
if __name__ == '__main__':
    app()
