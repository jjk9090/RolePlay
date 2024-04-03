
from transformers import AutoTokenizer, AutoModel, AutoConfig
import json
import pandas as pd
from datetime import datetime
import torch 
import typer
from typing import Annotated, Union
import os
app = typer.Typer(pretty_exceptions_show_locals=False)
import csv
import json
from data.weibo_data import WeiboData
from utils import load_model_and_tokenizer, create_config, create_directory_if_not_exists

def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def weibo_predict_profile(file_path, user_file_name, model, tokenizer, data: WeiboData):
    df = pd.DataFrame(columns=['name', 'user_id', 'gender', 'age', 'personality'])
    users = {} 
    real_users = data.users
    with open(file_path, 'r', encoding="utf-8") as f: 
        for line in f: 
            data = json.loads(line)
            
            for conv in data['conversations']:
                if conv['role'] == 'user':
                    prompt = conv['content']
                    start_index = prompt.find("我在微博上的网名是 ") + len("我在微博上的网名是 ")
                    end_index = prompt.find(".", start_index)
                    name = prompt[start_index:end_index]
            
                    profile_start_index = prompt.find("请以下列格式清楚作答:\n") + len("请以下列格式清楚作答:\n")
                    profile_end_index = prompt.find(":", profile_start_index)
                    profile = prompt[profile_start_index:profile_end_index]
                    print(name, profile)
                    response, history = model.chat(tokenizer, prompt, history=[], max_new_tokens=512)
                    if name not in users:
                        users[name] = {}
                    if profile not in users[name]:
                        users[name][profile] = response 

    for name, user in users.items():
        gender = user['性别'] if '性别' in user else ''
        age = user['年龄'] if '年龄' in user else ''
        personality = user['性格'] if '性格' in user else ''
        user_id = real_users[name]['uid']
        row = [name, user_id, gender, age, personality]
        df.loc[len(df)] = row
        df.to_csv(user_file_name, index=False)

@app.command()
def main(
    ft_type: Annotated[str, typer.Option(help='')],
    model_path: Annotated[
        str,
        typer.Option(
            help='A string that specifies the model id of a pretrained model configuration hosted on huggingface.co, or a path to a directory containing a model configuration file.'
        ),
    ],
):
    G_model, G_tokenizer = load_model_and_tokenizer(ft_type, model_path)
    
    folder_path = "/data/wl/hxy/RolePlay/dataset/weibo/res/"
    create_directory_if_not_exists(folder_path)

    config = create_config()
    data = WeiboData(config)
    
    G_profile_file_path = "/data/wl/hxy/RolePlay/finetune/dataset/weibo/like_retweet/G/dev.json"
    G_fake_user_file_name = folder_path + "G_fake_profiles.csv"
    weibo_predict_profile(G_profile_file_path, G_fake_user_file_name, G_model, G_tokenizer, data)
    
if __name__ == '__main__':
    app()