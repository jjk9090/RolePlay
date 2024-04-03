def create_config():
    folder = "/data/wl/hxy/RolePlay/dataset/weibo/like_retweet/"
    interaction_path = folder + "interaction.csv"
    item_path = folder + "item.csv"
    userinfo_path = folder + "userinfo_ad.csv"
    # G生成的文件路径
    fake_users_path = "/data/wl/hxy/RolePlay/finetune/dataset/exp/G_fake_profiles.csv"
    new_item_path = ""
    config = {}
    config['userinfo_path'] = userinfo_path
    config['item_path'] = item_path
    config['new_item_path'] = new_item_path
    config['interaction_path'] = interaction_path
    config['fake_user_path'] = fake_users_path
    return config

def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def load_model_and_tokenizer(ft_type, model_path):
    if ft_type == "LORA":
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device='cuda')
        model = model.eval()
    else:
        CHECKPOINT_PATH = "/data/wl/hxy/RolePlay/finetune/output/weibo/out_2024-03-27_16-07/checkpoint-3000"
        # 载入Tokenizer
        tokenizer = AutoTokenizer.from_pretrained("model/THUDM/chatglm3-6b", trust_remote_code=True)
        config = AutoConfig.from_pretrained("model/THUDM/chatglm3-6b", trust_remote_code=True, pre_seq_len=128)
        model = AutoModel.from_pretrained("model/THUDM/chatglm3-6b", config=config, trust_remote_code=True)
        # 加载新的权重
        new_prefix_state_dict = {}
        prefix_state_dict = torch.load(os.path.join(CHECKPOINT_PATH, "pytorch_model.bin"))
        for k, v in prefix_state_dict.items():
            if k.startswith("embedding.weight"):
                new_prefix_state_dict['embedding.weight'] = v
        model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
        model = model.cuda().quantize(4)
        model = model.half()
        model.transformer.prefix_encoder.float()
        model = model.eval()
    return model, tokenizer
