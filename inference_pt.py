from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch
import os
CHECKPOINT_PATH = "output/weibo/out_2024-03-27_10-10/checkpoint-3000"
# 载入Tokenizer
tokenizer = AutoTokenizer.from_pretrained("model/THUDM/chatglm3-6b", trust_remote_code=True)

config = AutoConfig.from_pretrained("model/THUDM/chatglm3-6b", trust_remote_code=True, pre_seq_len=128)
model = AutoModel.from_pretrained("model/THUDM/chatglm3-6b", config=config, trust_remote_code=True)
prefix_state_dict = torch.load(os.path.join(CHECKPOINT_PATH, "pytorch_model.bin"))
new_prefix_state_dict = {}
for k, v in prefix_state_dict.items():
    if k.startswith("transformer.prefix_encoder."):
        new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

# 之后根据需求可以进行量化
# Comment out the following line if you don't use quantization
model = model.quantize(4)
model = model.half().cuda()
model.transformer.prefix_encoder.float()
model = model.eval()

response, history = model.chat(tokenizer, "你好", history=[])