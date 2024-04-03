## 安装ChatGLM3
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download THUDM/chatglm3-6b --local-dir THUDM/chatglm3-6b
## 命令行输入
`cd finetune`
## finetune
G finetune
```
CUDA_VISIBLE_DEVICES=2 python finetune_hf.py  dataset/weibo/like_retweet/G  model/THUDM/chatglm3-6b  configs/lora.yaml
```
D finetune
```
CUDA_VISIBLE_DEVICES=2 python finetune_hf.py  dataset/weibo/like_retweet/D  model/THUDM/chatglm3-6b  configs/lora.yaml
```

## lora合并
G
```
CUDA_VISIBLE_DEVICES=2 python model_export_hf.py output/weibo/G/_80/ --out-dir ./model/chatglm3-6b_G
```
D
```
CUDA_VISIBLE_DEVICES=2 python model_export_hf.py output/weibo/D/_80/ --out-dir ./model/chatglm3-6b_D
```

## G生成profile
```
CUDA_VISIBLE_DEVICES=2 python generate_res_G.py --ft-type LORA --model-path ../finetune/model/chatglm3-6b_G
```
位于文件：
dataset/exp/G_fake_profiles.csv

