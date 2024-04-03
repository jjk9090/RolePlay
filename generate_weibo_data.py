import csv
import json
from data.weibo_data import WeiboData
from utils import create_config
import random
def generate_random_description():
    interests = ["美妆", "健康", "娱乐", "时尚", "艺术", "旅行", "电影"]
    personality_traits = ["积极向上的生活态度", "好奇心旺盛", "乐观", "开朗", "充满活力", "善于表达"]
    interests_num = random.randint(2, 4)  
    selected_interests = random.sample(interests, interests_num)  
    personality = random.choice(personality_traits)  
    description = f"我{personality}，爱好是关注{', '.join(selected_interests)}多个领域，对{', '.join(selected_interests)}感兴趣，表现出{personality}的生活态度。"
    return description

def generate_random_personality():
    descriptions = []
    for _ in range(500):
        description = generate_random_description()
        descriptions.append(description)
    
    return descriptions

def get_random_different_number(x):
    while True:
        y = random.randint(10, 99)
        if y != x:
            return y

def generate_behaviors(data: WeiboData):
    interactions = data.interactions
    items = data.items
    users = data.users
    infos = []
    personalitys = generate_random_personality()
    for user_id, user in users.items():
        if user_id in interactions:
            inters = interactions[user_id]
        else:
            continue 
        
        have_action = False
        prompt = f"我在微博上的网名是 {user['name']}，并且我在微博平台的行为如下：\n\n"
        for item_id, actions in inters.items():
            if item_id in items:
                item = items[item_id]
            else:    
                continue
            have_action = True
            moves = [('Attitude', '点赞'), ('Repost', '转发'), ('Comment', '评论')]
            actions = [move[1] if move[0] in actions else "" for move in moves]
            behaviors = "、".join(filter(None, actions)) 
            prompt += (
                    f"在看了微薄平台的一个内容为：“{item['content']}”的帖子后, "
                +   f"我{behaviors}了这则帖子。\n"
            )

        if not have_action:
            continue
        profiles = [('性别', 'gender'), ('年龄', 'age'), ('性格', 'personality')]
        for profile in profiles:
            true_profile = user[profile[1]]
            if profile[1] == 'gender':
                random_profile = "female" if true_profile == "male" else "male"
            if profile[1] == 'age':
                random_profile = get_random_different_number(true_profile)
            if profile[1] == 'personality':
                random_profile = personalitys[random.randint(0, 499)]
            

            question = (
                    f"请判断[{profile[0]}: {true_profile}]这个profile是否属于我吗？请注意这只是你的判断，你事先不知道我的信息。请以下列格式清楚作答:\n"
                +   f"是/否\n"
            )
            user_content = prompt + question
            info1 = {}
            info1['conversations'] = []
            info1['conversations'].append({
                "role": "user",
                "content": user_content
            })
                    
            user_profile = f"{profile[0]}: {user[profile[1]]}\n"
            info1['conversations'].append({
                "role": "assistant",
                "content": "是"
            })

            question = (
                    f"请判断[{profile[0]}: {random_profile}]这个profile是否属于我吗？请注意这只是你的判断，你事先不知道我的信息。请以下列格式清楚作答:\n"
                +   f"是/否\n"
            )
            user_content = prompt + question
            info2 = {}
            info2['conversations'] = []
            info2['conversations'].append({
                "role": "user",
                "content": user_content
            })
                    
            info2['conversations'].append({
                "role": "assistant",
                "content": "否"
            })

            random_num = random.randint(1, 2)
            if random_num == 1:
                infos.append(info1)
                infos.append(info2)
            else:
                infos.append(info2)
                infos.append(info1)
    return infos

def create_D_dataset(config, data):
    infos = generate_behaviors(data)
    print(len(infos))
    train_path = "../finetune/dataset/weibo/like_retweet/D/train.json"
    dev_path = "../finetune/dataset/weibo/like_retweet/D/dev.json"
    with open(train_path, "w") as f:
        for e in infos[:714]:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    with open(dev_path, "w") as f:
        for e in infos[714:]:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

def main():
    config = create_config()
    data = WeiboData(config)
    create_D_dataset(config, data)

if __name__ == "__main__":
    main()