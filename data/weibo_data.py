import csv
class WeiboData():
    def __init__(self, config):
        self.interactions = {}
        self.items = {}
        self.users = {}
        self.fake_users = {}
        self.load_users(config['userinfo_path'])
        self.load_interactions(config['interaction_path'])
        self.load_items(config['item_path'])
        if config['new_item_path'] != "":
            self.expand_items(config['new_item_path'])
        # self.load_fake_users(config['fake_user_path'])

    def load_users(self, userinfo_path):
        with open(userinfo_path, "r", newline="") as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                row = row[:13]
                uid, name, gender, age, cash, interest, feature, personality, occupation, fun_number, following_number, following, history = row
                self.users[uid] = {
                    'name': name,
                    'gender': "male" if gender == "1" else "female",
                    'age': age,
                    'personality': personality,
                    'fun_number': fun_number,
                    'following_number': following_number
                }
                self.users[name] = {
                    'uid': uid,
                    'gender': gender,
                    'age': age,
                    'personality': personality,
                    'fun_number': fun_number,
                    'following_number': following_number
                }
    
    def load_items(self, item_path):
        with open(item_path, "r", newline="") as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                row = row[:5]
                mid, poster, description, content, timestamp = row
                self.items[mid] = {'poster': poster, 'description': description, 'content': content}
    
    def load_interactions(self, interaction_path):
        with open(interaction_path, "r", newline="") as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                user_id,item_id,action,timestamp = row
                if user_id not in self.interactions:
                    self.interactions[user_id] = {}
                if item_id not in self.interactions[user_id]:
                    self.interactions[user_id][item_id] = {}

                self.interactions[user_id][item_id][action] = action
    
    def expand_items(self, item_path):
       with open(item_path, "r", newline="") as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                row = row[:5]
                mid, poster, description, content, timestamp = row
                if mid not in self.items:
                    self.items[mid] = {'poster': poster, 'description': description, 'content': content}
    
    def load_fake_users(self, file_path):
        with open(file_path, "r", newline="") as file:
            reader = csv.reader(file) 
            next(reader)
            for row in reader:
                name, user_id, gender, age, personality = row 
                if name in self.users:
                    user_id = self.users[name]['uid']
                    self.fake_users[user_id] = {
                        "name": name,
                        "gender": gender,
                        "age": age,
                        "personality": personality
                    }