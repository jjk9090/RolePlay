import csv
class TwitterData():
    def __init__(self, config):
        self.users = {}
        self.tweets = {}
        self.mbti_labels = {}
        self.load_users(config['user_path'])
        self.load_tweets(config['tweet_path'])
        self.load_mbti(config['mbti_path'])
    
    def load_users(self, user_path):
        with open(user_path, "r", newline="") as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                id,id_str,name,screen_name,location,description,verified,followers_count,friends_count,listed_count,favourites_count,statuses_count,number_of_quoted_statuses,number_of_retweeted_statuses,total_retweet_count,total_favorite_count,total_hashtag_count,total_url_count,total_mentions_count,total_media_count,number_of_tweets_scraped,average_tweet_length,average_retweet_count,average_favorite_count,average_hashtag_count,average_url_count,average_mentions_count,average_media_count = row 
                self.users[id] = {
                    "name": name,
                    "screen_name": screen_name,
                    "location": location,
                    "description": description,
                    "verified": verified,
                    "followers_count": followers_count,
                    "friends_count": friends_count,
                    "listed_count": listed_count,
                    "favourites_count": favourites_count,
                    "statuses_count": statuses_count,
                    "number_of_quoted_statuses": number_of_quoted_statuses,
                    "number_of_retweeted_statuses": number_of_retweeted_statuses,
                    "total_retweet_count": total_retweet_count,
                    "total_favorite_count": total_favorite_count,
                    "total_hashtag_count": total_hashtag_count,
                    "total_url_count": total_url_count,
                    "total_mentions_count": total_mentions_count,
                    "total_media_count": total_media_count,
                    "number_of_tweets_scraped": number_of_tweets_scraped,
                    "average_tweet_length": average_tweet_length,
                    "average_retweet_count": average_retweet_count,
                    "average_favorite_count": average_favorite_count,
                    "average_hashtag_count": average_hashtag_count,
                    "average_url_count": average_url_count,
                    "average_mentions_count": average_mentions_count,
                    "average_media_count": average_media_count
                }

    def load_tweets(self, tweet_path):
        with open(tweet_path, "r", newline="") as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                if len(row) >= 2:
                    self.tweets[row[0]] = row[1:]
    
    def load_mbti(self, mbti_path):
         with open(mbti_path, "r", newline="") as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                id, mbti_personality = row
                self.mbti_labels[id] = mbti_personality
