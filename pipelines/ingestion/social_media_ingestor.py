import os
import logging
from typing import Dict, Any, List
import time
from datetime import datetime, timedelta
import pandas as pd
import tweepy
from facebook_scraper import get_posts
from instagram_private_api import Client, ClientCompatPatch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SocialMediaIngestor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.twitter_api = self._init_twitter_api()
        self.instagram_api = self._init_instagram_api()

    def _init_twitter_api(self):
        auth = tweepy.OAuthHandler(
            os.getenv("TWITTER_CONSUMER_KEY"), os.getenv("TWITTER_CONSUMER_SECRET")
        )
        auth.set_access_token(
            os.getenv("TWITTER_ACCESS_TOKEN"), os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
        )
        return tweepy.API(auth)

    def _init_instagram_api(self):
        return Client(os.getenv("INSTAGRAM_USERNAME"), os.getenv("INSTAGRAM_PASSWORD"))

    def ingest_twitter_data(self, query: str, count: int = 100) -> List[Dict[str, Any]]:
        logger.info(f"Ingesting Twitter data for query: {query}")
        tweets = []
        try:
            for tweet in tweepy.Cursor(self.twitter_api.search_tweets, q=query).items(
                count
            ):
                tweets.append(
                    {
                        "id": tweet.id,
                        "text": tweet.text,
                        "user": tweet.user.screen_name,
                        "created_at": tweet.created_at,
                        "retweet_count": tweet.retweet_count,
                        "favorite_count": tweet.favorite_count,
                    }
                )
            logger.info(f"Successfully ingested {len(tweets)} tweets")
        except tweepy.TweepError as e:
            logger.error(f"Error ingesting Twitter data: {str(e)}")
        return tweets

    def ingest_facebook_data(
        self, page_name: str, pages: int = 1
    ) -> List[Dict[str, Any]]:
        logger.info(f"Ingesting Facebook data for page: {page_name}")
        posts = []
        try:
            for post in get_posts(page_name, pages=pages):
                posts.append(
                    {
                        "id": post["post_id"],
                        "text": post["text"],
                        "time": post["time"],
                        "likes": post["likes"],
                        "comments": post["comments"],
                        "shares": post["shares"],
                    }
                )
            logger.info(f"Successfully ingested {len(posts)} Facebook posts")
        except Exception as e:
            logger.error(f"Error ingesting Facebook data: {str(e)}")
        return posts

    def ingest_instagram_data(
        self, username: str, max_posts: int = 20
    ) -> List[Dict[str, Any]]:
        logger.info(f"Ingesting Instagram data for user: {username}")
        posts = []
        try:
            user_id = self.instagram_api.username_info(username)["user"]["pk"]
            results = self.instagram_api.user_feed(user_id)
            items = results.get("items", [])

            for item in items[:max_posts]:
                post = ClientCompatPatch.media(item)
                posts.append(
                    {
                        "id": post["id"],
                        "caption": post.get("caption", {}).get("text", ""),
                        "likes": post["like_count"],
                        "comments": post["comment_count"],
                        "created_at": datetime.fromtimestamp(post["taken_at"]),
                    }
                )
            logger.info(f"Successfully ingested {len(posts)} Instagram posts")
        except Exception as e:
            logger.error(f"Error ingesting Instagram data: {str(e)}")
        return posts

    def save_to_csv(self, data: List[Dict[str, Any]], filename: str):
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        logger.info(f"Data saved to {filename}")


def main():
    config = {
        "twitter_query": "#CustomerService",
        "facebook_page": "SomeCompany",
        "instagram_username": "somecompany",
    }

    ingestor = SocialMediaIngestor(config)

    twitter_data = ingestor.ingest_twitter_data(config["twitter_query"])
    ingestor.save_to_csv(twitter_data, "twitter_data.csv")

    facebook_data = ingestor.ingest_facebook_data(config["facebook_page"])
    ingestor.save_to_csv(facebook_data, "facebook_data.csv")

    instagram_data = ingestor.ingest_instagram_data(config["instagram_username"])
    ingestor.save_to_csv(instagram_data, "instagram_data.csv")


if __name__ == "__main__":
    main()
