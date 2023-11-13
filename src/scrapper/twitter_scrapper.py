import sys

from src.utils.path import get_main_path

sys.path.insert(0,"..")

import snscrape.modules.twitter as sntwitter
import pandas as pd


path = get_main_path()

# Using TwitterSearchScraper to scrape data and append tweets to list
# Creating list to append tweet data
temp_tweets_list = []

# number of tweets to be requested
request_tweets_target = 10_000


def create_dataframe_from_tweet_list(tweet_list: list) -> pd.DataFrame:
    tweets_df = pd.DataFrame(
        tweet_list,
        columns=[
            "id",
            "date",
            "user_name",
            "user_id",
            "user_displayname",
            "user_location",
            "user_followers",
            "user_friends",
            "user_statuses",
            "retweeted",
            "content",
            "lang",
            "is_missinginfo",
        ],
    )
    tweets_df.set_index("id", inplace=True)
    return tweets_df


key_words = "vacina teste"
# Using TwitterSearchScraper to scrape data and append tweets to list
for i, tweet in enumerate(
    sntwitter.TwitterSearchScraper(
        f"{key_words} since:2021-01-01 until:2021-05-31"
    ).get_items()
):
    if i >= request_tweets_target:
        break

    temp_tweets_list.append(
        [
            tweet.id,
            tweet.date,
            tweet.user.username,
            tweet.user.id,
            tweet.user.displayname,
            tweet.user.location,
            tweet.user.followersCount,
            tweet.user.friendsCount,
            tweet.user.statusesCount,
            tweet.retweetedTweet,
            tweet.rawContent,
            tweet.lang,
            None,
        ]
    )

    if i % 100 == 0:
        print(f"\rLoading tweets: {i}...", end="")
        temp_df = create_dataframe_from_tweet_list(temp_tweets_list)
        temp_tweets_list = []
        write_header = False
        if i == 0:  # write header the first time
            write_header = True
        file_path = path + f"\\datasets\\{key_words}-tweets.csv"
        temp_df.to_csv(
            file_path,
            mode="a",
            index=True,
            header=write_header,
        )


print(f"\rLoaded tweets: {i}")
print("TASK COMPLETED!!!")
