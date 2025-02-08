import tweepy
import time
import datetime
import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Retrieve the bearer token from environment variables
bearer_token = os.getenv('BEARER_TOKEN')

# Initialize the Tweepy client
client = tweepy.Client(bearer_token)

# Ensure the bearer token is available
if bearer_token is None:
    print("Error: BEARER_TOKEN is not set in the environment variables.")
    exit(1)

def handle_rate_limit(e):
    """
    Handles Twitter API rate limits by extracting the reset time from response headers
    and waiting until the limit resets before retrying.

    Args:
        e (tweepy.errors.TooManyRequests): The exception raised when rate limits are exceeded.
    """
    reset_timestamp = int(e.response.headers['x-rate-limit-reset'])
    reset_time = datetime.datetime.utcfromtimestamp(reset_timestamp)

    # Calculate wait time before retrying
    current_time = datetime.datetime.utcnow()
    wait_time = (reset_time - current_time).total_seconds()

    print(f"Rate limit exceeded. Waiting for {wait_time:.2f} seconds until {reset_time}.")
    time.sleep(wait_time)


def fetch_tweets(query, max_results, next_token=None, fetched=0, tweets=None):
    """
    Fetches recent tweets based on a given query.

    Args:
        query (str): The search query for fetching tweets.
        max_results (int): The maximum number of tweets to fetch.
        next_token (str, optional): The token for paginating results. Defaults to None.
        fetched (int, optional): The number of tweets already fetched. Defaults to 0.
        tweets (list, optional): A list to store fetched tweet texts. Defaults to None.

    Returns:
        list: A list of tweet texts retrieved from the API.
    """
    if tweets is None:
        tweets = []

    try:
        response = client.search_recent_tweets(query=query, max_results=(max_results - fetched), next_token=next_token)

        if response.data:
            fetched += len(response.data)  # Update fetched count

            for tweet in response.data:
                print(f"Tweet ID: {tweet.id}")
                print(f"Tweet Text: {tweet.text}")
                tweets.append(tweet.text)

        if 'next_token' in response.meta and fetched < max_results:
            print("Next page exists, fetching more tweets...")
            tweets = fetch_tweets(query, max_results - fetched, next_token=response.meta['next_token'], fetched=fetched, tweets=tweets)

        return tweets

    except tweepy.errors.TooManyRequests as e:
        handle_rate_limit(e)
        return fetch_tweets(query, max_results, next_token, fetched=fetched, tweets=tweets)

    except tweepy.TweepyException as e:
        print(f"Error: {e}")
        return tweets
