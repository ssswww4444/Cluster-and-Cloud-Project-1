import json
import re
import argparse
import numpy as np

TWEET_FILE_NAME = "tinyTwitter.json"
GRID_FILE_NAME = "melbGrid.json"

# Obtaining args from terminal
def get_args():
    
    parser = argparse.ArgumentParser(description="Extract features for files in the directory using openSMILE")
    
    # filenames
    parser.add_argument("-t", "--tweet_file", type = str, required = True, help = "Twitter data file")
    parser.add_argument("-g", "--grid_file", type = str, required = True, help = "Grid data file")

    args = parser.parse_args()
    
    return args

def read_grid(grid_file):
    with open(grid_file, "r") as f:
        grid_data = json.load(f)
        f.close()
    return grid_data

def read_tweet(tweet_file):
    with open(tweet_file, "r") as f:
        tweet_data = []
        for line in f:
            fixed_line = re.sub(r"ObjectId\( (.*) \)", r"\g<1>",line)
            tweet_data.append(json.loads(fixed_line))
        f.close()
    return tweet_data

def get_tweet_grid(coordinate, grid_data):
    grids = grid_data["features"]

    for grid in grids:
        # 4 points of each grid
        xmin = grid["properties"]["xmin"]
        xmax = grid["properties"]["xmax"]
        ymin = grid["properties"]["ymin"]
        ymax = grid["properties"]["ymax"]

        x = coordinate[0]
        y = coordinate[1]

        # not need to deal with the boundary problem because of the order of the grids
        if x >= xmin and x <= xmax and y >= ymin and y <= ymax:
            return grid["properties"]["id"]

    # not in any grid
    return "Other"

def get_filtered_tweets(tweet_data, grid_data):

    filtered_tweet_data = []

    # iterate through each tweet
    for tweet in tweet_data:

        # coordinates
        if tweet["coordinates"] == None:
            continue   # skip this tweet if no coordinates
        coordinates = tweet["coordinates"]["coordinates"]

        grid = get_tweet_grid(coordinates, grid_data)

        # skip this tweet if not in the grids
        if grid == "Other":
            continue

        # hashtags
        hashtags = tweet["entities"]["hashtags"]

        # only using the grid and hashtags info of tweets
        filtered_tweet_data.append({"grid": grid, "hashtags": hashtags})

    return filtered_tweet_data

def stat_tweet(tweet_data):
    grid_dict = {}
    for tweet in tweet_data:
        grid = tweet["grid"]
        
        if grid not in grid_dict:
            grid_dict[grid] = {"num_posts": 0, "hashtags": {}}

        # increment the number of posts
        grid_dict[grid]["num_posts"] = grid_dict[grid]["num_posts"] + 1

        # add hashtags if available
        for hashtag in tweet["hashtags"]:
            text = hashtag["text"]
            grid_dict[grid]["hashtags"][text] = grid_dict[grid]["hashtags"].get(text, 0) + 1   

    return grid_dict

def main():

    args = get_args()

    # list of dict
    tweet_data = read_tweet(args.tweet_file)
    grid_data = read_grid(args.grid_file)

    filtered_tweet_data = get_filtered_tweets(tweet_data, grid_data)

    # get statistics
    grid_dict = stat_tweet(filtered_tweet_data)
    grid_ls = sorted(grid_dict.items(), key=lambda x: x[1]["num_posts"], reverse = True)

    # TASK 1
    print("TASK - 1")
    for tuple in grid_ls:
        print("{}: {} posts".format(tuple[0], tuple[1]["num_posts"]))
        
    # TASK 2
    print("TASK - 2")
    for tuple in grid_ls:
        hashtag_dict = tuple[1]["hashtags"]
        hashtag_ls = sorted(hashtag_dict.items(), key=lambda x: x[1], reverse = True)[:5]  # take top 5
        print("{}: {}".format(tuple[0], hashtag_ls))

# If running the file directly
if __name__ == "__main__":
    main()