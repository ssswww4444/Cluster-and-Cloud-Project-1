import json
import argparse
import numpy as np
from mpi4py import MPI
from collections import Counter

def get_args():
    """ Obtaining args from terminal """
    parser = argparse.ArgumentParser(description="Processing tweets")
    # filenames
    parser.add_argument("-t", "--tweet_file", type = str, required = True, help = "Twitter data file")
    parser.add_argument("-g", "--grid_file", type = str, required = True, help = "Grid data file")
    args = parser.parse_args()
    
    return args

def fix_line(line):
    """ Fix the line readed for coverting into json """
    # need to end with "}"
    i = -1
    while line[i] != "}":
        i -= 1
    return line[:i+1]

def get_hashtags(text):
    """ extract hashtags in the text following pattern <space>#something<space> """
    hashtags = set()
    word_ls = text.split()

    # indices
    start = 1
    end = -1
    if text[0] == " ":
        start = 0
    if text[-1] == " ":
        end = len(word_ls)

    for word in word_ls[start:end]:
        if word[0] == "#":
            hashtags.add(word.lower())

    return list(hashtags)

def get_coordinates(tweet):
    """ get the coordinates by search over all 3 places """
    
    # place 1: doc - coordinates - coordinates
    try:
        target = tweet["doc"]["coordinates"]["coordinates"]
        if target != None:
            return target
    except:
        # no need to handle
        pass

    # place 2: value - geometry - coordinates 
    try:
        target = tweet["value"]["geometry"]["coordinates"]
        if target != None:
            return target
    except:
        pass

    # place 3: doc - geo - coordinates (deprecated, in form (y,x))
    try:
        target = tweet["doc"]["geo"]["coordinates"]
        if target != None:
            return target[::-1]  # swap to get (x,y)
    except:
        pass

    # not found any coordinates
    return None

def get_tweet_grid(coordinates, grid_data):
    """ Check which grid the tweet is in """
    # representing each grid by a number
    grid_num = 0

    # current coordinates
    x = coordinates[0]
    y = coordinates[1]

    for grid in grid_data:
        # 4 points of each grid
        xmin = grid["properties"]["xmin"]
        xmax = grid["properties"]["xmax"]
        ymin = grid["properties"]["ymin"]
        ymax = grid["properties"]["ymax"]

        # not need to deal with the boundary problem because of the order of the grids
        if x >= xmin and x <= xmax and y >= ymin and y <= ymax:
            return grid_num
        
        grid_num += 1

    # not in any grid
    return -1


def read_tweet(tweet_file, grid_data, rank, size):
    """ Read twitter data from file """
    with open(tweet_file, "r", encoding="utf-8") as f:

        # remove the header line
        tweet_data = []
        for i, line in enumerate(f):

            # first / last line
            if i == 0 or (i % (size) != rank):  # assign line according to the rank
                continue
            elif line[0] != "{":
                continue

            tweet = json.loads(fix_line(line))

            coordinates = get_coordinates(tweet)

            # skip the tweet if no geocode
            if coordinates == None:
                continue
            
            grid = get_tweet_grid(coordinates, grid_data)

            # out of victoria
            if grid == -1:
                continue

            hashtags = get_hashtags(tweet["doc"]["text"])

            # only using the grid and hashtags info of tweets
            filtered_dict = {"grid": grid, 
                             "hashtags": hashtags}

            tweet_data.append(filtered_dict)
        f.close()

    return tweet_data

def read_grid(grid_file):
    """ Read grid data from file """
    with open(grid_file, "r") as f:

        grid_data = json.load(f)["features"]
        f.close()

    return grid_data

def stat_tweet(tweet_data, num_grid):
    """ Find statistics for both task 1 and 2
    Task 1 - post counts for each grid
    Task 2 - top 5 hashtags for each grid 
    """
    # numpy array for post count
    grid_post_count = np.zeros(num_grid, dtype=int)
    grid_hashtag_dict = {}
    for i in range(num_grid):
        grid_hashtag_dict[i] = Counter()

    # iterate through each tweet
    for tweet in tweet_data:

        # **************** TASK 1 ****************

        # count post
        grid_id = tweet["grid"]
        grid_post_count[grid_id] += 1

        # **************** TASK 2 ****************

        # add hashtags if available
        for hashtag in tweet["hashtags"]:
            grid_hashtag_dict[grid_id][hashtag] += 1 

    return grid_post_count, grid_hashtag_dict

def get_grid_ls(grid_data, grid_post_count, grid_hashtag_dict):
    """ Make the result list for the grids """
    grid_dict = {}
    for i in range(len(grid_data)):
        grid_id = grid_data[i]["properties"]["id"]
        # values: numpy array and counter
        grid_dict[grid_id] = {"post_num": grid_post_count[i], "hashtags": grid_hashtag_dict[i]}

    # sort dict by value["post_num"] in descending order
    grid_ls = sorted(grid_dict.items(), key=lambda x: x[1]["post_num"], reverse = True)

    return grid_ls

def handle_gathered_dict(grid_data, reduced_post_count, gathered_hashtag_dict_ls):
    """ Combine the gathered dictionary and convert into a list """
    # gathered as a list of dictionaries
    combined_hashtag_dict = gathered_hashtag_dict_ls[0]

    # put into a single dict
    for grid_hashtag_dict in gathered_hashtag_dict_ls[1:]:           # grid_hashtag_dict from each process
        for grid_id, hashtag_counter in grid_hashtag_dict.items():   # hashtag_dict for each grid
                combined_hashtag_dict[grid_id] += hashtag_counter

    return combined_hashtag_dict

def take_top5_hashtags(hashtag_ls):
    # no need to do slicing
    if len(hashtag_ls) <= 5:
        return hashtag_ls
    
    # find top 5 index
    top = 0
    prev = None
    for i in range(len(hashtag_ls)):
        if hashtag_ls[i][1] != prev:
            prev = hashtag_ls[i][1]
            top += 1
        if top > 5:
            i -= 1
            break
    
    return hashtag_ls[:i+1]

def print_tasks(grid_ls):
    """ Print both results of task 1 and task 2 """
    # TASK 1
    print("TASK - 1")
    print("----------")
    for grid_tuple in grid_ls:
        if grid_tuple[1]["post_num"] != 0:
            print("{}: {} posts".format(grid_tuple[0], grid_tuple[1]["post_num"]))

    print("\n")

    # TASK 2
    print("TASK - 2")
    print("----------")
    for grid_tuple in grid_ls:
        hashtag_counter = grid_tuple[1]["hashtags"]

        if len(hashtag_counter) != 0:
            # gather by master
            hashtag_ls = hashtag_counter.most_common()
            hashtag_ls = take_top5_hashtags(hashtag_ls)
            print("{}: {}".format(grid_tuple[0], tuple(hashtag_ls)))

def main():
    """ main function of this program """

    args = get_args()

    # MPI: mpi4py
    comm = MPI.COMM_WORLD         # every process
    comm_rank = comm.Get_rank()   # rank of this process
    comm_size = comm.Get_size()   # total num of processes

    # list of dicts
    grid_data = read_grid(args.grid_file)
    tweet_data = read_tweet(args.tweet_file, grid_data, comm_rank, comm_size)

    # get statistics
    grid_post_count, grid_hashtag_dict = stat_tweet(tweet_data, len(grid_data))

    # *************** REDUCE **************

    reduced_post_count = None
    if comm_rank == 0:
        reduced_post_count = np.zeros(len(grid_data), dtype=int)
    
    comm.Reduce(grid_post_count, reduced_post_count, op=MPI.SUM, root=0)

    # ************* GATHER ***************

    gathered_hashtag_dict_ls = comm.gather(grid_hashtag_dict, root=0)

    if comm_rank == 0:
        combined_hashtag_dict = handle_gathered_dict(grid_data, reduced_post_count, gathered_hashtag_dict_ls)
        grid_ls = get_grid_ls(grid_data, reduced_post_count, combined_hashtag_dict)
        # print tasks
        print_tasks(grid_ls)


if __name__ == "__main__":
    """ If running the file directly """
    main()