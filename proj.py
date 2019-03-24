import json
import argparse
import numpy as np
from mpi4py import MPI
import time

# CONSTANTS
TOTAL_GRID_NUM = 16

# Obtaining args from terminal
def get_args():
    
    parser = argparse.ArgumentParser(description="Processing tweets")
    
    # filenames
    parser.add_argument("-t", "--tweet_file", type = str, required = True, help = "Twitter data file")
    parser.add_argument("-g", "--grid_file", type = str, required = True, help = "Grid data file")

    args = parser.parse_args()
    
    return args

def read_grid(grid_file):
    with open(grid_file, "r") as f:
        grid_data = json.load(f)["features"]
        f.close()
    return grid_data

def fix_line(line):
    # need to end with "}"
    i = -1
    while line[i] != "}":
        i -= 1
    return line[:i+1]

def read_tweet(tweet_file, rank, size):

    with open(tweet_file, "r", encoding="utf-8") as f:

        # remove the header line
        tweet_data = []
        for i, line in enumerate(f):

            # first / last line
            if i == 0 or (i % (size) != rank):  # assign line according to the rank
                continue
            elif line[0] != "{":
                continue
                
            old_dict = json.loads(fix_line(line))["doc"]

            # only using the coordinates and hashtags info of tweets
            new_dict = {"coordinates": old_dict["coordinates"], 
                        "hashtags": old_dict["entities"]["hashtags"]}
            tweet_data.append(new_dict)
        f.close()
    return tweet_data

def get_tweet_grid(coordinate, grid_data):

    # representing each grid by a number
    grid_num = 0

    for grid in grid_data:
        # 4 points of each grid
        xmin = grid["properties"]["xmin"]
        xmax = grid["properties"]["xmax"]
        ymin = grid["properties"]["ymin"]
        ymax = grid["properties"]["ymax"]

        x = coordinate[0]
        y = coordinate[1]

        # not need to deal with the boundary problem because of the order of the grids
        if x >= xmin and x <= xmax and y >= ymin and y <= ymax:
            return grid_num
        
        grid_num += 1

    # not in any grid
    return -1

def stat_tweet(tweet_data, grid_data):

    grid_post_count = np.zeros(TOTAL_GRID_NUM, dtype=int)
    grid_hashtag_dict = {}

    # initialise empty hashtag dict for all grids
    for i in range(TOTAL_GRID_NUM):
        grid_hashtag_dict[i] = {}

    # iterate through each tweet
    for tweet in tweet_data:

        # **************** TASK 1 ****************

        # skip this tweet if no coordinates
        if tweet["coordinates"] == None:
            continue   

        grid_id = get_tweet_grid(tweet["coordinates"]["coordinates"], grid_data)
        # skip this tweet if not in the grids
        if grid_id == -1:
            continue

        # count post
        grid_post_count[grid_id] += 1

        # **************** TASK 2 ****************

        # add hashtags if available
        for hashtag in tweet["hashtags"]:
            text = hashtag["text"].lower()     # not case sensitive
            grid_hashtag_dict[grid_id][text] = grid_hashtag_dict[grid_id].get(text, 0) + 1   

    return grid_post_count, grid_hashtag_dict

def get_grid_ls(grid_data, grid_post_count, grid_hashtag_dict):
    grid_dict = {}
    for i in range(TOTAL_GRID_NUM):
        grid_id = grid_data[i]["properties"]["id"]
        grid_dict[grid_id] = {"post_num": grid_post_count[i], "hashtags": grid_hashtag_dict[i]}

    grid_ls = sorted(grid_dict.items(), key=lambda x: x[1]["post_num"], reverse = True)

    return grid_ls

def print_tasks(grid_ls):
    # TASK 1
    print("TASK - 1")
    for grid_tuple in grid_ls:
        print("{}: {} posts".format(grid_tuple[0], grid_tuple[1]["post_num"]))

    print("-------------")

    # TASK 2
    print("TASK - 2")
    for grid_tuple in grid_ls:
        hashtag_dict = grid_tuple[1]["hashtags"]

        # gather by master

        hashtag_ls = sorted(hashtag_dict.items(), key=lambda x: x[1], reverse = True)[:5]  # take top 5
        print("{}: {}".format(grid_tuple[0], hashtag_ls))
    

def main():

    # time
    start_time = time.time()

    args = get_args()

    # MPI: mpi4py
    comm = MPI.COMM_WORLD         # every process
    comm_rank = comm.Get_rank()   # rank of this process
    comm_size = comm.Get_size()   # total num of processes

    # list of dicts
    tweet_data = read_tweet(args.tweet_file, comm_rank, comm_size)
    grid_data = read_grid(args.grid_file)

    # get statistics
    grid_post_count, grid_hashtag_dict = stat_tweet(tweet_data, grid_data)

    # *************** REDUCE **************

    reduced_post_count = None
    if comm_rank == 0:
        reduced_post_count = np.zeros(TOTAL_GRID_NUM, dtype=int)
    
    comm.Reduce(grid_post_count, reduced_post_count, op=MPI.SUM, root=0)

    # ************* GATHER ***************

    gathered_hashtag_dict = comm.gather(grid_hashtag_dict, root=0)

    if comm_rank == 0:

        # gathered as a list of dictionaries
        final_hashtag_dict = gathered_hashtag_dict[0]

        for agrid_hashtag_dict in gathered_hashtag_dict[1:]:   # agrid_hashtag_dict from each process
            for grid_id, hashtag_dict in agrid_hashtag_dict.items():   # hashtag_dict for each grid
                for hashtag, count in hashtag_dict.items():
                    final_hashtag_dict[grid_id][hashtag] = final_hashtag_dict[grid_id].get(hashtag, 0) + count 

        # put into a dict
        grid_ls = get_grid_ls(grid_data, reduced_post_count, final_hashtag_dict)

        # print tasks
        print_tasks(grid_ls)

        elapse_time = time.time() - start_time
        print("elapse_time: {0:.2f} ms".format(elapse_time*1000))

    # elapse_time = int(time.time() - start_time)
    # print("elapse_time: " + 
    #       "{:02d}:{:02d}:{:02d}".format(
    #           elapse_time // 3600, 
    #           elapse_time % 3600 // 60, 
    #           elapse_time % 60))

# If running the file directly
if __name__ == "__main__":
    main()