import time
from argparse import ArgumentParser
from collections import Counter
from multiprocessing import Pool

import pandas as pd


def parse_args():
    parser = ArgumentParser("Script to get all pairs of users within observations")
    parser.add_argument('--input_obs', '-i', type=str, help='path to observations .csv file')
    parser.add_argument('--input_users', '-u', type=str, help='path to users .csv file')
    parser.add_argument('--cores', '-c', type=int, help='number of cores for parallel processing', default=1)
    parser.add_argument('--output', '-o', type=str, help='path to output csv file', default='observation_data.csv')
    return parser.parse_args()


def find_matches(group_obs):
    """
    Helper function to find pairs of users for validation
    :param group_obs: pandas DataFrame with all observations within one group checklist

    :return: string with two users IDs separated by '_' or the Boolean False if there are less than 2 users
    """

    users = sorted(set(group_obs['OBSERVER ID']))
    if len(users) != 2:
        return False
    else:
        return '_'.join(users)


def main():
    # read arguments
    args = parse_args()

    # read observation data and get list of target users and # checklists
    start = time.time()
    obs_data = pd.read_csv(args.input_obs, usecols=['OBSERVER ID', 'GROUP IDENTIFIER'])
    user_data = pd.read_csv(args.input_users, usecols=['Unnamed: 0', 'n_checklists'], index_col=0)
    users = {idx: row['n_checklists'] for idx, row in user_data.iterrows()}
    print(
        f"Read {len(obs_data)} observations in  "
        f"{time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}")
    start = time.time()

    # filter observation data by user and group size
    obs_data = obs_data.loc[obs_data['OBSERVER ID'].isin(users)]
    obs_data = obs_data.loc[[type(ele) != float for ele in obs_data['GROUP IDENTIFIER']]]

    # sort by group and find breakpoints between groups
    obs_data = obs_data.sort_values(by='GROUP IDENTIFIER')
    group_ids = list(obs_data['GROUP IDENTIFIER'])
    breaks = [0] + [idx + 1 for idx in range(len(group_ids) - 1) if group_ids[idx] != group_ids[idx + 1]]
    print(
        f"Filtered {len(obs_data)} observations into {len(breaks)} groups in "
        f"{time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}")
    start = time.time()

    # process groups
    chunks = (obs_data.iloc[range(breaks[idx], breaks[idx + 1])] for idx in range(len(breaks) - 1))

    # process observations in parallel
    pool = Pool(args.cores)

    # run processing pool
    out = pool.imap_unordered(find_matches, chunks)
    pool.close()
    pool.join()

    # filter invalid and count matches
    matches = [ele for ele in out if type(ele) == str]
    matches = Counter(matches)

    # reformat matches and write output to file
    matches_df = pd.DataFrame()
    for match in matches:
        # get users
        user1, user2 = match.split('_')

        # get average number of checklists
        avg_n_checklist = (users[user1] + users[user2]) / 2

        # add match
        matches_df = matches_df.append(
            pd.Series({'user_1': user1, 'user_2': user2, 'count': matches[match],
                       'count_percent': matches[match] / avg_n_checklist}, name=match))
    matches_df.to_csv(args.output)
    print(
        f"Wrote {len(matches_df)} valid pairs to {args.output} in "
        f"{time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}")


if __name__ == "__main__":
    main()
