__all__ = ['get_user_data']

import datetime
import time
from argparse import ArgumentParser
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
from shapely.geometry import Point


def parse_args():
    parser = ArgumentParser("Script to process eBird observations into user data")
    parser.add_argument('--input_csv', '-i', type=str, help='path to observations .csv file')
    parser.add_argument('--cores', '-c', type=int, help='number of cores for parallel processing', default=1)
    parser.add_argument('--output', '-o', type=str, help='path to output csv file', default='observation_data.csv')
    return parser.parse_args()


# helper function to handle missing data on numeric columns
def handle_numerical(nums):
    if type(nums[0]) == str:
        return [float(ele) if len(ele) > 0 else float('NaN') for ele in nums]
    else:
        return [float(ele) for ele in nums]


# processing function
def get_user_data(user_data, bird_stats, latest=2015):
    """
    Helper function to process data from observations of a single user
    :param user_data:
    :param client:
    :param latest:
    :return:
    """
    # store user name
    # try:
    #     user = client.get_checklist(user_data['SAMPLING EVENT IDENTIFIER'].iloc[0])['userDisplayName']
    # except IndexError:
    #     print('failed to load username')
    #     return False

    # event data
    event_data = pd.DataFrame()

    for event in set(user_data['SAMPLING EVENT IDENTIFIER']):
        event_data = event_data.append(user_data.loc[user_data['SAMPLING EVENT IDENTIFIER'] == event].iloc[0],
                                       ignore_index=True)

    # add event statistics
    n_checklists = len(event_data)
    n_observations = len(user_data)

    # time variables

    # find median interval between observations
    dates = [datetime.date(*[int(ele) for ele in date.split('-')]) for date in event_data['OBSERVATION DATE']]

    # get year of first checklist
    since = int(dates[0].year)

    # check if user has been active for the last year
    if int(dates[-1].year) < latest:
        return False

    if n_checklists > 1:
        dates = sorted(dates)
        intervals = [(dates[idx + 1] - dates[idx]).total_seconds() for idx in range(len(dates) - 1)]
        median_interval = np.median(intervals)
    else:
        median_interval = 0

    # effort variables

    # deal with missing data on numerical columns

    distances = handle_numerical([ele for ele in event_data['EFFORT DISTANCE KM']])
    median_distance = np.nanmedian(distances)

    durations = handle_numerical([ele for ele in event_data['DURATION MINUTES']])
    median_duration = np.nanmedian(durations)

    percents = handle_numerical([ele for ele in event_data['ALL SPECIES REPORTED']])
    percent_all = np.mean(percents)

    observers = handle_numerical([ele for ele in event_data['NUMBER OBSERVERS']])
    mean_observers = np.mean(observers)

    # location
    percent_hotspot = sum(event_data['LOCALITY TYPE'] == 'H') / n_checklists
    locations = [Point([float(ele[0]), float(ele[1])]) for ele in zip(event_data['LATITUDE'], event_data['LONGITUDE'])]
    distances = [sum([ele1.distance(ele2) for ele1 in locations]) for ele2 in locations]
    centroid = locations[np.argmin(distances)]

    # trip type
    percent_travel = sum(event_data['PROTOCOL TYPE'] == 'Traveling') / n_checklists
    if percent_travel > 0:
        travels = [locations[idx] for idx, ele in enumerate(event_data['PROTOCOL TYPE']) if ele == 'Traveling']
        median_travel_distance = np.nanmedian([centroid.distance(ele) for ele in travels])
    else:
        median_travel_distance = 0

    # median starting time
    times = [int(tm[:2]) * 3600 + int(tm[3:5]) * 60 + int(tm[6:]) if type(tm) == str else float('NaN') for tm in
             event_data['TIME OBSERVATIONS STARTED']]
    median_start = np.nanmedian(times)

    # observation data
    percent_media = np.mean([int(ele) for ele in user_data['HAS MEDIA']])
    n_species = len(set(user_data['COMMON NAME']))

    # species data
    valid_obs = [spc for spc in user_data['COMMON NAME'] if spc in bird_stats.index]
    species_size = np.mean([bird_stats.loc[spc, 'log_10_mass'] for spc in valid_obs])
    species_color = np.mean([bird_stats.loc[spc, 'max_color_contrast'] for spc in valid_obs])
    species_resident = np.mean([bird_stats.loc[spc, 'resident'] for spc in valid_obs])
    species_introduced = np.mean([bird_stats.loc[spc, 'introduced'] for spc in valid_obs])
    species_common = np.mean([bird_stats.loc[spc, 'commonness'] for spc in valid_obs])

    # get a sample checklist
    if percent_media > 0:
        checklist = np.random.choice(user_data.loc[user_data['HAS MEDIA'] == 1]['SAMPLING EVENT IDENTIFIER'])
    else:
        checklist = np.random.choice(user_data['SAMPLING EVENT IDENTIFIER'])

    sample_checklist = "https://ebird.org/checklist/" + checklist

    # add user

    user_df = pd.DataFrame()
    user_df = user_df.append(pd.Series({'n_checklists': n_checklists,
                                        'n_species': n_species,
                                        'n_observations': n_observations,
                                        'since': since,
                                        'geometry': centroid,
                                        'species_size': species_size,
                                        'species_color': species_color,
                                        'species_resident': species_resident,
                                        'species_introduced': species_introduced,
                                        'species_common': species_common,
                                        'median_distance': median_distance,
                                        'median_duration': median_duration,
                                        'median_interval': median_interval,
                                        'median_travel_distance': median_travel_distance,
                                        'median_start': median_start,
                                        'percent_all': percent_all,
                                        'percent_media': percent_media,
                                        'percent_travel': percent_travel,
                                        'percent_hotspot': percent_hotspot,
                                        'sample_checklist': sample_checklist,
                                        'mean_group_size': mean_observers}, name=user_data['OBSERVER ID'].iloc[0]))

    # return user data
    return user_df


def main():
    # read arguments
    args = parse_args()

    # # setup client instance
    # api_key = '6qmfvb8pg9dk'
    # locale = 'en_US'
    # client = Client(api_key, locale)

    start = time.time()
    # read observations database
    chunks = []
    for df_chunk in pd.read_csv(args.input_csv,
                                chunksize=10E5):
        chunks.append(df_chunk)

        # write output to csv
    observations = pd.concat(chunks).sort_values(by=['OBSERVER ID'])[:50000]

    # user breakpoints
    breaks = []
    observers = list(observations['OBSERVER ID'])
    prev = 0
    for idx in range(len(observers) - 1):
        if observers[idx] != observers[idx + 1]:
            if idx - prev >= 40:
                breaks.append([prev, idx + 1])
            prev = idx + 1

    # create chunks
    chunks = (observations.iloc[range(chunk_idx[0], chunk_idx[1])] for chunk_idx in breaks[:-1])

    # read bird statistics dataframe
    bird_stats = pd.read_csv('Datasets/bird_stats.csv', index_col=0)

    # process observations in parallel
    pool = Pool(args.cores)

    # out = []
    #
    # for chunk in chunks:
    #     out.append(pool.apply_async(partial(get_user_data, client=client, bird_stats=bird_stats), (chunk,)).get())
    # pool.close()
    # pool.join()
    # results = [pool.apply_async(partial(get_user_data, bird_stats=bird_stats), (chunk,)) for chunk in
    #            chunks]
    # out = [p.get() for p in results]
    out = pool.map_async(partial(get_user_data, bird_stats=bird_stats), chunks)

    user_df = pd.concat(out.get())
    user_df.to_csv(args.output)
    print(
         f"Finished compiling {len(observations)} observations into {len(user_df)} users in  "
         f"{time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}")


if __name__ == "__main__":
    main()
