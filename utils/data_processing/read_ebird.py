"""
Read eBird
==========================================================
Reads raw eBird text files and saves columns of interest. Reads .txt in chunks and appends them to an existing
observations .csv file.

Author: Bento Gonçalves
License: MIT
Copyright: 2020-2021
"""

__all__ = ['load_ebird']

import time
from argparse import ArgumentParser

import numpy as np
import pandas as pd


def parse_args():
    parser = ArgumentParser("eBird database .txt file muncher")
    parser.add_argument('--input_txt', '-i', type=str, help='path to eBird database file')
    parser.add_argument('--period', '-p', type=str, help='start year to end year separated by a dash',
                        default='2017-2020')
    parser.add_argument('--output', '-o', type=str, help='path to output csv file', default='observation_data.csv')
    return parser.parse_args()


def load_ebird(filename: str, period: list, cores: int = 1, output: str = 'observation_data.csv', rows: int = 10e5):
    """
    eBird database '.txt' file muncher. Reads observations from text file by chunks and writes data to a '.csv' file
    sorted by observer ID and date.
    :param filename: input file
    :param period: starting year, filters data before starting year
    :param cores
    :param output: path to output .csv file
    :param rows: number of rows in a chunk
    :return: None
    """
    # columns of interest
    target_columns = ['SAMPLING EVENT IDENTIFIER', 'COMMON NAME', 'OBSERVATION COUNT',
                      'LOCALITY TYPE', 'LATITUDE', 'LONGITUDE', 'OBSERVATION DATE',
                      'TIME OBSERVATIONS STARTED', 'PROTOCOL TYPE', 'DURATION MINUTES',
                      'EFFORT DISTANCE KM', 'NUMBER OBSERVERS', 'ALL SPECIES REPORTED',
                      'HAS MEDIA', 'OBSERVER ID', 'GROUP IDENTIFIER']

    # load the big file in smaller chunks
    start = time.time()
    first = True

    for df_chunk in pd.read_csv(filename,
                                sep='\t',
                                chunksize=rows,
                                usecols=target_columns,
                                dtype={'HAS MEDIA': np.uint8,
                                       'ALL SPECIES REPORTED': np.uint8,
                                       'DURATION MINUTES': np.float32,
                                       'EFFORT DISTANCE KM': np.float32,
                                       'NUMBER OBSERVERS': np.float32}):

        valid = [int(period[0]) <= int(ele.split('-')[0]) < int(period[1]) for ele in df_chunk['OBSERVATION DATE']]
        df_chunk = df_chunk.loc[valid]

        # don`t append invalid rows
        if len(df_chunk) > 0:
            if first:
                df_chunk.to_csv(output)
                first = False

            else:
                df_chunk.to_csv(output, mode='a', header=False)

    print(
        f"Finished reading raw data and writing to .csv in {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}")


def main():
    args = parse_args()
    load_ebird(filename=args.input_txt, period=args.period.split('-'), output=args.output)


if __name__ == "__main__":
    main()
