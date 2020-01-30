__all__ = ['load_ebird']

import pandas as pd
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser("eBird database .txt file muncher")
    parser.add_argument('--input_txt', '-i', type=str, help='path to eBird database file')
    parser.add_argument('--start_year', '-y', type=int, help='year to start data collection', default=2017)
    parser.add_argument('--output', '-o', type=str, help='path to output csv file', default='observation_data.csv')
    return parser.parse_args()


def load_ebird(filename: str, start_year: int = 2017, output: str = 'obseration_data.csv', rows: int = 10e5):
    """
    eBird database .txt file muncher. Reads observations from text file by chunks and writes combined data sorted by
    observer ID into a .csv file
    :param filename: input file
    :param start_year: starting year, filters data before starting year
    :param output: path to output .csv file
    :param rows: number of rows in a chunk
    :return: None
    """
    # columns of interest
    target_columns = ['SAMPLING EVENT IDENTIFIER', 'COMMON NAME', 'OBSERVATION COUNT',
                      'LOCALITY TYPE', 'LATITUDE', 'LONGITUDE', 'OBSERVATION DATE',
                      'TIME OBSERVATIONS STARTED', 'PROTOCOL TYPE', 'DURATION MINUTES',
                      'EFFORT DISTANCE KM', 'EFFORT AREA HA', 'NUMBER OBSERVERS',
                      'ALL SPECIES REPORTED', 'HAS MEDIA', 'OBSERVER ID']

    # accumulate chunks
    chunks = []

    # load the big file in smaller chunks
    for df_chunk in pd.read_csv(filename,
                                sep='\t',
                                chunksize=rows,
                                usecols=target_columns):

        valid = [int(ele.split('-')[0]) >= start_year for ele in df_chunk['OBSERVATION DATE']]
        df_chunk = df_chunk.loc[valid]

        # don`t append invalid rows
        if len(df_chunk) > 0:
            chunks.append(df_chunk)

    # write output to csv
    combined = pd.concat(chunks).sort_values(by=['OBSERVER ID'])
    combined.to_csv(output)


def main():
    args = parse_args()
    load_ebird(filename=args.input_txt, start_year=args.start_year, output=args.output)


if __name__ == "__main__":
    main()
