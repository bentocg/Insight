from argparse import ArgumentParser

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


def parse_args():
    parser = ArgumentParser("copies a .csv dataframe with latitude and longitude columns into a GIS shapefile")
    parser.add_argument('--input_csv', '-i', type=str, default="Datasets/users_NA_2005-2019.csv",
                        help="Path to .csv file.")
    parser.add_argument("--output_shp", "-o", type=str, default="Datasets/Shapefiles/users_NA_2005-2019.shp",
                        help="path to output shapefile")
    return parser.parse_args()


def main():
    # read arguments
    args = parse_args()
    input_path = args.input_csv
    output_path = args.output_shp

    # read input
    users_df = pd.read_csv(input_path)

    # get latitude and longitude from str geometry
    lat = [float(ele.split(' ')[1][1:]) for ele in users_df.geometry]
    lon = [float(ele.split(' ')[2][:-1]) for ele in users_df.geometry]
    geometry = [Point(lon[idx], lat[idx]) for idx in range(len(lon))]

    # add to GeoDataFrame
    users_df['latitude'] = lat
    users_df['longitude'] = lon
    users_df['geometry'] = geometry
    users_df = gpd.GeoDataFrame(users_df)

    # save output
    users_df.to_file(output_path)


if __name__ == "__main__":
    main()
