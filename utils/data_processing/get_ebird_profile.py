"""
Get Profiles
==========================================================
Users webbot to get log into ebird and get user profile urls when available. Profiles are found using sample checklist
from user and urls are written to an output .txt file.

Author: Bento Gonçalves
License: MIT
Copyright: 2020-2021
"""

import re
from argparse import ArgumentParser
from functools import partial
from multiprocessing import Pool

import geopandas as gpd
from bs4 import BeautifulSoup
from webbot import Browser


def parse_args():
    parser = ArgumentParser("Uses webbot to extract user profile urls from ebird")
    parser.add_argument('--input_users', 'i', type=str,
                        help='path to users dataframe with checklist IDs to search for profiles',
                        default='../Datasets/Shapefiles/users_US_2005-2019.csv')
    parser.add_argument('--output_txt', '-o', type=str,
                        help="path to .txt file where user profile urls will be written to",
                        default='../Datasets/profiles.txt')
    return parser.parse_args()


# helper to get username
def get_profile(checklist_url: str, out: str):
    """
    Gets user profile from ebird. Users webbot to login to ebird.org and searches for user profile address
    on sample checklist entry for user.

    :param checklist_url: path to a checklist url, extracted from users dataframe column
    :param out: path to output text file
    :return: None
    """
    web = Browser(showWindow=False)
    web.go_to("https://secure.birds.cornell.edu/cassso/login")
    web.type('birds_of_a_feather', into='Username')
    web.type('y&2m#9B3B2NGGzp', into='Password')
    web.press(web.Key.ENTER)
    web.go_to(checklist_url)
    source = web.get_page_source()
    soup = BeautifulSoup(source, 'lxml')
    for link in soup.findAll('a', attrs={'href': re.compile("/profile/")}):
        profile = f"https://ebird.org/{link.get('href')}"
        with open(out, 'a') as src:
            src.write(f"{checklist_url}_{profile}\n")


def main():
    args = parse_args()
    # read checklist data
    user_data = gpd.read_file(args.input_users)
    input_checklists = user_data['sample_che']

    # start multiprocessing pool
    pool = Pool(24)
    pool.map(partial(get_profile, out=args.output_txt), input_checklists)


if __name__ == "__main__":
    main()
