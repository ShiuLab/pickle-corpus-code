"""
Using the Results By Year output from PubMed, calculate the percent-per-year
increase in publications for a given search term.

Used to calculate the value used in the introduction of the manuscript.

Author: Serena G. Lotreck
"""
import argparse
from os.path import abspath
import pandas as pd


def main(results_py, start_year, out_loc):

    # Read in file
    df = pd.read_csv(results_py, skiprows=[0], header=0) # First row is search term

    # Sort values to put dataframe in ascending order
    df = df.sort_values(['Year'], ascending=True)

    # Drop years before start year
    if start_year != 0:
        df = df[df['Year'] >= start_year]

    # Drop this year
    df = df[df['Year'] != 2023]

    # Calculate percent increase per year
    df['pct_chng'] = df['Count'].pct_change()

    # Get the mean increase per year and print
    first_year = df['Year'].min()
    last_year = df['Year'].max()
    mean_inc = df['pct_chng'].mean()
    print(f'\nThe mean increase in publications from {first_year} to '
            f'{last_year} is {mean_inc}')

    # Save out the dataframe
    df.to_csv(out_loc)
    print(f'\nDataframe saved as {out_loc}')

    print('\nDone!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                description='Get percent per year increase')

    parser.add_argument('results_py', type=str,
            help='Path to results per year file')
    parser.add_argument('-start_year', type=int, default=0,
            help='Year to start calculation. Default is to use all years')
    parser.add_argument('-out_loc', type=str,
            help='Path to save output, including file name')

    args = parser.parse_args()

    args.results_py = abspath(args.results_py)
    args.out_loc = abspath(args.out_loc)

    main(args.results_py, args.start_year, args.out_loc)
