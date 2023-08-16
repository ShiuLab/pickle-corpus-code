"""
Script to compare two datasets. Outputs a summary file and plots where
appropriate.

Author: Serena G. Lotreck
"""
import argparse
from os.path import abspath

import jsonlines
import json
from dataset import Dataset

def not_in_pickle(pickle, dset2):
    """
    Define the fraction of dset2 that is not in the PICKLE vocab.
    
    parameters:
        pickle, Dataset obj: PICKLE
        dset2, Dataset obj: comparison dataset

    returns:
        fracs, dict of float: fraction of PICKLE that is OOV for dset2 for each
            of unigrams, bigrams and trigrams
        oov_grams, dict of list of str: OOV grams for each of unigrams, bigrams,
            and trigrams
    """
    # Get the vocabularies
    pickle_vocab = pickle.get_dataset_vocab()
    dset2_vocab = dset2.get_dataset_vocab()
    
    # Compare
    fracs = {}
    oov_grams = {}
    for key in pickle_vocab.keys():
        oov_dset1 = pickle_vocab[key] - dset2_vocab[key]
        fracs[key] = len(oov_dset1)/len(pickle_vocab[key])
        oov_grams[key] = list(oov_dset1)
    
    return fracs, oov_grams


def read_dset(path, dset_name):
    """
    Read in a dataset as a Dataset object from a file path..

    parameters:
        path, str: absolute path to the dataset jsonl file
        dset_name, str: name of the dataset
    returns:
        dset, Dataset object: dataset object for this dataset
    """
    dset_list = []
    with jsonlines.open(path) as reader:
        for obj in reader:
            dset_list.append(obj)
    dset = Dataset(dset_name, dset_list)

    return dset


def main(pickle_path, dset2_path, dset2_name, 
        out_loc, out_prefix):

    # Read in the datasets
    verboseprint('\nReading in the datasets...')
    pickle = read_dset(pickle_path, 'PICKLE')
    dset2 = read_dset(dset2_path, dset2_name)

    # Look for out-of-vocabulary words
    verboseprint('\nComparing out-of-vocabulary words...')
    fracs, oov_grams = not_in_pickle(pickle, dset2)
    frac_save_name = f'{out_loc}/{out_prefix}_fracs.json'
    oov_save_name = f'{out_loc}/{out_prefix}_oovs.json'
    with open(frac_save_name, mode='w') as myf:
       json.dump(fracs, myf)
    with open(oov_save_name, mode='w') as myf:
       json.dump(oov_grams, myf)
    verboseprint(f'Saved out-of-vocabulary comparison as {oov_save_name} and '
                f'{frac_save_name}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Compare two annotated datasets')

    parser.add_argument('pickle_path', type=str,
            help='Path to jsonl file containing PICKLE to compare')
    parser.add_argument('dset2', type=str,
            help='Path to jsonl file containing the dataset to compare')
    parser.add_argument('dset2_name', type=str,
            help='String identifying the dataset to compare')
    parser.add_argument('out_loc', type=str,
            help='Path to save output')
    parser.add_argument('out_prefix', type=str,
            help='String to prepend to all output file names')
    parser.add_argument('-v', '--verbose', action='store_true',
             help='Whether or not to print updates to stdout')

    args = parser.parse_args()

    args.pickle_path = abspath(args.pickle_path)
    args.dset2 = abspath(args.dset2)
    args.out_loc = abspath(args.out_loc)

    verboseprint = print if args.verbose else lambda *a, **k: None
    main(args.pickle_path, args.dset2, args.dset2_name, args.out_loc,
        args.out_prefix)
