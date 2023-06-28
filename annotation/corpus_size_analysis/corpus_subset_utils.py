"""
Utils for generating serial subset of a dataset.

Author: Serena G. Lotreck
"""
import jsonlines
from random import shuffle


def subset_corpus(corpus_path, start_size, subset_size, num_subs, dev_size,
                    test_size, dataset_name, out_loc):
    """
    Subset a dygiepp-formatted corpus.

    parameters:
        corpus_path, str: path to dygiepp-formatted corpus
        start_size, int: unmber of docs to put in smallest training set
        subset_size, int: number of new docs to add each time
        num_subs, int: number of times to add new docs
        dev_size, int: number of docs for dev set
        test_size, int: number of docs for test set
        dataset_name, str: name of dataset to prepend to output files
        out_loc, str: path to directory to save the doc sets

    returns:
        trainsets, list of str: names of training set files
        dev_name, str: dev set filename
        test_name, str: test set filename
    """
    # Read in the main corpus
    with jsonlines.open(corpus_path) as reader:
        full_corpus = []
        for obj in reader:
            full_corpus.append(obj)

    # Common-sense check the requested numbers
    assert test_size + dev_size + start_size < len(full_corpus), ('Requested '
                'test, dev and start train sets are larger than the available '
                'corpus resources')

    # Establish test and dev sets
    shuffle(full_corpus)
    test = full_corpus[-test_size:]
    del full_corpus[-test_size:]
    test_name = f'{out_loc}/{dataset_name}_test_{test_size}.jsonl'
    with jsonlines.open(test_name, 'w') as writer:
        writer.write_all(test)
    dev = full_corpus[-dev_size:]
    del full_corpus[-dev_size:]
    dev_name = f'{out_loc}/{dataset_name}_dev_{dev_size}.jsonl'
    with jsonlines.open(dev_name, 'w') as writer:
        writer.write_all(dev)

    # Establish base training set
    trainsets = []
    train = full_corpus[-start_size:]
    del full_corpus[-start_size:]
    base_train_name = f'{out_loc}/{dataset_name}_train_{start_size}.jsonl'
    trainsets.append(base_train_name)
    with jsonlines.open(base_train_name, 'w') as writer:
        writer.write_all(train)

    # Then do repeated increments for train set
    for i in range(num_subs):
        assert len(full_corpus) > subset_size, ('There are no more documents '
                'left in the corpus!')
        add_train = full_corpus[-subset_size:]
        del full_corpus[-subset_size:]
        train += add_train
        next_name = f'{out_loc}/{dataset_name}_train_{len(train)}.jsonl'
        trainsets.append(next_name)
        with jsonlines.open(next_name, 'w') as writer:
            writer.write_all(train)

    return trainsets, dev_name, test_name
