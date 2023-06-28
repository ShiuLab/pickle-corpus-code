"""
Script to detect and correct incorrect sentence splits in a jsonl dataset.

This script relies on the assumption that a cross-sentence relation in a jsonl
dataset is a result of an incorrect sentence split, rather than intentional. If
a cross-sentence relation is found, all sentences between the sentences
containing the two joined entities will be combined into one sentence.

Example: BioInfer.d70 is one sentence only, with two relations. However, the
conversion to jsonl results in the following doc dictionary:

{"doc_key": "BioInfer.d70",
"dataset": "bioinfer",
"sentences": [["Aprotinin", "inhibited", "platelet", "aggregation", "induced",
        "by", "thrombin", "(", "0.25", "U.ml-1", ")", "with", "IC50", "200",
        "kIU.ml-1", ",", "and", "inhibited", "the", "rise", "of", "cytosolic",
        "free", "calcium", "concentration", "in", "platelets", "stimulated", "by",
        "thrombin", "(", "0.1", "U.ml-1", ")", "in", "the", "absence", "and", "in",
        "the", "presence", "of", "Ca2", "+", "0.5", "mmol", "."],
    ["L-1", "(","IC50", "117", "and", "50", "kIU.ml-1", ",", "respectively",
        ")", ",", "but", "had", "no", "effect", "on", "the", "amounts", "of",
        "actin", "and", "myosin", "heavy", "chain", "associated", "with",
        "cytoskeletons", "."]],
"ner": [[[29, 29, "Individual_protein"], [0, 0, "Individual_protein"],
    [6, 6, "Individual_protein"]],
    [[68, 70, "Individual_protein"], [66, 66, "Individual_protein"]]],
"relations": [[[29, 29, 0, 0, "PPI"], [0, 0, 66, 66, "PPI"]],
    [[68, 70, 0, 0, "PPI"]]]}

Originally written for serenalotreck/obie-wan
Author: Serena G. Lotreck
"""
import argparse
from os.path import abspath, splitext
import jsonlines
from tqdm import tqdm


def check_correct_doc(doc):
    """
    Detect incorrectly split sentences and fix.

    parameters:
        doc, dict: dygiepp-formatted doc to fix

    returns:
        new_doc, dict: corrected doc
    """
    # Get the sentence start and end indices
    sent_idxs = []
    for i, sent in enumerate(doc['sentences']):
        if i == 0:
            sent_start = 0
        else:
            sent_start = sent_idxs[i-1][1] + 1
        sent_end = sent_start + len(sent)  - 1
        sent_idxs.append((sent_start, sent_end))

    # For each relation, check if it crosses sentence boundaries
    sents_to_join = []
    for i, relset in enumerate(doc['relations']):
        for rel in relset:
            e1_start = rel[0]
            e2_start = rel[2]
            sent_mems = []
            for i, sent in enumerate(sent_idxs):
                if sent[0] <= e1_start <= sent[1]:
                    sent_mems.append(i)
                if sent[0] <= e2_start <= sent[1]:
                    sent_mems.append(i)
            if sent_mems[0] != sent_mems[1]:
                sents_to_join.append(sent_mems)
    sents_to_join = list(set([tuple(pair) for pair in sents_to_join]))

    # Join sentences that need it
    if len(sents_to_join) == 0:
        new_doc = doc
        return new_doc
    else:
        new_doc = {'doc_key': doc['doc_key'], 'dataset': doc['dataset']}
        for key in ['sentences', 'ner', 'relations']:
            joined = []
            # Not accounting for there being multiples to join
            assert len(sents_to_join) == 1, ('There is more than one pair of '
                                            'sentences to join: '
                                            f'sents: {doc["sentences"]} '
                                            f'\nsents_to_join: {sents_to_join}')
            for pair in sents_to_join:
                # Not accounting for a sentence having been split in more than two
                assert abs(pair[1] - pair[0]) == 1, ('Sentences are not adjacent: '
                                                    f'sents: {doc["sentences"]} '
                                                    f'\nsent idxs to join: {pair}')
                # Add all sentences before the first to join
                first_idx = min(pair)
                joined.extend(doc[key][:first_idx])

                # Join the pair and add
                last_idx = max(pair)
                pair_j = [doc[key][first_idx] + doc[key][last_idx]]
                joined.extend(pair_j)

                # Add any after
                if last_idx != len(doc[key]) - 1:
                    joined.extend(doc[key][last_idx + 1:])

                # Add to new doc
                new_doc[key] = joined

        return new_doc


def main(dataset):

    # Read in dataset
    print('\nReading in dataset...')
    with jsonlines.open(dataset) as reader:
        dset = []
        for obj in reader:
            dset.append(obj)

    # Go through and check for cross sentence rels and correct
    print('\nDetecting and correcting errors...')
    corrected_dset = []
    num_corrected = 0
    for doc in tqdm(dset):
        corrected_doc = check_correct_doc(doc)
        if corrected_doc != doc:
            num_corrected += 1
        corrected_dset.append(corrected_doc)
    print(f'A total of {num_corrected} documents were corrected.')

    # Save out
    print('\nSaving...')
    path_and_name, ext = splitext(dataset)
    new_save_name = f'{path_and_name}_CORRECTED{ext}'
    with jsonlines.open(new_save_name, 'w') as writer:
        writer.write_all(corrected_dset)
    print(f'Dataset saved as {new_save_name}')

    print('\nDone!\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Join mis-split sentences')

    parser.add_argument('dataset', type=str,
        help='Path to dataset to check. Output will be saved back to the same '
        'directory, with the string CORRECTED appended to the filename')

    args = parser.parse_args()

    args.dataset = abspath(args.dataset)

    main(args.dataset)
