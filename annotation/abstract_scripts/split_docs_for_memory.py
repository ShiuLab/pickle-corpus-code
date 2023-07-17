"""
Having documents longet than a normal abstract length can cause OOM errors with
DyGIE++. The developer's reccomended solution
(https://github.com/dwadden/dygiepp/issues/52) is to split the documents up to
make them smaller. This script splits any documents with more than 20 sentences
into two documents, roughly in half.

Author: Serena G. Lotreck
"""
import argparse
from os.path import abspath, splitext
import jsonlines


def adjust_indices(first_half, second_half):
    """
    Subtract the number of tokens from the first half of the document from all
    of the indices in the various fields of the second document.

    parameters:
        first_half, dict: first half of document
        second_half, dict: second half of document with original indices

    returns:
        second_half, dict: second half doc with updated indices
    """
    # Get the number that we have to subtract from the second half
    tokens = [tok for sent in first_half["sentences"] for tok in sent]
    to_subtract = len(tokens)

    # Then subtract
    for key in ["ner", "relations"]:
        idxs = 4 if key == "relations" else 2
        updated_value = []
        for sent in second_half[key]:
            updated_sent = []
            for elt in sent:
                updated_elt = []
                for idx in range(idxs):
                    updated_elt.append(elt[idx] - to_subtract)
                updated_elt += elt[idxs:]
                updated_sent.append(updated_elt)
            updated_value.append(updated_sent)
        second_half[key] = updated_value

    return second_half
        

def split_docs(dset):
    """
    Split documents with more than 20 sentences.

    parameters:
        dset, list of dict: dataset to process

    returns:
        updated_dset, list of dict: dset with large documents split
    """
    # Go through docs and split if necessary
    docs_split = []
    updated_dset = []
    for doc in dset:
        num_sents = len(doc["sentences"])
        if num_sents < 20:
            updated_dset.append(doc)
        else:
            # See how many sentences should go in each half
            num_half, remainder = divmod(num_sents, 2)
            first_doc_len = num_half
            second_doc_len = num_half + remainder
            print(f'Split doc lengths: {first_doc_len}, {second_doc_len}')
            # Generate new doc keys
            doc_key_first_half = doc["doc_key"] + '_split_1'
            doc_key_second_half = doc["doc_key"] + '_split_2'
            # Make new docs
            first_half = {"doc_key": doc_key_first_half, "dataset": doc["dataset"]}
            second_half = {"doc_key": doc_key_second_half, "dataset": doc["dataset"]}
            for key in ["sentences", "ner", "relations"]:
                first_half[key] = doc[key][:first_doc_len]
                second_half[key] = doc[key][first_doc_len:]
            second_half = adjust_indices(first_half, second_half)
            # Add to dataset
            updated_dset.append(first_half)
            updated_dset.append(second_half)
            docs_split.append(doc["doc_key"])

    print(f'A total of {len(docs_split)} documents were split, with the '
    f'following doc_keys: {docs_split}')
    
    return updated_dset


def main(dset_path):

    # Read in the dataset
    print('\nReading in daataset...')
    with jsonlines.open(dset_path) as reader:
        dset = []
        for obj in reader:
            dset.append(obj)

    # Process
    print('\nProcessing dataset...')
    updated_dset = split_docs(dset)

    # Save
    print('\nSaving dataset...')
    outname = splitext(dset_path)[0] + '_SPLIT' + splitext(dset_path)[1]
    with jsonlines.open(outname, 'w') as writer:
        writer.write_all(updated_dset)
    print(f'Saved split output as {outname}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split docs to save memory')

    parser.add_argument('dset_path', type=str,
            help='Path to dataset to process. Output is saved to same location '
            'with "_SPLIT" appended to the original filename')

    args = parser.parse_args()

    args.dset_path = abspath(args.dset_path)

    main(args.dset_path)
            