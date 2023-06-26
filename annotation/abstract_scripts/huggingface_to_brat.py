"""
Script to convert huggingface datasets to brat format for visualization.
Designed specifically for ChemProt and BioInfer.

Author: Serena G. Lotreck
"""
import argparse
from os.path import abspath
from datasets import load_dataset


def get_rels(ann, doc, dataset_name, ent_convert):
    """
    Format rels into brat.

    parameters:
        ann, str: ann string with entities
        doc, dict: doc to format
        dataset_name, str: name of dataset being processed
        ent_convert, dict: keys are old ids and values are new ids if dataset is
            bioinfer, else empty dict

    returns:
        ann, str: updated with rels
    """
    dropped_rels = 0
    if dataset_name == 'bigbio/chemprot':
        for i in range(len(doc['relations']['type'])):
            rel_str = f"R{i}\t{doc['relations']['type'][i]} Arg1:{doc['relations']['arg1'][i]} Arg2:{doc['relations']['arg2'][i]}\n"
            ann += rel_str
    elif dataset_name == 'bigbio/bioinfer':
        for i in range(len(doc['relations'])):
            rel = doc['relations'][i]
            try:
                rel_id = f'R{i+1-dropped_rels}'
                rel_str = f"{rel_id}\t{rel['type']} Arg1:{ent_convert[rel['arg1_id']]} Arg2:{ent_convert[rel['arg2_id']]}\n"
            except KeyError:
                dropped_rels += 1
                rel_str = ''
            ann += rel_str

    if dataset_name == 'bigbio/bioinfer':
        print(f'{dropped_rels} relations of {len(doc["relations"])} dropped because they relied on dropped entities')
    return ann


def get_ents(ann, doc, dataset_name):
    """
    Format ents into brat.

    parameters:
        ann, str: empty ann string
        doc, dict: doc to format
        dataset_name, str: name of dataset being processed

    returns:
        ann, str: updated with entities
        ent_convert, dict: keys are original entity IDs, values are new IDs
            in brat ann doc. Empty dict if not necessary
    """
    ent_convert = {}
    dropped_ents = 0
    if dataset_name == 'bigbio/chemprot':
        for i in range(len(doc['entities']['id'])):
            assert len(doc['entities']['offsets'][i]) == 2
            ent_str = f"{doc['entities']['id'][i]}\t{doc['entities']['type'][i]} {doc['entities']['offsets'][i][0]} {doc['entities']['offsets'][i][1]}\t{doc['entities']['text'][i]}\n"
            ann += ent_str
    elif dataset_name == 'bigbio/bioinfer':
        for i in range(len(doc['entities'])):
            ent = doc['entities'][i]
            if len(ent['offsets']) == 1:
                old_id = ent['id']
                new_id = f'T{i+1-dropped_ents}'
                ent_convert[old_id] = new_id
                ent_str = f"{new_id}\t{ent['type']} {ent['offsets'][0][0]} {ent['offsets'][0][1]}\t{ent['text'][0]}\n"
            else:
                dropped_ents += 1
                ent_str = ''
            ann += ent_str

    if dataset_name == 'bigbio/bioinfer':
        print(f'{dropped_ents} entities of {len(doc["entities"])} dropped due to being discontinuous')

    return ann, ent_convert


def main(dataset_name, doc_identifier, out_loc):

    # Read in dataset
    full_dset = load_dataset(dataset_name)

    # Go through splits and make brat docs for all of them
    for dset_split, dset in full_dset.items():
        for doc in dset:
            doc_id = doc[doc_identifier]
            txt = doc['text']
            ann = ''
            ann, ent_convert = get_ents(ann, doc, dataset_name)
            ann = get_rels(ann, doc, dataset_name, ent_convert)
            ann_outname = f'{out_loc}/{doc_id}.ann'
            txt_outname = f'{out_loc}/{doc_id}.txt'
            with open(ann_outname, 'w') as myf:
                myf.write(ann)
            with open(txt_outname, 'w') as myf:
                myf.write(txt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert huggingface to brat')

    parser.add_argument('dataset_name', type=str,
            help='String that specifies a Huggingface dataset')
    parser.add_argument('doc_identifier', type=str,
            help='String key used to indicate the column of unique instance '
            '(document) identifier in the dataset; eg "pmid" or "document_id"')
    parser.add_argument('out_loc', type=str,
            help='Path to save outputs')

    args = parser.parse_args()

    args.out_loc = abspath(args.out_loc)

    main(args.dataset_name, args.doc_identifier, args.out_loc)