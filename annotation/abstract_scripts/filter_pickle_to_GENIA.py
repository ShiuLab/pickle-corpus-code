"""
Remove PICKLE types that aren't mapped to GENIA types.

Author: Serena G. Lotreck
"""
import argparse
from os.path import abspath, splitext
import json
import jsonlines


def main(data_path, type_map):

    # Read in data
    print('\nReading in original dataset...')
    with jsonlines.open(data_path) as reader:
        data = []
        for obj in reader:
            data.append(obj)

    # Read in map
    print('\nReading in type map...')
    with open(type_map) as myf:
        type_map_dict = json.load(myf)

    # Filter
    print('\nFiltering dataset...')
    filtered = []
    for doc in data:
        updated_doc = {}
        for key, value in doc.items():
            if key != 'ner':
                updated_doc[key] = value
            else:
                updated_ner = []
                for sent in value:
                    updated_sent = []
                    for ent in sent:
                        if ent[2].lower() in [t.lower() for t in type_map_dict.keys()]:
                            updated_sent.append(ent)
                    updated_ner.append(updated_sent)
                updated_doc[key] = updated_ner
        filtered.append(updated_doc)

    # Save
    savename = splitext(data_path)[0] + '_FILTERED_TO_GENIA' + splitext(data_path)[1]
    print(f'\nSaving out filtered dataset to {savename}')
    with jsonlines.open(savename, 'w') as writer:
        writer.write_all(filtered)
    print('\nDone!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter PICKLE')

    parser.add_argument('data_path', type=str,
            help='Path to dataset to filter')
    parser.add_argument('type_map', type=str,
            help='Path to type map')

    args = parser.parse_args()

    args.data_path = abspath(args.data_path)
    args.type_map = abspath(args.type_map)

    main(args.data_path, args.type_map)
