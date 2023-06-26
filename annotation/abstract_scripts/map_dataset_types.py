"""
Maps entity and relation types from one dataset to another's. If a type doesn't
map, then the annotations with that type are dropped.

Author: Serena G. Lotreck
"""
import argparse
from os.path import abspath, splitext, isfile
import json
from os import listdir
from subprocess import run
from collections import OrderedDict


def map_ann(ann, entity_map, relation_map):
    """
    Convert annotation types.

    parameters:
        ann, str: annotation file to map
        entity_map, dict: entity type map
        relation_map, dict: relation type map

    returns:
        mapped_ann, str: mapped ann file contents
    """
    # Make a dict where keys are annotation IDs, values are the rest of the line
    ann_list = ann.split('\n')
    ann_dict = OrderedDict([(l.split('\t')[0], '\t'.join(l.split('\t')[1:])) for l in ann_list])

    # Split into entities and relations
    ent_dict = OrderedDict([(k, v) for k,v in ann_dict.items() if k[0] == 'T'])
    rel_dict = OrderedDict([(k, v) for k,v in ann_dict.items() if k[0] == 'R'])

    # Go through entities first, remove unmappable
    updated_ent_dict = {}
    for ent_id, rest_ent in ent_dict.items():
        middle, text = rest_ent.split('\t')
        middle_split = middle.split(' ')
        new_type = entity_map[middle_split[0]]
        if new_type == '':
            continue
        else:
            new_middle = [new_type] + middle_split[1:]
            new_middle = ' '.join(new_middle)
            new_rest = new_middle + '\t' + text
            updated_ent_dict[ent_id] = new_rest

    # Then go through relations
    updated_rel_dict = {}
    for rel_id, rest_rel in rel_dict.items():
        middle = rest_rel.split(' ')
        arg1_id = middle[1][middle[1].index(':')+1:]
        arg2_id = middle[2][middle[2].index(':')+1:]
        if (arg1_id not in updated_ent_dict.keys()) or (arg2_id not in updated_ent_dict.keys()):
            continue
        else:
            new_type = relation_map[middle[0]]
            if new_type == '':
                continue
            else:
                updated_rest = [new_type] + middle[1:]
                updated_rel_dict[rel_id] = ' '.join(updated_rest)

    # Combine back into ann file
    ent_list = [f'{k}\t{v}\n' for k, v in updated_ent_dict.items()]
    rel_list = [f'{k}\t{v}\n' for k,v in updated_rel_dict.items()]
    mapped_ann = ''.join(ent_list + rel_list)

    return mapped_ann


def main(input_dir, entity_map, relation_map, output_dir):

    # Read in the maps
    with open(entity_map) as myf:
        entity_map = json.load(myf)
    with open(relation_map) as myf:
        relation_map = json.load(myf)

    # Get unique instances
    f_list = listdir(input_dir)
    unique_list = [splitext(f)[0] for f in f_list if isfile(f'{input_dir}/{f}')]

    # Read in and map all ann files, copy text file
    for doc in unique_list:
        with open(f'{input_dir}/{doc}.ann') as myf:
            ann = myf.read().strip()
        mapped_ann = map_ann(ann, entity_map, relation_map)
        with open(f'{output_dir}/{doc}.ann', 'w') as myf:
            myf.write(mapped_ann)
        cp_text = ['cp', f'{input_dir}/{doc}.txt', output_dir]
        run(cp_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Map annotations')

    parser.add_argument('input_dir', type=str,
            help='Path to directory containing dataset files to map')
    parser.add_argument('entity_map', type=str,
            help='Path to json file with entity type mappings')
    parser.add_argument('relation_map', type=str,
            help='Path to json file with relation type mappings')
    parser.add_argument('output_dir', type=str,
            help='Path to save mapped files')

    args = parser.parse_args()

    args.input_dir = abspath(args.input_dir)
    args.entity_map = abspath(args.entity_map)
    args.relation_map = abspath(args.relation_map)
    args.output_dir = abspath(args.output_dir)

    main(args.input_dir, args.entity_map, args.relation_map,
            args.output_dir)
