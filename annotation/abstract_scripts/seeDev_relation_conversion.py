"""
In the SeeDev binary relations dataset, events have been decomposed into binary
relations. However, the format is still the Event format, there are just only
two arguments in each, and brat is unable to render these annotations.
Therefore, this script is to change the "E" in the id to an "R", as well as to
change the event-specific argument names to "Arg1" and "Arg2".

Additionally, the entity annotations contained in the .a1 (input annotations)
file are not contained in the .a2 (output annotations) file, which means that
brat can't render annotations from just the .a2 file. Therefore, this script
also combines the .a1 and .a2 annotations.

Author: Serena G. Lotreck
"""
import argparse
from os.path import abspath, splitext, isfile
from os import listdir
from tqdm import tqdm


def convert_rels(a2):
    """
    Changes E to R and event arg names to Arg1 and Arg2.

    parameters:
        a2, str: annotation file contents to convert

    returns:
        updated_a2, str: updated annotation list
    """
    a2_list = a2.split('\n')

    updated_a2 = ''
    for rel in a2_list:
        if rel == '':
            continue
        elif rel[0] == 'E':
            newrel = ''
            relid, body = rel.split('\t')
            new_id = 'R' + relid[1:]
            newrel += new_id + '\t'
            splitbody = body.split(' ')
            newrel += splitbody[0] + ' '
            arg1 = splitbody[1].split(':')[1]
            arg2 = splitbody[2].split(':')[1]
            newrel += f'Arg1:{arg1} Arg2:{arg2}\n'
            updated_a2 += newrel
        else:
            updated_a2 += rel

    return updated_a2


def main(input_dir, output_dir):

    # Get the unique file names
    all_files = listdir(input_dir)
    unique_names = list(set([splitext(f)[0] for f in all_files if isfile(f'{input_dir}/{f}')]))

    # For each unique entry, convert
    for doc in tqdm(unique_names):
        try:
            with open(f'{input_dir}/{doc}.a1') as myf:
                a1 = myf.read()
            with open(f'{input_dir}/{doc}.a2') as myf:
                a2 = myf.read()
            if a1[-1] != '\n':
                a1 += '\n'
            a2 = convert_rels(a2)
            full_ann = a1 + a2
            with open(f'{output_dir}/{doc}.ann', 'w') as myf:
                myf.write(full_ann)
            with open(f'{input_dir}/{doc}.txt') as myf:
                txt_list = myf.readlines()
                txt_list = [l.strip() for l in txt_list]
                txt = ' '.join(txt_list)
            with open(f'{output_dir}/{doc}.txt', 'w') as myf:
                myf.write(txt)
        except FileNotFoundError:
            print(f'Non-annotation doc found: {doc}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Format SeeDev')

    parser.add_argument('input_dir', type=str,
            help='Directory with .a1 and .a2 files to convert')
    parser.add_argument('output_dir', type=str,
            help='Directory to save merged files')

    args = parser.parse_args()

    args.intput_dir = abspath(args.input_dir)
    args.output_dir = abspath(args.output_dir)

    main(args.input_dir, args.output_dir)