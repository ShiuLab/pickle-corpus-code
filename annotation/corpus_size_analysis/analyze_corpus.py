"""
Analyze the effect of training corpus size on model performance.

Author: Serena G. Lotreck
"""
import argparse
from os.path import abspath, basename, splitext
from corpus_subset_utils import subset_corpus
from subprocess import run


def main(full_corpus_path, dataset_name, dygiepp_path, config_template,
            train_job_template, test_size, dev_size, start_train_size,
            train_subset_size, num_train_subsets, out_loc):

    # Sample to get the train sets
    trainsets, dev_name, test_name = subset_corpus(full_corpus_path,
                                        start_train_size, train_subset_size,
                                        num_train_subsets, dev_size, test_size,
                                        dataset_name, out_loc)

    #Read in templates
    with open(config_template) as myf:
        config = myf.read()
    with open(train_job_template) as myf:
        train_job = myf.read()

    # Iterate through training sets
    for tset in trainsets:

        # Format training config
        current_config = config.replace('XXXX', tset)
        current_config = current_config.replace('YYYY', dev_name)
        current_config = current_config.replace('ZZZZ', test_name)
        train_base = splitext(basename(tset)[1])[0]
        current_conf_name = f'{dygiepp_path}/training_config/{train_base}.jsonnet'
        with open(current_conf_name, 'w') as myf:
            myf.write(current_config)

        # Format training job
        num_docs = splitext(tset)[0].split('_')[-1]
        current_train_job = train_job.replace('XXXX', num_docs)
        current_train_job = current_train_job.replace('YYYY', dataset_name)
        current_train_job = current_train_job.replace('ZZZZ', test_name)
        current_train_name = f'{out_loc}/{train_base}_job.sb'
        with open(current_train_name, 'w') as myf:
            myf.write(current_train_job)

        # Submit job
        #run(['sbatch', current_train_name])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze effect of train size')

    parser.add_argument('full_corpus_path', type=str,
            help='Path to full corpus from which to sample')
    parser.add_argument('dataset_name', type=str,
            help='Dataset name')
    parser.add_argument('dygiepp_path', type=str,
            help='Path to top level of dygiepp repo')
    parser.add_argument('config_template', type=str,
            help='Path to file with XXXX, YYYY and ZZZ in place of the '
            'filenames for train, dev and test, respectively')
    parser.add_argument('train_job_template', type=str,
            help='Path to job script with XXXX in place of the document size, '
            'YYYY in place of the dataset name, and ZZZZ in place of the full '
            'path to the test file')
    parser.add_argument('test_size', type=int,
            help='Num docs for test set')
    parser.add_argument('dev_size', type=int,
            help='Num docs for dev set')
    parser.add_argument('start_train_size', type=int,
            help='Number of docs to include in smallest train set')
    parser.add_argument('train_subset_size', type=int,
            help='Number of docs to add each time')
    parser.add_argument('num_train_subsets', type=int,
            help='Number of times to add to train set')
    parser.add_argument('out_loc', type=str,
            help='Place to save output files. Job error and output files will '
            'be saved to the directory from which the script is run.')

    args = parser.parse_args()

    args.full_corpus_path = abspath(args.full_corpus_path)
    args.dygiepp_path = abspath(args.dygiepp_path)
    args.config_template = abspath(args.config_template)
    args.train_job_template = abspath(args.train_job_template)
    args.out_loc = abspath(args.out_loc)

    main(args.full_corpus_path, args.dataset_name, args.dygiepp_path,
        args.config_template, args.train_job_template, args.test_size,
        args.dev_size, args.start_train_size, args.train_subset_size,
        args.num_train_subsets, args.out_loc)