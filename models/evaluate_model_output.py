"""
Given the output of a model in DyGIE++ format and a gold standard annotation,
calculate precision, recall and F1 for entity and relation prediction.

Author: Serena G. Lotreck
"""
import argparse
from os.path import abspath, basename, join, splitext
from os import listdir
import warnings
import sys
sys.path.append('../annotation/abstract_scripts')
from map_dataset_types import map_jsonl
from dygie.training.f1 import compute_f1  # Must have dygiepp developed in env
import jsonlines
import pandas as pd
import numpy as np


def calculate_CI(prec_samples, rec_samples, f1_samples):
    """
    Calculates CI from bootstrap samples using the percentile method with
    alpha = 0.05 (95% CI).

    parameters:
        prec_samples, list of float: list of precision values for bootstraps
        rec_samples, list of float: list of recall values for bootstraps
        f1_samples, list of float: list of f1 values for bootstraps

    returns list with elements:
        prec_CI, tuple of float: CI for precision
        rec_CI, tuple of float: CI for recall
        f1_CI, tuple of float: CI for F1 score
    """
    alpha = 0.05
    CIs = {}
    for name, samp_set in {
            'prec_samples': prec_samples,
            'rec_samples': rec_samples,
            'f1_samples': f1_samples
    }.items():
        lower_bound = np.percentile(samp_set, 100 * (alpha / 2))
        upper_bound = np.percentile(samp_set, 100 * (1 - alpha / 2))
        name = name.split('_')[0] + '_CI'
        CIs[name] = (lower_bound, upper_bound)

    return [CIs['prec_CI'], CIs['rec_CI'], CIs['f1_CI']]


def check_rel_matches(pred, gold_sent, check_types=False, sym_rels=None):
    """
    Checks for matches of pred in gold_sent, order-agnostically if check_types
    is False or if the type of pred is symmetrical.

    Note that pred/gold are relative, can be swapped to get false negatives by
    comparing a "pred" from the gold standard against the "gold_sent" of
    predictions from the model, as is done in get_doc_rel_counts.

    parameters:
        pred, list: 4 integers (entity bounds) and a string (relation type),
            with optional softmax and logit scores
        gold_sent, list of list: internal lists are relation representations
            with the same format as pred
        check_types, bool: Whether or not to consider types in evaluations
        sym_rels, list of str or None: whether any of the relations to be
            checked are symmetrical

    returns:
        True if a match exists in gold_sent, False otherwise
    """
    # Make sym_rels an empty list if there are None
    sym_rels = [] if sym_rels is None else sym_rels

    # Checking when order matters is straighforweard
    if check_types and pred[4] not in sym_rels:
        gold_sent = [g[:5] for g in gold_sent] # In case there are softmax/logit
        if pred[:5] in gold_sent:
            return True
        else:
            return False

    # If we don't care about types, or if the type is symmetric, check order-agnostically
    elif not check_types or pred[4] in sym_rels:

        # Need to mark down whether or not we care about type
        if pred[4] in sym_rels:
            vals_compare = 5
        else: vals_compare = 4

        # Make all gold rels into strings to make it easier to search for ent idx pairs
        gold_rel_strs = []
        for rel in gold_sent:
            rel_str = ' '.join([str(i) for i in rel[:vals_compare]])
            gold_rel_strs.append(rel_str)

        # Make each ent and type in pred into separate string so we can search order-agnostically
        pred_ent_1_str = ' '.join([str(i) for i in pred[:2]])
        pred_ent_2_str = ' '.join([str(i) for i in pred[2:4]])
        pred_rel_type = pred[4]

        # Check if ent 1 is in one of the relations
        ent1_list = [True if pred_ent_1_str in i else False for i in gold_rel_strs]
        # Check if ent 2 is in one of the relations
        ent2_list = [True if pred_ent_2_str in i else False for i in gold_rel_strs]
        # Check if the relation type is in one of the relations
        type_list = [True if pred_rel_type in i else False for i in gold_rel_strs]

        # Indices that all have True in them are matches
        if pred[4] in sym_rels:
             matches_list = [
                True if (ent1_list[i] & ent2_list[i] & type_list[i]) else False
                for i in range(len(gold_rel_strs))
                ]
        else:
            matches_list = [
                True if (ent1_list[i] & ent2_list[i]) else False
                for i in range(len(gold_rel_strs))
                ]

        # Determine return type
        if matches_list.count(True) >= 1:
            return True
        else:
            return False


def get_doc_ent_counts(doc, gold_std, ent_pos_neg, mismatch_rows,
        check_types=False):
    """
    Get the true/false positives and false negatives for entity prediction for
    a single document.

    parameters:
        doc, dict: dygiepp-formatted dictionary, with keys "predicted_ner" and
            "ner"
        gold_std, dict, dygiepp-formatted dicitonary with gold standard for
            this doc, same key requirements as doc
        ent_pos_neg, dict: keys are "tp", "fp", "fn". Should keep passing the
            same object for each doc to get totals for the entire set of
            documents.
        mismatch_rows, dict: empty dict or dict with mismatch_cols as keys and
            lists as rows
        check_types, bool: Whether or not to consider types in evaluations

    returns:
        ent_pos_neg, dict: updated match counts for entities
        mismatch_rows, dict: empty dict if emtpy dict was passed, otherwise
            updated mismatch_rows dict
    """
    # Whether or not we care about type determines what index we compare to
    vals_compare = 2 if not check_types else 3

    # Go through each sentence for entities
    sent_num = 0
    for pred_sent, gold_sent in zip(doc['predicted_ner'], gold_std['ner']):

        # Iterate through predictions and check for them in gold standard
        for pred in pred_sent:
            found = False
            for gold_ent in gold_sent:
                if pred[:vals_compare] == gold_ent[:vals_compare]:
                    ent_pos_neg['tp'] += 1
                    if len(mismatch_rows.keys()) != 0:
                        mismatch_rows['doc_key'].append(doc['doc_key'])
                        mismatch_rows['mismatch_type'].append(1)
                        mismatch_rows['sent_num'].append(sent_num)
                        mismatch_rows['ent_list'].append(gold_ent)
                        mismatch_rows['ent_type'].append(gold_ent[2])
                    found = True
            if not found:
                ent_pos_neg['fp'] += 1

        # Iterate through gold standard and check for them in predictions
        for gold in gold_sent:
            found = False
            for pred in pred_sent:
                if gold[:vals_compare] == pred[:vals_compare]:
                    found = True
            if not found:
                ent_pos_neg['fn'] += 1
                if len(mismatch_rows.keys()) != 0:
                    mismatch_rows['doc_key'].append(doc['doc_key'])
                    mismatch_rows['mismatch_type'].append(0)
                    mismatch_rows['sent_num'].append(sent_num)
                    mismatch_rows['ent_list'].append(gold)
                    mismatch_rows['ent_type'].append(gold[2])
        sent_num += 1

    return ent_pos_neg, mismatch_rows


def get_doc_rel_counts(doc, gold_std, rel_pos_neg, doc_key, check_types=False,
                        sym_rels=None):
    """
    Get the true/false positives and false negatives for relation prediction for
    a single document.

    parameters:
        doc, dict: dygiepp-formatted dictionary, with keys "predicted_relations"
            and "relations"
        gold_std, dict, dygiepp-formatted dicitonary with gold standard for
            this doc, same key requirements as doc
        rel_pos_neg, dict: keys are "tp", "fp", "fn". Should keep passing the
            same object for each doc to get totals for the entire set of
            documents.
        doc_key, str: doc ID, to use for warning if there are exact duplicate
            relations in any of the sentences
        check_types, bool: Whether or not to consider types in evaluations
        sym_rels, list of str or None: whether any of the relations to be
            checked are symmetrical

    returns:
        rel_pos_neg, dict: updated match counts for relations
    """
    # Go through each sentence for relations
    for pred_sent, gold_sent in zip(doc['predicted_relations'],
                                    gold_std['relations']):

        # Iterate through the predictions and check for them in the gold standard
        for pred in pred_sent:
            matched = check_rel_matches(pred, gold_sent, check_types, sym_rels)
            if matched:
                rel_pos_neg['tp'] += 1
            else:
                rel_pos_neg['fp'] += 1

        # Iterate through gold standard and check for them in predictions
        for gold in gold_sent:
            matched = check_rel_matches(gold, pred_sent, check_types, sym_rels)
            if matched:
                continue
            else:
                rel_pos_neg['fn'] += 1

    return rel_pos_neg


def get_f1_input(gold_standard_dicts,
                 prediction_dicts,
                 input_type,
                 mismatch_rows={}, check_types=False, sym_rels=None):
    """
    Get the number of true and false postives and false negatives for the
    model to calculate the following inputs for compute_f1 for both entities
    and relations:
        predicted = true positives + false positives
        gold = true positives + false negatives
        matched = true positives

    parameters:
        gold_standard_dicts, list of dict: dygiepp formatted annotations
        prediction_dicts, list of dict: dygiepp formatted predictions
        input_type, str: 'ent' or 'rel', determines which of the prediction
            types will be evaluated
        mismatch_rows, dict: empty dict (default) or dict where keys are
            columns for mismatch_df. Only used if not an empty dict.
        check_types, bool: Whether or not to consider types in evaluations
        sym_rels, list of str or None: whether any of the relations to be
            checked are symmetrical

    returns:
        predicted, int
        gold, int
        matched, int
        mismatch_rows, dict: empty dict if empty dict was passed, updated
            mismatch_rows dict if dict with keys was passed
    """
    pos_neg = {'tp': 0, 'fp': 0, 'fn': 0}

    # Rearrange gold standard so that it's a dict with keys that are doc_id's
    gold_standard_dict = {d['doc_key']: d for d in gold_standard_dicts}

    # Go through the docs
    for doc in prediction_dicts:

        # Get the corresponding gold standard
        gold_std = gold_standard_dict[doc['doc_key']]

        # Get tp/fp/fn counts for this document
        if input_type == 'ent':
            pos_neg, mismatch_rows = get_doc_ent_counts(
                doc, gold_std, pos_neg, mismatch_rows, check_types)

        elif input_type == 'rel':
            pos_neg = get_doc_rel_counts(doc, gold_std, pos_neg,
                                         doc["doc_key"], check_types, sym_rels)

    predicted = pos_neg['tp'] + pos_neg['fp']
    gold = pos_neg['tp'] + pos_neg['fn']
    matched = pos_neg['tp']

    return (predicted, gold, matched, mismatch_rows)


def draw_boot_samples(pred_dicts, gold_std_dicts, num_boot, input_type,
                        check_types=False, sym_rels=None):
    """
    Draw bootstrap samples.

    parameters:
        pred_dicts, list of dict: dicts of model predictions
        gold_std_dicts, list of dict: dicts of gold standard annotations
        num_boot, int: number of bootstrap samples to draw
        input_type, str: 'ent' or 'rel'
        check_types, bool: Whether or not to consider types in evaluations
        sym_rels, list of str or None: whether any of the relations to be
            checked are symmetrical

    returns:
        prec_samples, list of float: list of precision values for bootstraps
        rec_samples, list of float: list of recall values for bootstraps
        f1_samples, list of float: list of f1 values for bootstraps
    """
    prec_samples = []
    rec_samples = []
    f1_samples = []

    # Draw the boot samples
    for _ in range(num_boot):

        # Sample prediction dicts with replacement
        pred_samp = np.random.choice(pred_dicts,
                                     size=len(pred_dicts),
                                     replace=True)

        # Get indices of the sampled instances in the pred_dicts list
        idx_list = [pred_dicts.index(i) for i in pred_samp]

        # Since the lists are sorted the same, can use indices to get equivalent
        # docs in gold std
        gold_samp = np.array([gold_std_dicts[i] for i in idx_list])

        # Calculate performance for the sample
        pred, gold, match, _ = get_f1_input(gold_samp, pred_samp, input_type,
                                            check_types=check_types, sym_rels=sym_rels)
        prec, rec, f1 = compute_f1(pred, gold, match)

        # Append each of the performance values to their respective sample lists
        prec_samples.append(prec)
        rec_samples.append(rec)
        f1_samples.append(f1)

    return (prec_samples, rec_samples, f1_samples)


def get_performance_row(pred_file, gold_std_file, bootstrap,
                        num_boot, df_rows, mismatch_rows, map_types=False, entity_map='',
                        relation_map='', check_types=False, sym_rels=None):
    """
    Gets performance metrics and returns as a list.

    parameters:
        pred_file, str: name of the file used for predictions
        gold_std_file, str: name of the gold standard file
        bootstrap, bool, whether or not to bootstrap a confidence interval
        num_boot, int: if bootstrap is True, how many bootstrap samples to take
        df_rows, dict: keys are column names, values are lists of performance
            values and CIs to which new results will be appended
        mismatch_rows, dict: if save_mismatches, keys are mismatch col names,
            values are lists. Else, empty dict
        map_types, bool: whether or not to may prediction types to new ontology
        entity_map, str: path to entity map if map_types, else ''
        relation_map, str: path to relation map if map_types, else ''
        check_types, bool: Whether or not to consider types in evaluations
        sym_rels, list of str or None: whether any of the relations to be
            checked are symmetrical


    returns:
        df_rows, dict: df_rows updated with new row values
        mismatch_rows, dict: mismtach_updated if save_mismatches, else empty
            dict
    """
    # Read in the files
    gold_std_dicts = []
    with jsonlines.open(gold_std_file) as reader:
        for obj in reader:
            gold_std_dicts.append(obj)
    pred_dicts = []
    with jsonlines.open(pred_file) as reader:
        for obj in reader:
            pred_dicts.append(obj)

    # Map types if requested
    if map_types:
        verboseprint(
            '\nMapping entity and relation types to chosen ontologies...')
        de, dr = map_jsonl(pred_dicts, entity_map, relation_map)
        verboseprint(
            f'{de} entities were dropped because their types didn\'t '
            f'have an equivalent, and {dr} relations were dropped for the same '
            'reason or because they relied on a dropped entity.')

    # Make sure all prediction files are also in the gold standard
    gold_doc_keys = [g['doc_key'] for g in gold_std_dicts]
    for doc in pred_dicts:
        if doc['doc_key'] in gold_doc_keys:
            continue
        else:
            verboseprint(
                f'Document {doc["doc_key"]} is not in the gold standard. '
                'Skipping this document for performance calculation.')
            _ = pred_dicts.remove(doc)

    # Sort pred and gold lists by doc key to ensure they're in the same order
    gold_std_dicts = sorted(gold_std_dicts, key=lambda d: d['doc_key'])
    pred_dicts = sorted(pred_dicts, key=lambda d: d['doc_key'])

    # Check if the predictions include relations
    pred_rels = True
    try:
        [d['predicted_relations'] for d in pred_dicts]
    except KeyError:
        pred_rels = False

    # Check if there are any relations in the gold standard
    gold_rels = False
    for doc in gold_std_dicts:
        for sent in doc['relations']:
            if len(sent) != 0:
                gold_rels = True
    if not gold_rels:
        warnings.warn(
            '\n\nThere are no gold standard relation annotations. '
            'Performance values and CIs will be 0 for models that predict '
            'relations, please disregard.')

    # Bootstrap sampling
    if bootstrap:
        ent_boot_samples = draw_boot_samples(pred_dicts, gold_std_dicts,
                                             num_boot, 'ent', check_types,
                                             sym_rels)
        if pred_rels:
            rel_boot_samples = draw_boot_samples(pred_dicts, gold_std_dicts,
                                                 num_boot, 'rel', check_types,
                                                 sym_rels)

        # Calculate confidence interval
        ent_CIs = calculate_CI(ent_boot_samples[0], ent_boot_samples[1],
                               ent_boot_samples[2])

        if pred_rels:
            rel_CIs = calculate_CI(rel_boot_samples[0], rel_boot_samples[1],
                                   rel_boot_samples[2])
        else:
            rel_CIs = [np.nan for i in range(3)]

        # Get means
        ent_means = [np.mean(samp) for samp in ent_boot_samples]
        if pred_rels:
            rel_means = [np.mean(samp) for samp in rel_boot_samples]
        else:
            rel_means = [np.nan for i in range(3)]

        # Feels excessive but safer than returning a list, from experience
        df_rows['pred_file'].append(basename(pred_file))
        df_rows['gold_std_file'].append(basename(gold_std_file))
        df_rows['ent_precision'].append(ent_means[0])
        df_rows['ent_recall'].append(ent_means[1])
        df_rows['ent_F1'].append(ent_means[2])
        df_rows['rel_precision'].append(rel_means[0])
        df_rows['rel_recall'].append(rel_means[1])
        df_rows['rel_F1'].append(rel_means[2])
        df_rows['ent_precision_CI'].append(ent_CIs[0])
        df_rows['ent_recall_CI'].append(ent_CIs[1])
        df_rows['ent_F1_CI'].append(ent_CIs[2])
        df_rows['rel_precision_CI'].append(rel_CIs[0])
        df_rows['rel_recall_CI'].append(rel_CIs[1])
        df_rows['rel_F1_CI'].append(rel_CIs[2])

        return df_rows, {}

    else:
        # Calculate performance
        pred_ent, gold_ent, match_ent, mismatch_rows = get_f1_input(
            gold_std_dicts, pred_dicts, 'ent', mismatch_rows, check_types=check_types)
        ent_means = compute_f1(pred_ent, gold_ent, match_ent)
        if pred_rels:
            pred_rel, gold_rel, match_rel, _ = get_f1_input(
                gold_std_dicts, pred_dicts, 'rel', check_types, sym_rels)
            rel_means = compute_f1(pred_rel, gold_rel, match_rel)
        else:
            rel_means = [np.nan for i in range(3)]

        df_rows['pred_file'].append(basename(pred_file))
        df_rows['gold_std_file'].append(basename(gold_std_file))
        df_rows['ent_precision'].append(ent_means[0])
        df_rows['ent_recall'].append(ent_means[1])
        df_rows['ent_F1'].append(ent_means[2])
        df_rows['rel_precision'].append(rel_means[0])
        df_rows['rel_recall'].append(rel_means[1])
        df_rows['rel_F1'].append(rel_means[2])

        return df_rows, mismatch_rows


def main(gold_standard, out_name, predictions, check_types, bootstrap, num_boot,
         save_mismatches, map_types, entity_map, relation_map, sym_rels):

    # Calculate performance
    verboseprint('\nCalculating performance...')
    if bootstrap:
        cols = [
            'pred_file', 'gold_std_file', 'ent_precision', 'ent_recall',
            'ent_F1', 'rel_precision', 'rel_recall', 'rel_F1',
            'ent_precision_CI', 'ent_recall_CI', 'ent_F1_CI',
            'rel_precision_CI', 'rel_recall_CI', 'rel_F1_CI'
        ]
    else:
        cols = [
            'pred_file', 'gold_std_file', 'ent_precision', 'ent_recall',
            'ent_F1', 'rel_precision', 'rel_recall', 'rel_F1'
        ]
    df_rows = {k: [] for k in cols}
    if save_mismatches:
        # 'mismatch_type' column: 1 if the model correctly matched the gold
        # standard (true positive), 0 if the model failed to match the gold
        # standard (false negative)
        # doc_key + sent_num + ent_list allows recovery of the text of the
        # entity later on, while ent_type makes type access easier
        mismatch_cols = [
            'doc_key', 'mismatch_type', 'sent_num', 'ent_list', 'ent_type',
            'model'
        ]
        mismatch_rows = {k: [] for k in mismatch_cols}
    else:
        # To avoid having to add too much code, if the user doesn't want
        # mismatches, will just add nothing to an empty dict to allow the same
        # returns
        mismatch_rows = {}

    for model in predictions:
        verboseprint(f'\nEvaluating model predictions from file {model}...')
        df_rows, mismatch_rows = get_performance_row(model, gold_standard,
                                                     bootstrap,
                                                     num_boot, df_rows,
                                                     mismatch_rows=mismatch_rows, map_types=map_types,
                                                     check_types=check_types,
                                                     sym_rels=sym_rels)

        if save_mismatches:
            # Add the model string onto the mismatch_rows df
            mismatch_col_lens = list(
                set([len(v) for k, v in mismatch_rows.items()]))
            assert len(mismatch_col_lens) == 2
            missing_model_name_num = abs(mismatch_col_lens[0] -
                                         mismatch_col_lens[1])
            missing_model_names = [
                model for i in range(missing_model_name_num)
            ]
            mismatch_rows['model'].extend(missing_model_names)

    # Make df
    verboseprint('\nMaking dataframe...')
    df = pd.DataFrame(df_rows, columns=cols)
    verboseprint(f'Snapshot of dataframe:\n{df.head()}')

    # Make mismatch df if specified
    if save_mismatches:
        verboseprint('\nMaking mismatch dataframe...')
        mismatch_df = pd.DataFrame(mismatch_rows, columns=mismatch_cols)
        verboseprint(f'Snapshot of dataframe:\n{mismatch_df.head()}')

    # Save
    verboseprint(f'\nSaving performance file as {out_name}')
    df.to_csv(out_name, index=False)
    if save_mismatches:
        mismatch_out_name = splitext(out_name)[0] + '_MISMATCHES.csv'
        verboseprint(f'\nSaving mismatch file as {mismatch_out_name}')
        mismatch_df.to_csv(mismatch_out_name, index=False)

    verboseprint('\nDone!\n')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Calculate performance')

    parser.add_argument('gold_standard',
        type=str,
        help='Path to dygiepp-formatted gold standard data')
    parser.add_argument('out_name',
        type=str,
        help='Name of save file for output (including path)')
    parser.add_argument(
        'prediction_dir',
        type=str,
        help='Path to directory with dygiepp-formatted model outputs')
    parser.add_argument(
        '--check_types',
        action='store_true',
        help='Whether or not to consider types when performing evaluations')
    parser.add_argument(
        '--bootstrap',
        action='store_true',
        help='Whether or not to bootstrap a confidence interval '
        'for performance metrics. Specify for deterministic model '
        'output.')
    parser.add_argument(
        '-num_boot',
        type=int,
        help='Number of bootstrap samples to use for calculating CI, '
        'default is 500',
        default=500)
    parser.add_argument(
        '-use_prefix',
        type=str,
        help='If a prefix is provided, only calculates performance for '
        'files beginning with the prefix in the directory.',
        default='')
    parser.add_argument(
        '--map_types',
        action='store_true',
        help='Whether or not to map predicted types to a different ontology')
    parser.add_argument(
        '-entity_map',
        type=str,
        help='Path to entity map, required if --map_types is provided',
        default='')
    parser.add_argument(
        '-relation_map',
        type=str,
        help='Path to relation map, required if --map_types is provided',
        default='')
    parser.add_argument('-sym_rels', type=str, nargs='+',
        help='Relations that should be evaluated symmetrically',
        default='')
    parser.add_argument(
        '--save_mismatches',
        action='store_true',
        help='Whether or not to save the types and numbers of  mismatches '
        'for all models. To be used in downstream analysis. Can only be '
        'specified if --bootstrap is not specified.')
    parser.add_argument('--verbose',
        '-v',
        action='store_true',
        help='If provided, script progress will be printed')

    args = parser.parse_args()

    if args.save_mismatches:
        assert not args.bootstrap, (
            '--save_mismatches and --bootstrap '
            'cannot be specified together, please remove --bootstrap if '
            'you would like to use --save_mismatches')

    if args.map_types:
        assert (args.entity_map != '') and (args.relation_map != ''), (
            'One or more of entity_map and relation_map has not been specified '
            ', both are required when map_types is passed')

    args.gold_standard = abspath(args.gold_standard)
    args.out_name = abspath(args.out_name)
    args.prediction_dir = abspath(args.prediction_dir)
    args.entity_map = abspath(args.entity_map)
    args.relation_map = abspath(args.relation_map)

    if args.sym_rels == '':
        args.sym_rels = None

    verboseprint = print if args.verbose else lambda *a, **k: None

    pred_files = [
        join(args.prediction_dir, f) for f in listdir(args.prediction_dir)
        if f.startswith(args.use_prefix)
    ]

    main(args.gold_standard, args.out_name, pred_files, args.check_types,
         args.bootstrap, args.num_boot, args.save_mismatches, args.map_types,
         args.entity_map, args.relation_map, args.sym_rels)
