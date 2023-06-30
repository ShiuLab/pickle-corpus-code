"""
Spot checks for evaluate_model_output.py

Author: Serena G. Lotreck
"""
import unittest
import sys

sys.path.append('../models/')

import numpy as np
import evaluate_model_output as emo

## Didn't write a test for drawing the samples,
## since it just randomly selects and then uses other funcs
## that already have tests

class TestCalculateCI:
    def setup_method(self):
        self.prec_samples = [0.1,0.3,0.3,0.5,0.6,0.9]
        self.rec_samples = [0.2,0.2,0.3,0.4,0.4,0.8]
        self.f1_samples = [0.1,0.3,0.4,0.5,0.7,0.7]

        # Calculated with same method as within func, feels un-kosher
        a = 0.05
        lower_p = 100*(a/2)
        upper_p = 100*(1-a/2)
        self.prec_CI = (np.percentile(self.prec_samples, lower_p),
                np.percentile(self.prec_samples, upper_p))
        self.rec_CI = (np.percentile(self.rec_samples, lower_p),
                np.percentile(self.rec_samples, upper_p))
        self.f1_CI = (np.percentile(self.f1_samples, lower_p),
                np.percentile(self.f1_samples, upper_p))

    def test_calculate_ci(self):
        prec_CI, rec_CI, f1_CI = emo.calculate_CI(self.prec_samples,
                self.rec_samples, self.f1_samples)

        assert prec_CI == self.prec_CI
        assert rec_CI == self.rec_CI
        assert f1_CI == self.f1_CI

class TestGetDocEntCountsWithoutTypes:
    maxDiff = None
    def setup_method(self):

        self.mismatch_rows_col_input = {'doc_key':[], 'mismatch_type':[],
                'sent_num':[], 'ent_list':[], 'ent_type':[]}

        self.doc1_gold = {
            "doc_key":
            "doc1",
            "sentences":
            [['Hello', 'world', ',', 'my', 'name', 'is', 'Sparty', '.'],
             [
                 'My', 'research', 'is', 'about', 'A.', 'thaliana', 'protein',
                 '5'
             ]],
            "ner": [[[0, 1, "ENTITY"], [6, 6, "ENTITY"]],
                    [[12, 13, "ENTITY"], [14, 15, "ENTITY"]]]
        }
        self.doc2_gold = {
            "doc_key": "doc2",
            "sentences": [['Hello']],
            "ner": [[]]
        }

        self.doc1_pred_perf = {
            "doc_key":
            "doc1",
            "sentences":
            [['Hello', 'world', ',', 'my', 'name', 'is', 'Sparty', '.'],
             [
                 'My', 'research', 'is', 'about', 'A.', 'thaliana', 'protein',
                 '5'
             ]],
            "predicted_ner": [[[0, 1, "Hello"], [6, 6, "Person"]],
                              [[12, 13, "Person"], [14, 15, "Protein"]]]
                            # Make sure that it still matches even if entity
                            # types are different
        }
        self.doc1_perf_mismatch_rows = {'doc_key':['doc1', 'doc1', 'doc1',
           'doc1'], 'mismatch_type':[1,1,1,1], 'sent_num':[0,0,1,1],
           'ent_list':[[0, 1, "ENTITY"], [6, 6, "ENTITY"], [12, 13, "ENTITY"],
               [14, 15, "ENTITY"]], 'ent_type':["ENTITY", "ENTITY", "ENTITY",
               "ENTITY"]}
        self.doc1_perf_dict = {'tp': 4, 'fp':0, 'fn':0}

        self.doc2_pred_perf = {
            "doc_key": "doc2",
            "sentences": [['Hello']],
            "predicted_ner": [[]]
        }
        self.doc2_perf_dict = {'tp': 0, 'fp':0, 'fn':0}

        # Use doc2 to test the case where save_mismatch isn't specified
        self.doc2_mismatch = {}

        self.doc1_pred_imperf = {
            "doc_key":
            "doc1",
            "sentences":
            [['Hello', 'world', ',', 'my', 'name', 'is', 'Sparty', '.'],
             [
                 'My', 'research', 'is', 'about', 'A.', 'thaliana', 'protein',
                 '5'
             ]],
            "predicted_ner": [[[0, 1, "ORG"]],
                              [[9, 9, "ORG"], [13, 13, "PERS"],
                               [14, 15, "PROTEIN"]]]
        }
        self.doc1_imperf_mismatch_rows = {'doc_key':['doc1', 'doc1', 'doc1', 'doc1'],
            'mismatch_type':[1,0,1,0], 'sent_num':[0,0,1,1],
            'sent_num':[0,0,1,1],
            'ent_list':[[0, 1, "ENTITY"], [6, 6, "ENTITY"],
               [14, 15, "ENTITY"], [12, 13, "ENTITY"]],
            'ent_type':["ENTITY", "ENTITY", "ENTITY",
               "ENTITY"]}

        self.doc1_imperf_dict = {'tp':2, 'fp':2, 'fn':2}

        self.doc2_pred_imperf = {
            "doc_key": "doc2",
            "sentences": [['Hello']],
            "predicted_ner": [[[0, 1, "ORG"]]]
        }
        self.doc2_imperf_dict = {'tp':0, 'fp':1, 'fn':0}

    def test_get_doc_ent_counts_doc1_perf(self):
        counts, mismatch_rows = emo.get_doc_ent_counts(self.doc1_pred_perf, self.doc1_gold,
                {'tp':0, 'fp':0, 'fn':0}, self.mismatch_rows_col_input)

        assert counts == self.doc1_perf_dict

    def test_get_doc_ent_counts_doc1_perf_mismatch_rows(self):
        counts, mismatch_rows = emo.get_doc_ent_counts(self.doc1_pred_perf, self.doc1_gold,
                {'tp':0, 'fp':0, 'fn':0}, self.mismatch_rows_col_input)

        assert mismatch_rows == self.doc1_perf_mismatch_rows

    def test_get_doc_ent_counts_doc2_perf(self):
        counts, mismatch_rows = emo.get_doc_ent_counts(self.doc2_pred_perf, self.doc2_gold,
                {'tp':0, 'fp':0, 'fn':0}, self.doc2_mismatch)

        assert counts == self.doc2_perf_dict

    def test_get_doc_ent_counts_doc2_perf_mismatch(self):
        counts, mismatch_rows = emo.get_doc_ent_counts(self.doc2_pred_perf, self.doc2_gold,
                {'tp':0, 'fp':0, 'fn':0}, self.doc2_mismatch)

        assert mismatch_rows == self.doc2_mismatch

    def test_get_doc_ent_counts_doc1_imperf(self):
        counts, mismatch_rows = emo.get_doc_ent_counts(self.doc1_pred_imperf, self.doc1_gold,
                {'tp':0, 'fp':0, 'fn':0}, self.mismatch_rows_col_input)

        assert counts == self.doc1_imperf_dict

    def test_get_doc_ent_counts_doc1_imperf_mismatch(self):
        counts, mismatch_rows = emo.get_doc_ent_counts(self.doc1_pred_imperf, self.doc1_gold,
                {'tp':0, 'fp':0, 'fn':0}, self.mismatch_rows_col_input)

        assert mismatch_rows == self.doc1_imperf_mismatch_rows

    def test_get_doc_ent_counts_doc2_imperf(self):
        counts, mismatch_rows = emo.get_doc_ent_counts(self.doc2_pred_imperf, self.doc2_gold,
                {'tp':0, 'fp':0, 'fn':0}, self.doc2_mismatch)

        assert counts == self.doc2_imperf_dict

    def test_get_doc_ent_counts_doc2_imperf_mismatch(self):
        counts, mismatch_rows = emo.get_doc_ent_counts(self.doc2_pred_imperf, self.doc2_gold,
                {'tp':0, 'fp':0, 'fn':0}, self.doc2_mismatch)

        assert mismatch_rows == self.doc2_mismatch

class TestGetDocEntCountsWithTypes:
    maxDiff = None
    def setup_method(self):

        self.mismatch_rows_col_input = {'doc_key':[], 'mismatch_type':[],
                'sent_num':[], 'ent_list':[], 'ent_type':[]}

        self.doc1_gold = {
            "doc_key":
            "doc1",
            "sentences":
            [['Hello', 'world', ',', 'my', 'name', 'is', 'Sparty', '.'],
             [
                 'My', 'research', 'is', 'about', 'A.', 'thaliana', 'protein',
                 '5'
             ]],
            "ner": [[[0, 1, "Greeting"], [6, 6, "Person"]],
                    [[12, 13, "Person"], [14, 15, "Biological_entity"]]]
        }
        self.doc2_gold = {
            "doc_key": "doc2",
            "sentences": [['Hello']],
            "ner": [[]]
        }

        self.doc1_pred_perf = {
            "doc_key":
            "doc1",
            "sentences":
            [['Hello', 'world', ',', 'my', 'name', 'is', 'Sparty', '.'],
             [
                 'My', 'research', 'is', 'about', 'A.', 'thaliana', 'protein',
                 '5'
             ]],
            "predicted_ner": [[[0, 1, "Greeting"], [6, 6, "Person"]],
                              [[12, 13, "Person"], [14, 15, "Biological_entity"]]]
        }
        self.doc1_perf_mismatch_rows = {'doc_key':['doc1', 'doc1', 'doc1',
           'doc1'], 'mismatch_type':[0,0,0,0], 'sent_num':[0,0,1,1],
           'ent_list':[[0, 1, "Greeting"], [6, 6, "Person"], [12, 13, "Person"],
               [14, 15, "Biological_entity"]], 'ent_type':["Greeting", "Person", "Person",
               "Biological_entity"]}
        self.doc1_perf_dict = {'tp': 4, 'fp':0, 'fn':0}

        self.doc2_pred_perf = {
            "doc_key": "doc2",
            "sentences": [['Hello']],
            "predicted_ner": [[]]
        }
        self.doc2_perf_dict = {'tp': 0, 'fp':0, 'fn':0}

        # Use doc2 to test the case where save_mismatch isn't specified
        self.doc2_mismatch = {}

        self.doc1_pred_imperf = {
            "doc_key":
            "doc1",
            "sentences":
            [['Hello', 'world', ',', 'my', 'name', 'is', 'Sparty', '.'],
             [
                 'My', 'research', 'is', 'about', 'A.', 'thaliana', 'protein',
                 '5'
             ]],
            "predicted_ner": [[[0, 1, "Greeting"]],
                              [[9, 9, "Greeting"], [13, 13, "Person"],
                               [14, 15, "Protein"]]]
        }
        self.doc1_imperf_mismatch_rows = {'doc_key':['doc1', 'doc1', 'doc1', 'doc1'],
            'mismatch_type':[1,0,0,0], 'sent_num':[0,0,1,1],
            'sent_num':[0,0,1,1],
            'ent_list':[[0, 1, "Greeting"], [6, 6, "Person"],
               [14, 15, "Person"], [12, 13, "Biological_entity"]],
            'ent_type':["Greeting", "Person", "Person",
               "Biological_entity"]}

        self.doc1_imperf_dict = {'tp':1, 'fp':3, 'fn':3}

        self.doc2_pred_imperf = {
            "doc_key": "doc2",
            "sentences": [['Hello']],
            "predicted_ner": [[[0, 1, "ORG"]]]
        }
        self.doc2_imperf_dict = {'tp':0, 'fp':1, 'fn':0}

    def test_get_doc_ent_counts_doc1_perf(self):
        counts, mismatch_rows = emo.get_doc_ent_counts(self.doc1_pred_perf, self.doc1_gold,
                {'tp':0, 'fp':0, 'fn':0}, self.mismatch_rows_col_input, check_types=True)

        assert counts == self.doc1_perf_dict

    def test_get_doc_ent_counts_doc1_perf_mismatch_rows(self):
        counts, mismatch_rows = emo.get_doc_ent_counts(self.doc1_pred_perf, self.doc1_gold,
                {'tp':0, 'fp':0, 'fn':0}, self.mismatch_rows_col_input, check_types=True)

        assert mismatch_rows == self.doc1_perf_mismatch_rows

    def test_get_doc_ent_counts_doc2_perf(self):
        counts, mismatch_rows = emo.get_doc_ent_counts(self.doc2_pred_perf, self.doc2_gold,
                {'tp':0, 'fp':0, 'fn':0}, self.doc2_mismatch, check_types=True)

        assert counts == self.doc2_perf_dict

    def test_get_doc_ent_counts_doc2_perf_mismatch(self):
        counts, mismatch_rows = emo.get_doc_ent_counts(self.doc2_pred_perf, self.doc2_gold,
                {'tp':0, 'fp':0, 'fn':0}, self.doc2_mismatch, check_types=True)

        assert mismatch_rows == self.doc2_mismatch

    def test_get_doc_ent_counts_doc1_imperf(self):
        counts, mismatch_rows = emo.get_doc_ent_counts(self.doc1_pred_imperf, self.doc1_gold,
                {'tp':0, 'fp':0, 'fn':0}, self.mismatch_rows_col_input, check_types=True)

        assert counts == self.doc1_imperf_dict

    def test_get_doc_ent_counts_doc1_imperf_mismatch(self):
        counts, mismatch_rows = emo.get_doc_ent_counts(self.doc1_pred_imperf, self.doc1_gold,
                {'tp':0, 'fp':0, 'fn':0}, self.mismatch_rows_col_input, check_types=True)

        assert mismatch_rows == self.doc1_imperf_mismatch_rows

    def test_get_doc_ent_counts_doc2_imperf(self):
        counts, mismatch_rows = emo.get_doc_ent_counts(self.doc2_pred_imperf, self.doc2_gold,
                {'tp':0, 'fp':0, 'fn':0}, self.doc2_mismatch, check_types=True)

        assert counts == self.doc2_imperf_dict

    def test_get_doc_ent_counts_doc2_imperf_mismatch(self):
        counts, mismatch_rows = emo.get_doc_ent_counts(self.doc2_pred_imperf, self.doc2_gold,
                {'tp':0, 'fp':0, 'fn':0}, self.doc2_mismatch, check_types=True)

        assert mismatch_rows == self.doc2_mismatch

class TestEliminateRelDups:
    def setup_method(self):

        self.doc_key = 'doc_1'
        self.sent_type = 'predictions'

        # Note: have to compare outputs as sets, because the internal
        # conversion to set destroys the order

        # No duplicates
        self.no_dup = [[1, 3, 5, 6, "hello-world"],
                        [3, 4, 5, 6, "good-morning-world"]]
        self.no_result = set([(1, 3, 5, 6, "hello-world"),
                        (3, 4, 5, 6, "good-morning-world")])

        # All duplicated
        self.all_dup = [[1, 3, 5, 6, "hello-world"],
                [1, 3, 5, 6, "hello-world"],
                [3, 4, 5, 6, "good-morning-world"],
                [3, 4, 5, 6, "good-morning-world"]]
        self.all_result = set([(1, 3, 5, 6, "hello-world"),
                (3, 4, 5, 6, "good-morning-world")])

        # Only one duplicated
        self.one_dup = [[1, 3, 5, 6, "hello-world"],
                [1, 3, 5, 6, "hello-world"],
                [3, 4, 5, 6, "good-morning-world"]]
        self.one_result = set([(1, 3, 5, 6, "hello-world"),
                (3, 4, 5, 6, "good-morning-world")])

        # Same rel but different type
        self.diff_dup = [[1, 3, 5, 6, "hello-world"],
                [1, 3, 5, 6, "goodybe-world"],
                [3, 4, 5, 6, "good-morning-world"]]
        self.diff_result = set([(1, 3, 5, 6, "hello-world"),
                (1, 3, 5, 6, "goodybe-world"),
                (3, 4, 5, 6, "good-morning-world")])

    def test_eliminate_rel_dups_no_dups(self):
        result = emo.eliminate_rel_dups(self.no_dup, self.doc_key,
                self.sent_type)

        result = set([tuple(i) for i in result])

        assert result == self.no_result

    def test_eliminate_rel_dups_all_dup(self):
        result = emo.eliminate_rel_dups(self.all_dup, self.doc_key,
                self.sent_type)

        result = set([tuple(i) for i in result])

        assert result == self.all_result

    def test_eliminate_rel_dups_one_dup(self):
        result = emo.eliminate_rel_dups(self.one_dup, self.doc_key,
                self.sent_type)

        result = set([tuple(i) for i in result])

        assert result == self.one_result

    def test_eliminate_rel_dups_diff_dup(self):
        result = emo.eliminate_rel_dups(self.diff_dup, self.doc_key,
                self.sent_type)

        result = set([tuple(i) for i in result])

        assert result == self.diff_result


class TestCheckRelMatches:
    def setup_method(self):

        # No matches
        self.inc_pred = [1, 3, 5, 6, "hello-world"]
        self.inc_gold = [[1, 2, 6, 7, "goodbye-world"],
                [3, 4, 5, 6, "good-morning-world"],
                [1, 3, 8, 9, "good-afternoon-world"]]
        self.inc_result = False

        # Matches all in correct order
        self.corr_pred = [1, 3, 5, 6, "hello-world"]
        self.corr_out_gold = [[1, 3, 5, 6, "goodbye-world"],
                # Should ignore type
                [3, 4, 5, 6, "good-morning-world"],
                [1, 3, 8, 9, "good-afternoon-world"]]
        self.corr_result = True

        # Matches out of order
        ## TODO make sure this is actually what we want
        self.out_pred = [5, 6, 1, 3, "hello-world"]
        self.out_result = True

    def test_check_rel_matches_no_matches(self):
        result = emo.check_rel_matches(self.inc_pred, self.inc_gold)

        assert result == self.inc_result

    def test_check_rel_matches_ordered_match(self):
        result = emo.check_rel_matches(self.corr_pred, self.corr_out_gold)

        assert result == self.corr_result

    def test_check_rel_matches_out_of_order(self):
        result = emo.check_rel_matches(self.out_pred, self.corr_out_gold)

        assert result == self.out_result


class TestGetDocRelCounts:
    def setup_method(self):

        self.gold = {
            "doc_key":
            "doc1",
            "sentences":
            [['Hello', 'world', ',', 'my', 'name', 'is', 'Sparty', '.'],
             [
                 'My', 'research', 'is', 'about', 'A.', 'thaliana', 'protein',
                 '5'
             ]],
            "ner": [[[0, 1, "ENTITY"], [6, 6, "ENTITY"]],
                    [[12, 13, "ENTITY"], [14, 15, "ENTITY"]]],
            "relations": [[], [[12, 13, 14, 15, "Research"]]]}


        self.doc_pred_perf = {
            "doc_key":
            "doc1",
            "sentences":
            [['Hello', 'world', ',', 'my', 'name', 'is', 'Sparty', '.'],
             [
                 'My', 'research', 'is', 'about', 'A.', 'thaliana', 'protein',
                 '5'
             ]],
            "predicted_ner": [[[0, 1, "Hello"], [6, 6, "Person"]],
                              [[12, 13, "Person"], [14, 15, "Protein"]]],
            "predicted_relations": [[], [[12, 13, 14, 15, "Random-type"]]]
            }
        self.doc_perf_dict = {'tp':1, 'fp':0, 'fn':0}

        self.doc_pred_imperf = {
            "doc_key":
            "doc1",
            "sentences":
            [['Hello', 'world', ',', 'my', 'name', 'is', 'Sparty', '.'],
             [
                 'My', 'research', 'is', 'about', 'A.', 'thaliana', 'protein',
                 '5'
             ]],
            "predicted_ner": [[[0, 1, "Hello"], [6, 6, "Person"]],
                              [[12, 13, "Person"], [14, 15, "Protein"]]],
            "predicted_relations": [[[0, 1, 6, 6, "Random-type:"]],
                [[12, 13, 14, 15, "Random-type"]]]
            }
        self.doc_imperf_dict = {'tp':1, 'fp':1, 'fn':0}

        self.doc_pred_none  = {
            "doc_key":
            "doc1",
            "sentences":
            [['Hello', 'world', ',', 'my', 'name', 'is', 'Sparty', '.'],
             [
                 'My', 'research', 'is', 'about', 'A.', 'thaliana', 'protein',
                 '5'
             ]],
            "predicted_ner": [[[0, 1, "Hello"], [6, 6, "Person"]],
                              [[12, 13, "Person"], [14, 15, "Protein"]]],
            "predicted_relations": [[], []]
            }
        self.doc_none_dict = {'tp':0, 'fp':0, 'fn':1}


    def test_get_doc_rel_counts_perfect(self):
        counts = emo.get_doc_rel_counts(self.doc_pred_perf, self.gold,
                {'tp':0, 'fp':0, 'fn':0}, self.doc_pred_perf["doc_key"])

        assert counts == self.doc_perf_dict

    def test_get_doc_rel_counts_imperf(self):
        counts = emo.get_doc_rel_counts(self.doc_pred_imperf, self.gold,
                {'tp':0, 'fp':0, 'fn':0}, self.doc_pred_imperf["doc_key"])

        assert counts == self.doc_imperf_dict

    def test_get_doc_rel_counts_none(self):
        counts = emo.get_doc_rel_counts(self.doc_pred_none, self.gold,
                {'tp':0, 'fp':0, 'fn':0}, self.doc_pred_none["doc_key"])

        assert counts == self.doc_none_dict


class TestGetF1Input:
    maxDiff = None
    def setup_method(self):

        self.mismatch_rows_col_input = {'doc_key':[], 'mismatch_type':[],
                'sent_num':[], 'ent_list':[], 'ent_type':[]}

        self.gold_std = [{
            "doc_key":
            "doc1",
            "sentences":
            [['Hello', 'world', ',', 'my', 'name', 'is', 'Sparty', '.'],
             [
                 'My', 'research', 'is', 'about', 'A.', 'thaliana', 'protein',
                 '5'
             ]],
            "ner": [[[0, 1, "ENTITY"], [6, 6, "ENTITY"]],
                    [[12, 13, "ENTITY"], [14, 15, "ENTITY"]]],
            "relations": [[], [[12, 13, 14, 15, "Research"]]]
        }, {
            "doc_key": "doc2",
            "sentences": [['Hello']],
            "ner": [[]],
            "relations": [[]]
        }]

        self.pred_perf = [{
            "doc_key":
            "doc1",
            "sentences":
            [['Hello', 'world', ',', 'my', 'name', 'is', 'Sparty', '.'],
             [
                 'My', 'research', 'is', 'about', 'A.', 'thaliana', 'protein',
                 '5'
             ]],
            "predicted_ner": [[[0, 1, "Hello"], [6, 6, "Person"]],
                              [[12, 13, "Person"], [14, 15, "Protein"]]],
            "predicted_relations": [[], [[12, 13, 14, 15, "Random-type"]]]
            }, {
            "doc_key": "doc2",
            "sentences": [['Hello']],
            "predicted_ner": [[]],
            "predicted_relations": [[]]
        }]

        self.perf_mismatch_rows = {'doc_key':['doc1', 'doc1', 'doc1',
           'doc1'], 'mismatch_type':[1,1,1,1], 'sent_num':[0,0,1,1],
           'ent_list':[[0, 1, "ENTITY"], [6, 6, "ENTITY"], [12, 13, "ENTITY"],
               [14, 15, "ENTITY"]], 'ent_type':["ENTITY", "ENTITY", "ENTITY",
               "ENTITY"]}
        self.perf_pred_num_ent = 4
        self.perf_gold_num_ent = 4
        self.perf_matched_num_ent = 4
        self.perf_pred_num_rel = 1
        self.perf_gold_num_rel = 1
        self.perf_matched_num_rel = 1

        self.pred_imperf = [{
            "doc_key":
            "doc1",
            "sentences":
            [['Hello', 'world', ',', 'my', 'name', 'is', 'Sparty', '.'],
             [
                 'My', 'research', 'is', 'about', 'A.', 'thaliana', 'protein',
                 '5'
             ]],
            "predicted_ner": [[[0, 1, "Hello"], [6, 6, "Person"]],
                              [[12, 13, "Person"], [14, 14, "Protein"]]],
                            # One incorrect ent pred
            "predicted_relations": [[[0, 1, 6, 6, "Random-type:"]],
                [[12, 13, 14, 15, "Random-type"]]]
                # One incorrect rel pred
            }, {
            "doc_key": "doc2",
            "sentences": [['Hello']],
            "predicted_ner": [[[1, 1, "Hello"]]],
            # One incorrect ent pred
            "predicted_relations": [[]]
        }]

        self.imperf_mismatch_rows = {'doc_key':['doc1', 'doc1', 'doc1', 'doc1'],
            'mismatch_type':[1,1,1,0], 'sent_num':[0,0,1,1],
            'ent_list':[[0, 1, "ENTITY"], [6, 6, "ENTITY"], [12, 13, "ENTITY"],
               [14, 15, "ENTITY"]], 'ent_type':["ENTITY", "ENTITY", "ENTITY",
               "ENTITY"]}
        self.imperf_pred_num_ent = 5
        self.imperf_gold_num_ent = 4
        self.imperf_matched_num_ent = 3
        self.imperf_pred_num_rel = 2
        self.imperf_gold_num_rel = 1
        self.imperf_matched_num_rel = 1

    def test_get_f1_input_perfect_predicted_ent(self):

        predicted, gold, matched, mismatch_rows = emo.get_f1_input(self.gold_std,
                    self.pred_perf, 'ent', self.mismatch_rows_col_input)

        assert predicted == self.perf_pred_num_ent

    def test_get_f1_input_perfect_gold_ent(self):

        predicted, gold, matched, mismatch_rows = emo.get_f1_input(self.gold_std,
                    self.pred_perf, 'ent', self.mismatch_rows_col_input)

        assert gold == self.perf_gold_num_ent

    def test_get_f1_input_perfect_matched_ent(self):

        predicted, gold, matched, mismatch_rows = emo.get_f1_input(self.gold_std,
                       self.pred_perf, 'ent', self.mismatch_rows_col_input)

        assert matched == self.perf_matched_num_ent

    def test_get_f1_input_perfect_mismatch(self):

        predicted, gold, matched, mismatch_rows = emo.get_f1_input(self.gold_std,
                    self.pred_perf, 'ent', self.mismatch_rows_col_input)

        assert mismatch_rows == self.perf_mismatch_rows

    def test_get_f1_input_imperfect_predicted_ent(self):

        predicted, gold, matched, mismatch_rows = emo.get_f1_input(self.gold_std,
                     self.pred_imperf, 'ent', self.mismatch_rows_col_input)

        assert predicted == self.imperf_pred_num_ent

    def test_get_f1_input_imperfect_gold_ent(self):

        predicted, gold, matched, mismatch_rows = emo.get_f1_input(self.gold_std,
                    self.pred_imperf, 'ent', self.mismatch_rows_col_input)

        assert gold == self.imperf_gold_num_ent

    def test_get_f1_input_imperfect_matched_ent(self):

        predicted, gold, matched, mismatch_rows = emo.get_f1_input(self.gold_std,
                    self.pred_imperf, 'ent', self.mismatch_rows_col_input)

        assert matched == self.imperf_matched_num_ent

    def test_get_f1_input_imperfect_mismatch(self):

        predicted, gold, matched, mismatch_rows = emo.get_f1_input(self.gold_std,
                    self.pred_imperf, 'ent', self.mismatch_rows_col_input)

        assert mismatch_rows == self.imperf_mismatch_rows

    def test_get_f1_input_perfect_predicted_rel(self):

        predicted, gold, matched, mismatch_rows = emo.get_f1_input(self.gold_std,
                                                    self.pred_perf, 'rel')

        assert predicted == self.perf_pred_num_rel

    def test_get_f1_input_perfect_gold_rel(self):

        predicted, gold, matched, mismatch_rows = emo.get_f1_input(self.gold_std,
                                                    self.pred_perf, 'rel')

        assert gold == self.perf_gold_num_rel

    def test_get_f1_input_perfect_matched_rel(self):

        predicted, gold, matched, mismatch_rows = emo.get_f1_input(self.gold_std,
                                                    self.pred_perf, 'rel')

        assert matched == self.perf_matched_num_rel

    def test_get_f1_input_imperfect_predicted_rel(self):

        predicted, gold, matchedi, mismatch_rows = emo.get_f1_input(self.gold_std,
                                                    self.pred_imperf, 'rel')

        assert predicted == self.imperf_pred_num_rel

    def test_get_f1_input_imperfect_gold_rel(self):

        predicted, gold, matched, mismatch_rows = emo.get_f1_input(self.gold_std,
                                                    self.pred_imperf, 'rel')

        assert gold == self.imperf_gold_num_rel

    def test_get_f1_input_imperfect_matched_rel(self):

        predicted, gold, matched, mismatch_rows = emo.get_f1_input(self.gold_std,
                                                    self.pred_imperf, 'rel')

        assert matched == self.imperf_matched_num_rel
