"""
Spot checks for map_dataset_types.py

Currently only contains checks for josnl formatted mapping

Author: Serena G. Lotreck
"""
import pytest
import sys

sys.path.append('../annotation/abstract_scripts')
import map_dataset_types as mdt


@pytest.fixture
def entity_map():
    return {
        "CHEMICAL": "Organic_compound_other",
        "GENE": "DNA",
        "GENE-Y": "DNA",
        "GENE-N": "DNA",
        "UNDEFINED": ""  # Added this so I can test entity drops
    }


@pytest.fixture
def relation_map():
    return {
        "PART-OF": "is-in",
        "REGULATOR": "interacts",
        "DIRECT-REGULATOR": "interacts",
        "INDIRECT-REGULATOR": "interacts",
        "UPREGULATOR": "activates",
        "ACTIVATOR": "activates",
        "INDIRECT-UPREGULATOR": "activates",
        "DOWNREGULATOR": "inhibits",
        "INHIBITOR": "inhibits",
        "INDIRECT-DOWNREGULATOR": "inhibits",
        "AGONIST": "activates",
        "AGONIST-ACTIVATOR": "activates",
        "AGONIST-INHIBITOR": "inhibits",
        "ANTAGONIST": "inhibits",
        "MODULATOR": "interacts",
        "MODULATOR-ACTIVATOR": "activates",
        "MODULATOR-INHIBITOR": "inhibits",
        "COFACTOR": "interacts",
        "SUBSTRATE": "interacts",
        "PRODUCT-OF": "interacts",
        "SUBSTRATE_PRODUCT-OF": "interacts",
        "NOT": "",
        "UNDEFINED": ""
    }


# Character indices are wrong because I abbreviated the docs, but they just
# have to be consistent between entities and relations, not sentences
@pytest.fixture
def dygiepp_needs_rel_only_drops():
    return [{
        "doc_key":
        23551063,
        "dataset":
        "chemprot",
        "sentences":
        [[
            "The", "activities", "of", "UGTs", "1A3", ",", "1A8", ",", "1A9",
            ",", "2B4", "and", "2B7", "were", "low", ",", "whereas", "UGT1A1",
            "and", "UGT2B17", "exhibited", "no", "HFC", "glucuronidation",
            "activity", "."
        ],
         [
             "UGT1A6", "exhibited", "a", "significantly", "higher", "Vmax",
             "and", "Km", "values", "toward", "both", "HFC", "and",
             "UDP-glucuronic", "acid", "than", "the", "other", "UGTs", "."
         ]],
        "ner":
        [[[183, 183, "CHEMICAL"], [164, 173, "GENE"], [178, 178, "GENE"],
          [180, 180, "GENE"]],
         [[198, 198, "CHEMICAL"], [205, 205, "GENE"], [187, 187, "GENE"]]],
        "relations": [[[183, 183, 178, 178, "NOT"],
                       [183, 183, 180, 180, "NOT"]],
                      [[198, 198, 205, 205, "SUBSTRATE"],
                       [198, 198, 187, 187, "SUBSTRATE"]]]
    }]


@pytest.fixture
def dygiepp_needs_rel_only_drops_result():
    return [{
        "doc_key":
        23551063,
        "dataset":
        "chemprot",
        "sentences":
        [[
            "The", "activities", "of", "UGTs", "1A3", ",", "1A8", ",", "1A9",
            ",", "2B4", "and", "2B7", "were", "low", ",", "whereas", "UGT1A1",
            "and", "UGT2B17", "exhibited", "no", "HFC", "glucuronidation",
            "activity", "."
        ],
         [
             "UGT1A6", "exhibited", "a", "significantly", "higher", "Vmax",
             "and", "Km", "values", "toward", "both", "HFC", "and",
             "UDP-glucuronic", "acid", "than", "the", "other", "UGTs", "."
         ]],
        "ner": [[[183, 183, "Organic_compound_other"], [164, 173, "DNA"],
                 [178, 178, "DNA"], [180, 180, "DNA"]],
                [[198, 198, "Organic_compound_other"], [205, 205, "DNA"],
                 [187, 187, "DNA"]]],
        "relations": [[],
                      [[198, 198, 205, 205, "interacts"],
                       [198, 198, 187, 187, "interacts"]]]
    }]


@pytest.fixture
def dygiepp_needs_entity_and_rel_drops():
    return [{
        "doc_key":
        23551063,
        "dataset":
        "chemprot",
        "sentences": [
            [
                "The", "activities", "of", "UGTs", "1A3", ",", "1A8", ",",
                "1A9", ",", "2B4", "and", "2B7", "were", "low", ",", "whereas",
                "UGT1A1", "and", "UGT2B17", "exhibited", "no", "HFC",
                "glucuronidation", "activity", "."
            ],
            [
                "UGT1A6", "exhibited", "a", "significantly", "higher", "Vmax",
                "and", "Km", "values", "toward", "both", "HFC", "and",
                "UDP-glucuronic", "acid", "than", "the", "other", "UGTs", "."
            ],
        ],
        "ner":
        [[[183, 183, "CHEMICAL"], [164, 173, "GENE"], [178, 178, "GENE"],
          [180, 180, "GENE"]],
         [[198, 198, "UNDEFINED"], [205, 205, "GENE"],
          [187, 187, "UNDEFINED"]]],
        "relations": [[[183, 183, 178, 178, "NOT"],
                       [183, 183, 180, 180, "NOT"]],
                      [[198, 198, 205, 205, "SUBSTRATE"],
                       [198, 198, 187, 187, "SUBSTRATE"]]]
    }]


@pytest.fixture
def dygiepp_needs_entity_and_rel_drops_result():
    return [{
        "doc_key":
        23551063,
        "dataset":
        "chemprot",
        "sentences": [
            [
                "The", "activities", "of", "UGTs", "1A3", ",", "1A8", ",",
                "1A9", ",", "2B4", "and", "2B7", "were", "low", ",", "whereas",
                "UGT1A1", "and", "UGT2B17", "exhibited", "no", "HFC",
                "glucuronidation", "activity", "."
            ],
            [
                "UGT1A6", "exhibited", "a", "significantly", "higher", "Vmax",
                "and", "Km", "values", "toward", "both", "HFC", "and",
                "UDP-glucuronic", "acid", "than", "the", "other", "UGTs", "."
            ],
        ],
        "ner": [[[183, 183, "Organic_compound_other"], [164, 173, "DNA"],
                 [178, 178, "DNA"], [180, 180, "DNA"]], [[205, 205, "DNA"]]],
        "relations": [[], []]
    }]


def test_needs_rel_only_drops(entity_map, relation_map,
                              dygiepp_needs_rel_only_drops,
                              dygiepp_needs_rel_only_drops_result):

    de, dr = mdt.map_jsonl(dygiepp_needs_rel_only_drops,
                           entity_map,
                           relation_map,
                           predicted=False)

    assert dygiepp_needs_rel_only_drops == dygiepp_needs_rel_only_drops_result
    assert de == 0
    assert dr == 2


def test_needs_entity_and_rel_drops(entity_map, relation_map,
                                    dygiepp_needs_entity_and_rel_drops,
                                    dygiepp_needs_entity_and_rel_drops_result):

    de, dr = mdt.map_jsonl(dygiepp_needs_entity_and_rel_drops,
                           entity_map,
                           relation_map,
                           predicted=False)

    assert dygiepp_needs_entity_and_rel_drops == dygiepp_needs_entity_and_rel_drops_result
    assert de == 2
    assert dr == 4
