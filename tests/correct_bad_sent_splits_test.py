"""
Spot checks for correct_bad_sent_splits.py.

Author: Serena G. Lotreck
"""
import pytest
import sys
sys.path.append('../annotation/corpus_size_analysis/')
import correct_bad_sent_splits as csp


@pytest.fixture
def bad_split():
    return {"doc_key": "BioInfer.d70",
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
        
@pytest.fixture
def corrected_bad_split():
    return {"doc_key": "BioInfer.d70",
    "dataset": "bioinfer",
    "sentences": [["Aprotinin", "inhibited", "platelet", "aggregation", "induced",
            "by", "thrombin", "(", "0.25", "U.ml-1", ")", "with", "IC50", "200",
            "kIU.ml-1", ",", "and", "inhibited", "the", "rise", "of", "cytosolic",
            "free", "calcium", "concentration", "in", "platelets", "stimulated", "by",
            "thrombin", "(", "0.1", "U.ml-1", ")", "in", "the", "absence", "and", "in",
            "the", "presence", "of", "Ca2", "+", "0.5", "mmol", ".", "L-1",
            "(","IC50", "117", "and", "50", "kIU.ml-1", ",", "respectively",
            ")", ",", "but", "had", "no", "effect", "on", "the", "amounts", "of",
            "actin", "and", "myosin", "heavy", "chain", "associated", "with",
            "cytoskeletons", "."]],
    "ner": [[[29, 29, "Individual_protein"], [0, 0, "Individual_protein"],
        [6, 6, "Individual_protein"], [68, 70, "Individual_protein"],
        [66, 66, "Individual_protein"]]],
    "relations": [[[29, 29, 0, 0, "PPI"], [0, 0, 66, 66, "PPI"],
        [68, 70, 0, 0, "PPI"]]]}
        
@pytest.fixture
def normal_split():
    return {"doc_key": "BioInfer.d70",
    "dataset": "bioinfer",
    "sentences": [["Aprotinin", "inhibited", "platelet", "aggregation", "induced",
            "by", "thrombin", "(", "0.25", "U.ml-1", ")", "with", "IC50", "200",
            "kIU.ml-1", ",", "and", "inhibited", "the", "rise", "of", "cytosolic",
            "free", "calcium", "concentration", "in", "platelets", "stimulated", "by",
            "thrombin", "(", "0.1", "U.ml-1", ")", "in", "the", "absence", "and", "in",
            "the", "presence", "of", "Ca2", "+", "0.5", "mmol", ".", "L-1",
            "(","IC50", "117", "and", "50", "kIU.ml-1", ",", "respectively",
            ")", ",", "but", "had", "no", "effect", "on", "the", "amounts", "of",
            "actin", "and", "myosin", "heavy", "chain", "associated", "with",
            "cytoskeletons", "."]],
    "ner": [[[29, 29, "Individual_protein"], [0, 0, "Individual_protein"],
        [6, 6, "Individual_protein"], [68, 70, "Individual_protein"],
        [66, 66, "Individual_protein"]]],
    "relations": [[[29, 29, 0, 0, "PPI"], [0, 0, 66, 66, "PPI"],
        [68, 70, 0, 0, "PPI"]]]}
        

def test_check_correct_doc_bad_split(bad_split, corrected_bad_split):
    
    fixed = csp.check_correct_doc(bad_split)
    
    assert fixed == corrected_bad_split
    
def test_check_correct_doc_normal_split(normal_split):
    
    fixed = csp.check_correct_doc(normal_split)
    
    assert fixed == normal_split
