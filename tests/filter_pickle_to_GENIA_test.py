"""
Spot checks for filter_pickle_to_GENIA.py

Author: Serena G. Lotreck
"""
from os.path import splitext
import sys

sys.path.append('../annotation/abstract_scripts/')
from filter_pickle_to_GENIA import main
import pytest
import json
import jsonlines


@pytest.fixture
def type_map():
    return {
        "DNA": "DNA",
        "RNA": "RNA",
        "protein": "Protein",
        "cell_line": "Cell",
        "cell_type": "Cell"
    }


@pytest.fixture
def orig_dataset():
    return {
        "doc_key":
        "PMID16716453_abstract",
        "dataset":
        "pickle",
        "sentences":
        [[
            "Calcium-dependent", "protein", "kinases", "(", "CPKs", ")",
            "play", "important", "roles", "in", "multiple", "signal",
            "transduction", "pathways", "but", "the", "precise", "role", "of",
            "individual", "CPK", "is", "largely", "unknown", "."
        ],
         [
             "We", "isolated", "two", "cDNAs", "encoding", "two", "CPK",
             "isoforms", "(", "Cicer", "arietinum", "CPKs-CaCPK1", "and",
             "CaCPK2", ")", "of", "chickpea", "."
         ],
         [
             "Their", "expression", "in", "various", "organs", "and", "in",
             "response", "to", "various", "phytohormones", ",", "and",
             "dehydration", ",", "high", "salt", "stress", "and", "fungal",
             "spore", "in", "excised", "leaves", "as", "well", "as",
             "localization", "in", "leaf", "and", "stem", "tissues", "were",
             "analyzed", "in", "this", "study", "."
         ],
         [
             "CaCPK1", "protein", "and", "its", "activity", "were",
             "ubiquitous", "in", "all", "tissues", "examined", "."
         ],
         [
             "In", "contrast", ",", "CaCPK2", "transcript", ",", "CaCPK2",
             "protein", "and", "its", "activity", "were", "almost",
             "undetectable", "in", "flowers", "and", "fruits", "."
         ],
         [
             "Both", "CaCPK1", "and", "CaCPK2", "transcripts", "and",
             "proteins", "were", "abundant", "in", "roots", "but", "in",
             "minor", "quantities", "in", "leaves", "and", "stems", "."
         ],
         [
             "Of", "the", "three", "phytohormones", "tested", ",", "viz", ".",
             "indole-3-acetic", "acid", "(", "IAA", ")", ",", "gibberellin",
             "(", "GA(3", ")", ")", "and", "benzyladenine", "(", "BA", ")",
             ",", "only", "BA", "increased", "both", "CaCPK1", "and", "CaCPK2",
             "transcripts", ",", "proteins", "and", "their", "activities", "."
         ],
         [
             "GA(3", ")", "induced", "accumulation", "of", "CaCPK2",
             "transcript", "and", "protein", "but", "CaCPK1", "remained",
             "unaffected", "."
         ],
         [
             "The", "expression", "of", "CaCPK1", "and", "CaCPK2", "in",
             "leaves", "was", "enhanced", "in", "response", "to", "high",
             "salt", "stress", "."
         ],
         [
             "Treatments", "with", "Aspergillus", "sp.", "spores", "increased",
             "expression", "of", "CaCPK1", "in", "chickpea", "leaf", "tissue",
             "but", "had", "no", "effect", "on", "CaCPK2", "."
         ],
         [
             "Excised", "leaves", "subjected", "to", "dehydration", "showed",
             "increase", "in", "CaCPK2", "expression", "but", "not", "in",
             "CaCPK1", "."
         ],
         [
             "Both", "isoforms", "were", "located", "in", "the", "plasma",
             "membrane", "(", "PM", ")", "and", "chloroplast", "membrane",
             "of", "leaf", "mesophyll", "cells", "as", "well", "as", "in",
             "the", "PM", "of", "stem", "xylem", "parenchyma", "cells", "."
         ],
         [
             "These", "results", "suggest", "specific", "roles", "for",
             "CaCPK", "isoforms", "in", "phytohormone/defense/stress",
             "signaling", "pathways", "."
         ]],
        "ner": [[[0, 2, "Protein"], [4, 4, "Protein"], [20, 20, "Protein"]],
                [[31, 32, "Protein"], [34, 38, "Multicellular_organism"],
                 [41, 41, "Multicellular_organism"]],
                [[53, 53, "Plant_hormone"],
                 [59, 59, "Inorganic_compound_other"]], [[82, 83, "Protein"]],
                [[97, 98, "RNA"], [100, 101, "Protein"]],
                [[114, 119, "Organic_compound_other"]],
                [[136, 136, "Plant_hormone"], [141, 142, "Plant_hormone"],
                 [144, 144, "Plant_hormone"], [147, 147, "Plant_hormone"],
                 [149, 150, "Plant_hormone"], [153, 153, "Plant_hormone"],
                 [155, 155, "Plant_hormone"], [159, 159, "Plant_hormone"],
                 [162, 165, "RNA"]],
                [[172, 173, "Plant_hormone"],
                 [177, 180, "Organic_compound_other"], [182, 182, "Protein"]],
                [[189, 189, "DNA"], [191, 191, "DNA"],
                 [200, 200, "Inorganic_compound_other"]],
                [[205, 207, "Cell"], [211, 211, "DNA"], [213, 215, "Tissue"],
                 [221, 221, "DNA"]],
                [[231, 232, "Biochemical_process"], [236, 236, "DNA"]], [],
                [[274, 275, "Protein"], [277, 279, "Biochemical_pathway"]]],
        "relations": [[], [[34, 38, 41, 41, "is-in"]], [], [], [], [],
                      [[159, 159, 162, 165, "activates"]],
                      [[172, 173, 177, 180, "activates"]],
                      [[200, 200, 191, 191, "activates"],
                       [200, 200, 189, 189, "activates"]],
                      [[205, 207, 211, 211, "activates"],
                       [211, 211, 213, 215, "is-in"]], [], [],
                      [[274, 275, 277, 279, "is-in"]]]
    }


@pytest.fixture
def filtered_dataset():
    return {
        "doc_key":
        "PMID16716453_abstract",
        "dataset":
        "pickle",
        "sentences":
        [[
            "Calcium-dependent", "protein", "kinases", "(", "CPKs", ")",
            "play", "important", "roles", "in", "multiple", "signal",
            "transduction", "pathways", "but", "the", "precise", "role", "of",
            "individual", "CPK", "is", "largely", "unknown", "."
        ],
         [
             "We", "isolated", "two", "cDNAs", "encoding", "two", "CPK",
             "isoforms", "(", "Cicer", "arietinum", "CPKs-CaCPK1", "and",
             "CaCPK2", ")", "of", "chickpea", "."
         ],
         [
             "Their", "expression", "in", "various", "organs", "and", "in",
             "response", "to", "various", "phytohormones", ",", "and",
             "dehydration", ",", "high", "salt", "stress", "and", "fungal",
             "spore", "in", "excised", "leaves", "as", "well", "as",
             "localization", "in", "leaf", "and", "stem", "tissues", "were",
             "analyzed", "in", "this", "study", "."
         ],
         [
             "CaCPK1", "protein", "and", "its", "activity", "were",
             "ubiquitous", "in", "all", "tissues", "examined", "."
         ],
         [
             "In", "contrast", ",", "CaCPK2", "transcript", ",", "CaCPK2",
             "protein", "and", "its", "activity", "were", "almost",
             "undetectable", "in", "flowers", "and", "fruits", "."
         ],
         [
             "Both", "CaCPK1", "and", "CaCPK2", "transcripts", "and",
             "proteins", "were", "abundant", "in", "roots", "but", "in",
             "minor", "quantities", "in", "leaves", "and", "stems", "."
         ],
         [
             "Of", "the", "three", "phytohormones", "tested", ",", "viz", ".",
             "indole-3-acetic", "acid", "(", "IAA", ")", ",", "gibberellin",
             "(", "GA(3", ")", ")", "and", "benzyladenine", "(", "BA", ")",
             ",", "only", "BA", "increased", "both", "CaCPK1", "and", "CaCPK2",
             "transcripts", ",", "proteins", "and", "their", "activities", "."
         ],
         [
             "GA(3", ")", "induced", "accumulation", "of", "CaCPK2",
             "transcript", "and", "protein", "but", "CaCPK1", "remained",
             "unaffected", "."
         ],
         [
             "The", "expression", "of", "CaCPK1", "and", "CaCPK2", "in",
             "leaves", "was", "enhanced", "in", "response", "to", "high",
             "salt", "stress", "."
         ],
         [
             "Treatments", "with", "Aspergillus", "sp.", "spores", "increased",
             "expression", "of", "CaCPK1", "in", "chickpea", "leaf", "tissue",
             "but", "had", "no", "effect", "on", "CaCPK2", "."
         ],
         [
             "Excised", "leaves", "subjected", "to", "dehydration", "showed",
             "increase", "in", "CaCPK2", "expression", "but", "not", "in",
             "CaCPK1", "."
         ],
         [
             "Both", "isoforms", "were", "located", "in", "the", "plasma",
             "membrane", "(", "PM", ")", "and", "chloroplast", "membrane",
             "of", "leaf", "mesophyll", "cells", "as", "well", "as", "in",
             "the", "PM", "of", "stem", "xylem", "parenchyma", "cells", "."
         ],
         [
             "These", "results", "suggest", "specific", "roles", "for",
             "CaCPK", "isoforms", "in", "phytohormone/defense/stress",
             "signaling", "pathways", "."
         ]],
        "ner": [[[0, 2, "Protein"], [4, 4, "Protein"], [20, 20, "Protein"]],
                [
                    [31, 32, "Protein"],
                ], [], [[82, 83, "Protein"]],
                [[97, 98, "RNA"], [100, 101, "Protein"]], [],
                [[162, 165, "RNA"]], [[182, 182, "Protein"]],
                [
                    [189, 189, "DNA"],
                    [191, 191, "DNA"],
                ],
                [[205, 207, "cell_type"], [211, 211, "DNA"], [221, 221,
                                                              "DNA"]],
                [[236, 236, "DNA"]], [], [[274, 275, "Protein"]]],
        "relations": [[], [[34, 38, 41, 41, "is-in"]], [], [], [], [],
                      [[159, 159, 162, 165, "activates"]],
                      [[172, 173, 177, 180, "activates"]],
                      [[200, 200, 191, 191, "activates"],
                       [200, 200, 189, 189, "activates"]],
                      [[205, 207, 211, 211, "activates"],
                       [211, 211, 213, 215, "is-in"]], [], [],
                      [[274, 275, 277, 279, "is-in"]]]
    }


def test_main(tmp_path_factory, type_map, orig_dataset, filtered_dataset):
    d = tmp_path_factory.mktemp("data")
    type_map_path = f'{d}/type_map.json'
    with open(type_map_path, 'w') as myf:
        json.dump(type_map, myf)
    orig_dset_path = f'{d}/original_dataset.jsonl'
    with jsonlines.open(orig_dset_path, 'w') as writer:
        writer.write_all([orig_dataset])
    main(orig_dset_path, type_map_path)
    result_path = splitext(
        orig_dset_path)[0] + '_FILTERED_TO_GENIA' + splitext(orig_dset_path)[1]
    with jsonlines.open(result_path) as reader:
        result = []
        for obj in reader:
            result.append(obj)
    assert result == [filtered_dataset]
