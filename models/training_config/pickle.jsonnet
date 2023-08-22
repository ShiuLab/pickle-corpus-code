local template = import "template.libsonnet";

template.DyGIE {
  bert_model: "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
  cuda_device: 2,
  data_paths: {
    train: "~/Shiu_lab/pickle-corpus-code/data/straying_off_topic_data/unified_annotations_processed/jsonl_files/PICKLE_250_abstracts_entities_and_relations_FINAL_05Jul2023_CORRECTED_TRAIN.jsonl",
    validation: "~/Shiu_lab/pickle-corpus-code/data/straying_off_topic_data/unified_annotations_processed/jsonl_files/PICKLE_250_abstracts_entities_and_relations_FINAL_05Jul2023_CORRECTED_DEV.jsonl",
    test: "~/Shiu_lab/pickle-corpus-code/data/straying_off_topic_data/unified_annotations_processed/jsonl_files/PICKLE_250_abstracts_entities_and_relations_FINAL_05Jul2023_CORRECTED_TEST.jsonl",
  },
  loss_weights: {
    ner: 0.2,
    relation: 1.0,
    coref: 0.0,
    events: 0.0
  },
  target_task: "relation",
  trainer +: {
    num_epochs: 25
  },
}
