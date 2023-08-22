local template = import "template.libsonnet";

template.DyGIE {
  bert_model: "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
  cuda_device: 2,
  data_paths: {
    train: "~/Shiu_lab/pickle-corpus-code/data/straying_off_topic_data/SeeDev/15Jul2023_SeeDev_splits_from_train_dev_combined_TRAIN_SPLIT.jsonl",
    validation: "~/Shiu_lab/pickle-corpus-code/data/straying_off_topic_data/SeeDev/15Jul2023_SeeDev_splits_from_train_dev_combined_DEV_SPLIT.jsonl",
    test: "~/Shiu_lab/pickle-corpus-code/data/straying_off_topic_data/SeeDev/15Jul2023_SeeDev_splits_from_train_dev_combined_TEST_SPLIT.jsonl",
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
