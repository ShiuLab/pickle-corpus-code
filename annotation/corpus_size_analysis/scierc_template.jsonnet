local template = import "template.libsonnet";

template.DyGIE {
  bert_model: "allenai/scibert_scivocab_cased",
  cuda_device: 1,
  data_paths: {
    train: "XXXX",
    validation: "YYYY",
    test: "ZZZZ",
  },
  loss_weights: {
    ner: 0.2,
    relation: 1.0,
    coref: 1.0,
    events: 0.0
  },
  model +: {
    modules +: {
      coref +: {
        coref_prop: 1
      }
    }
  },
  target_task: "relation",
}
