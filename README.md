# pickle-corpus-code
Code for the project [*In a PICKLE: A gold standard entity and relation corpus for the molecular plant sciences*](https://academic.oup.com/insilicoplants/advance-article/doi/10.1093/insilicoplants/diad021/7413143). To cite this project:

```
@article{lotreck2023pickle,
  title={In a PICKLE: A gold standard entity and relation corpus for the molecular plant sciences},
  author={Lotreck, Serena and Segura Ab{\'a}, Kenia and Lehti-Shiu, Melissa and Seeger, Abigail and Brown, Brianna NI and Ranaweera, Thilanka and Schumacher, Ally and Ghassemi, Mohammad and Shiu, Shin-Han},
  journal={in silico Plants},
  pages={diad021},
  year={2023},
  publisher={Oxford University Press UK}
}
```

## Respository table of contents
* `annotation`: Contains scripts for annotation-related tasks. These include utility scripts (`abstract_scripts`), scripts to calulate IAA (`iaa`), and the most recent version of the `annotation.conf` files used for annotation in brat (`brat`)
* `data_retreival`: Contains scripts for obtaining raw text data. `abstracts_only` contains scripts to download abstracts from PubMed, and `doc_clustering` contains the scripts to choose from the downloaded abstracts for downstream use. More information on the document clustering pipeline can be found in [doc_clustering.md](https://github.com/serenalotreck/pickle-corpus-code/tree/master/data_retrieval/doc_clustering/doc_clustering.md)
* `jupyter_notebooks`: Jupyter notebooks with code to produce the data visualizations included in the manuscript
* `models`: Contains code to run models (`neural_models`), as well as a script to evaluate performance model-agnostically. 
* `tests`: Unit tests 

## Reproducing manuscript results

### Software environment
We used two different conda environments in this project; the one specified in the DyGIE++ repo for running model code, as well as our own environment specified by the `requirements.txt` in our repository. To recreate the DyGIE++ environment, clone the DyGIE++ repository (outside of the pickle-corpus-code repository), and follow the [instructions provided in their README](https://github.com/dwadden/dygiepp/tree/master#dependencies). To reproduce ours, from the root directory of this repository, run:

```
conda env create -f environment.yml
```
In all examples below, we are using the `pickle` environment unless otherwise specified by a `conda activate dygiepp` command.

### Dataset pre-processing
We used the data preprocessing code from the DyGIE++ repository to get the SciERC, ChemProt, and GENIA datasets, whihc automatically pulls the data from the web and processes it.

The PICKLE dataset is available from both Zenodo (in brat and jsonl format; link will become available upon publication) and [Huggingface](https://huggingface.co/datasets/slotreck/pickle) (jsonl format). We recommend using the Hugginface version (or the jsonl version from Zenodo), as all of the necessary pre-processing has been completed. However, if you would like to reproduce the preprocessing sets from the original brat dataset, these are the steps:
1. Run [`brat_to_input.py`](https://github.com/dwadden/dygiepp#updates) from the DyGIE++ repository:
```
python brat_to_input.py path/to/brat/data/dir save/path.jsonl dataset_name --use-scispacy
```
We have incorporated a quality checker into `brat_to_input.py`, as scispacy occasionally mis-tokenizes abbreviations, causing incorrect sentence splits that give the impression of relations crossing sentence boundaries, which causes DyGIE++ to break. 
2. Perform the train-dev-split:
```
cd models
python get_dev_test_splits.py /path/to/full/dataset.jsonl path/to/save/output/ output_prefix_string -test_frac 0.2 -dev_frac 0.12
```

The SeeDev dataset is available from [their website](https://sites.google.com/view/seedev2019/dataset) under "SeeDev Binary". The SeeDev test set is maintained as a closed dataset, so annotations are not available; only the Training and Development sets need to be downloaded. Because of the split annotation format (entity and relation annotations are contained in separate brat annotation files for each document) 
as well as the length of some documents, there was extra pre-processing involved to get the dataset into its proper format. To reproduce, after downloading the Training and Development sets, do the following:
1. Combine Training and Development sets into one directory. Since the true test set isn't available, we'll have to make our own train-dev-test spit. We copied the contents of the Training and Development sets into a single combined directory.
2. Combine `.a1` and `.a2` annotation files and change residual Event formatting to correct Relations formatting:
```
cd annotation/abstract_scripts
python seeDev_relation_conversion.py /path/to/seedev/directory/ /path/to/save/updated/output
```
3. Run `brat_to_input.py`. Follows the same format as for PICKLE.
4. Make train-dev-test split. Follows the same format as for PICKLE.
5. Split documents to fit them into memory. Some of the SeeDev docs are long enough that they can't be fit into memory during training. Run the following on each section of the dataset to split them into shorter documents:
```
python split_docs_for_memory.py /path/to/seedev.jsonl
```

### Training DyGIE++ models
The configurations used to train all models used in this manuscript are located in `models/training_config`. This also includes the [verbatim configs provided with the DyGIE++ repo](https://github.com/dwadden/dygiepp/tree/master/training_config). All configs contain local data paths to where we stored the data used in training. In order to use these configs, you need to: 
1. Move them to the `dygiepp/training_config` directory
2. Change the data paths for the train, dev adn test sets to wherever you saved the output from the previous step.

We include the pre-trained configs here for convenience, but we downloaded the pre-trained models directly from the [links provided in the DyGIE++ repo](https://github.com/dwadden/dygiepp/tree/master#available-models). These were: SciERC, SciERC lightweight, GENIA, GENIA lightweight, ChemProt (lightweight only), and ACE05 relation.

To train the models for PICKLE and SeeDev, once the training configs have had the data paths replaced with paths to your local data copies and moved to the `trainng_config` directory, from the root directory of the DyGIE++ repository, run:
```
conda activate dygiepp
bash scripts/train.sh name_of_config_without_extension
```

### Model evaluation
#### Full evaluation
First, we need to apply our trained model to generate predictions on the target test set. The test set provided to the model should have no gold standard annotations in it; we achieved this interactively by reading in the jsonl dataset and removing those keys:
```
import jsonlines
with jsonlines.open('path/to/test.jsonl') as reader:
  data = []
  for obj in reader:
    data.append(obj)
updated_data = []
for doc in data:
  updated_doc = {}
  for key in doc.keys():
    if (key != 'ner') & (key != 'relations'):
      updated_doc[key] = doc[key]
  updated_data.append(updated_doc)
with jsonlines.open('path/to/save/text.jsonl', 'w') as writer:
  writer.write_all(updated_data)
```
To apply the model, we do the following. Note that the output directoy doesn't have to already exist for this to run. All of the models passed to this script must exist either in `dygiepp/models/` or `dygiepp/pretrained/`
```
cd models/neural_models
conda activate dygiepp
python run_dygiepp.py /path/to/output/directory/ prefix_for_save_names /path/to/dygiepp/ /path/to/test/set/with/no/gold/anns.jsonl --no_eval -models_to_run ace05-relation scierc scierc-lightweight genia genia-lightweight chemprot pickle -v
```

Once we've applied the models, we use our own script to perform a bootstrapped evaluation of model performance. The evaluation must be run once for each gold standard that's being compared; for example, GENIA and GENIA lightweight can be cacluated together on the GENIA test set, and all models can be evaluated together on the PICKLE test set. It also must be run separately for an evaluation without types, versus one with types. To run without types:
```
cd models
python evaluate_model_output.py /path/to/gold/standard/test.jsonl  /path/to/save/output.csv /path/to/model/preds/ --bootstrap -use_prefix prefix_that_all_models_to_eval_have_in_common -v
```
To run with types:
```
cd models
python evaluate_model_output.py /path/to/gold/standard/test.jsonl  /path/to/save/output.csv /path/to/model/preds/ --bootstrap -use_prefix prefix_that_all_models_to_eval_have_in_common --check_types -sym_rels any_sym_rel_types -v
```
Note that of the models we used in the major analyses (excluding BioInfer), only the PICKLE corpus has symmetric relations; the specification for PICKLE is `-sym_rels interacts`, and this argument can be excluded for all other models.

#### Filtered evaluation
The filtered evaluation follows the same process as a full evaluation, with the addition of one step before evaluation where the PICKLE testset is filterd based on the types in GENIA. To run this step:
```
cd annotation/abstract_scripts
python filter_pickle_to_GENIA.py /path/tp/pickle/test.jsonl ontology_maps/GENIA_to_PICKLE_entities.json
```

### Other analyses
#### Corpus size analysis
We used the SLURM job submission capabilities of the university HPC in order to run this analysis in a maximally automated way. Therefore, it may take some effort to adapt to your system if SLURM is not available to you. However, if you are able to submit SLURM jobs, all of the templates provided in the `annotation/corpus_size_analysis` should work out-of-the-box.

A note on BioInfer: BioInfer is the only dataset that we hadn't already pulled for the models above. To get the dataset, load it from Huggingface with the following script from `annotation/abstract_scripts`:
```
python huggingface_to_brat.py bigbio/bioinfer document_id /path/to/save/output
```
Then run `brat_to_input.py` to get it in jsonl format.

In order to have enough documents to pull from for the  SciERC, BioInfer and PICKLE dataset, we had to combine their splits before performing the analysis. BioInfer is automatically combined by `huggingface_to_brat.py`, and there is one overall PICKLE jsonl from before we tan the train-dev-split, but we did this interactively for SciERC reading in all the jsonl files for each split, and then appending the lists together and writing out as a single dataset. For ChemProt and GENIA, we just used the trainng datasets, as they contained enough documents.

```
cd annotation/corpus_size_analysis
python analyze_corpus.py /path/to/dataset.jsonl dataset_name /path/to/dygiepp/ config_template_name.jsonnet train_job_template.sb <test_size> <dev_size> <start_train_size> <train_subset_size> <num_train_subsets> /path/to/save/output/
```
`config_template_name.jsonnet` should be replaced with one of `chemprot_bioinfer_pickle_template.jsonnet`, `scierc_template.jsonnet`, or `genia_template.jsonnet` depending on which dataset is being analyzed.

The following settings were used for each of the datasets:

| Dataset  | start_train_size | train_subset_size | num_train_subsets | dev_size | test_size |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| ChemProt  | 10  | 10 | 49 | 50 | 50 |
| SciERC | 10 | 10 | 39 | 50 | 50 |
| GENIA | 10 | 10 | 49 | 50 | 50 |
| Bioinfer | 10 | 10 | 49 | 50 | 50 |
| PICKLE | 10 | 10 | 14 | 50 | 50 |

The results of this analysis can be visualized by replacing the paths in `jupyter_notebooks/corpus_size_analysis.ipynb`.

#### Out-of-vocab analysis
For each comparison dataset, from `models/oov_comparison`, run:
```
python /path/to/pickle/train.jsonl /path.to/comparison/dataset/train.jsonl comparison_dataset_name /path/to/save/output/ output_prefix -v
```
Results can be visualized by replacing the paths in `jupyter_notebooks/out_of_vocab_comparison.ipynb`.
