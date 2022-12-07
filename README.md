# pickle-corpus-code
Code for the project *In a PICKLE: Entity and relation annotation guidelines for the molecular plant sciences*

## Respository contents
* `annotation`: Contains scripts for annotation-related tasks. These include utility scripts (`abstract_scripts`), scripts to calulate IAA (`iaa`), and the most recent version of the `annotation.conf` files used for annotation in brat (`brat`)
* `data_retreival`: Contains scripts for obtaining raw text data. `abstracts_only` contains scripts to download abstracts from PubMed, and `doc_clustering` contains the scripts to choose from the downloaded abstracts for downstream use. More information on the document clustering pipeline can be found in [doc_clustering.md](https://github.com/serenalotreck/pickle-corpus-code/tree/master/data_retrieval/doc_clustering/doc_clustering.md)
* `jupyter_notebooks`: Jupyter notebooks with code to produce the data visualizations included in the manuscript
* `models`: Contains code to run models (`neural_models`), as well as a script to evaluate performance model-agnostically. 
* `tests`: Unit tests 


