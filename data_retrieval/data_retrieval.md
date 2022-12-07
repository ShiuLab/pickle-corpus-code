# Data Retreival
This directory contains code for obtaining text data from articles on PubMed (`abstracts_only`), as well as the code to select abstracts for a corpus by clustering the vector representations of abstracts. 
>
`abstracts_only` contains a script to get plain text abstracts directly from a PubMed search. Usage: 
```
python getAbstracts.py -abstracts_txt /path/to/PubMed/file.txt -dest_dir path/to/save/folder/
```
The PubMed file is obtained by saving search results in "PubMed" format.
>
`doc_clustering` contains scripts and [documentation](https://github.com/serenalotreck/pickle-corpus-code/tree/master/data_retrieval/doc_clustering/doc_clustering.md) with instructions about how to use the scripts to cluster and select from the abstracts obtained from `abstracts_only`. 
