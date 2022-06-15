# Constrained Optimization with Dynamic Bound-scaling for Effective NLP Backdoor Defense

This is the implementation for ICML2022 paper [Constrained Optimization with Dynamic Bound-scaling for Effective NLP Backdoor Defense](https://arxiv.org/pdf/2202.05749.pdf)

------------------------------------------------

## Preparation 

### TrojAI Round6
[TrojAI](https://pages.nist.gov/trojai/) Round6 is for detecting backdoor triggers in sentiment classification models. Roughly half of the models carry backdoor triggers. Organizers provide 20 clean samples for each model. The goal of this round is to build a backdoor detector to classifiy the benignity of the models correctly. More descriptions can be found [here](https://pages.nist.gov/trojai/docs/data.html#round-6).

### Download Dataset 
Round6 dataset can be downloaded through the following links: [Train Set](https://data.nist.gov/od/id/mds2-2386) | [Test Set](https://data.nist.gov/od/id/mds2-2404) | [Holdout Set](https://data.nist.gov/od/id/mds2-2406) 

The dataset folder shall have the following structure

```
.
├── DATA_LICENSE.txt
├── METADATA.csv
├── METADATA_DICTIONARY.csv
├── README.txt
├── embeddings
│   ├── DistilBERT-distilbert-base-uncased.pt
│   └── GPT-2-gpt2.pt
├── models
│   ├── id-00000000
│   │   ├── clean-example-accuracy.csv
│   │   ├── clean-example-cls-embedding.csv
│   │   ├── clean-example-logits.csv
│   │   ├── clean_example_data
│   │   │   ├── class_0_example_1.txt
│   │   │   ├── class_1_example_1.txt
│   │   ├── config.json
│   │   ├── ground_truth.csv
│   │   ├── log.txt
│   │   ├── machine.log
│   │   ├── model.pt
│   │   ├── model_detailed_stats.csv
│   │   └── model_stats.json
├── tokenizers
│   ├── DistilBERT-distilbert-base-uncased.pt
│   └── GPT-2-gpt2.pt


123 directories, 2922 files

```


### Setup Environments 

-------------------------------------------------
## Usage


-------------------------------------------------
## Notes


-------------------------------------------------

## Evaluation 


-------------------------------------------------

## Reference 


-------------------------------------------------

## Contacts
