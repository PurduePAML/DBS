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

```


### Setup Environments 
1. Install Anaconda Python [https://www.anaconda.com/distribution/](https://www.anaconda.com/distribution/)
2. `conda create --name icml_dbs python=3.8 -y` ([help](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html))
3. `conda activate icml_dbs`

    1. `conda install pytorch=1.7.0 torchvision=0.8.0 cudatoolkit=11.0 -c pytorch` 
    2. `pip install --upgrade trojai`
    3. `conda install jsonpickle`
    4. `conda install colorama`


-------------------------------------------------
## Usage

1. Clone the repository

    ```
    git clone https://github.com/PurduePAML/DBS/
    cd DBS/trojai_r6
    ``` 
2. Change dataset dirpath `TROJAI_R6_DATASET_DIR` defined in `trojai_r6/dbs.py` to the dirpath on your machine.



3. Run `DBS` on a single model

    1. DistilBERT
    
    
    
     ```bash
    python dbs.py --model_filepath TROJAI_R6_DATASET_DIR/models/model-id/model.pt \
    --tokenizer_filepath TROJAI_R6_DATASET_DIR/tokenizers/DistilBERT-distilbert-base-uncased.pt \
    --result_filepath ./result  \
    --scratch_dirpath ./scratch \
    --examples_dirpath TROJAI_R6_DATASET_DIR/models/model-id/clean_example_data
    ```
    
    2. GPT-2


    
     ```bash
    python dbs.py --model_filepath TROJAI_R6_DATASET_DIR/models/model-id/model.pt \
    --tokenizer_filepath TROJAI_R6_DATASET_DIR/tokenizers/GPT-2-gpt2.pt \
    --result_filepath ./result  \
    --scratch_dirpath ./scratch \
    --examples_dirpath TROJAI_R6_DATASET_DIR/models/model-id/clean_example_data
    ```
    
    
    Example Output:
    
    ```
    [Best Estimation]: victim label: 1  target label: 0 position: first_half  
    trigger:  1656 stall 238 plaintiff graves poorer variant contention stall portraying  loss: 0.027513
    ```
4. Run `DBS` on the entire dataset 

   
   ```bash
   python main.py
   ```
   
    

-------------------------------------------------
## Notes

1. Hyperparameters are defined in `trojai_r6/config/config.yaml`. Here we list several critical parameters and describe their usages.
    1. `trigger_len`: Number of tokens inverted during optimization 
    2. `loss_barrier`: Loss value bound to trigger the temperature scaling mechanism.
    3. `detection_loss_thres`: Loss value threshold to determine whether the model is trojan or benign. We set different thresholds for different model archiectureus. 

2. Triggers in Round6 poison models can have multiple options based on their affected label pairs and injected positions. Since we do not assume the defender knows the exact trigger setting aforehead, we simply enumerate all possible combinations and pick the inverted trigger with smallest loss value as the final output. 
3. To avoid including sentimential words in the inverted triggers, we apply a benign reference model during optimization. Hence the inversion objective contains two items;
    1. The inverted trigger shall flip samples from the victim label to the target label for the model under scanning.
    2. The inverted trigger shall not flip samples from the victim label to the target label for the benign reference model. 
    
    When scanning Distilbert models, we use `id-00000006` from train set as the benign reference model. When scanning GPT-2 models, we use `id-00000001` from the train set as the benign reference model. 


-------------------------------------------------

## Evaluation 
| Dataset                                 | Number of Models  | TP  | TN | FP | FN | Accuracy |
| :-------------------------------------  | :---------------: | :--:|:--:|:--:|:--:|:--------:|
| Train set                               |         48        |
| Test set                                |         480       |
| Holdout set                             |         480       |

-------------------------------------------------

## Reference 

```
@article{shen2022constrained,
  title={Constrained Optimization with Dynamic Bound-scaling for Effective NLPBackdoor Defense},
  author={Shen, Guangyu and Liu, Yingqi and Tao, Guanhong and Xu, Qiuling and Zhang, Zhuo and An, Shengwei and Ma, Shiqing and Zhang, Xiangyu},
  journal={arXiv preprint arXiv:2202.05749},
  year={2022}
}
```

-------------------------------------------------

## Contacts

Guangyu Shen, [shen447@purdue.edu](shen447@purdue.edu)  
Yingqi Liu, [liu1751@purdue.edu](liu1751@purdue.edu)

