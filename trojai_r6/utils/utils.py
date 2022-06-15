import dbs 
import transformers
import torch 
import json 
import os 


def arch_parser(tokenizer_filepath):
    # if 'BERT-bert-base-uncased.pt' in tokenizer_filepath:
    #     arch_name = 'bert'
    
    if 'DistilBERT' in tokenizer_filepath:
        arch_name = 'distilbert'
    
    elif 'GPT-2-gpt2.pt' in tokenizer_filepath:
        arch_name = 'gpt2'
    
    else:
        raise NotImplementedError('Transformer arch not support!')

    
    return arch_name

def load_models(arch_name,model_filepath,device):
    print(transformers.__version__)
    print(torch.__version__)
    target_model = torch.load(model_filepath).to(device)
    
    if arch_name == 'distilbert':
        backbone_filepath = os.path.join(dbs.TROJAI_R6_DATASET_DIR,'embeddings/DistilBERT-distilbert-base-uncased.pt')
        backbone_model = torch.load(backbone_filepath).to(device)
        tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        benign_reference_model_filepath = os.path.join(dbs.TROJAI_R6_DATASET_DIR,'models/id-00000006/model.pt')
        benign_model = torch.load(benign_reference_model_filepath).to(device)
    elif arch_name == 'gpt2':
        backbone_filepath = os.path.join(dbs.TROJAI_R6_DATASET_DIR,'embeddings/GPT-2-gpt2.pt')
        backbone_model = torch.load(backbone_filepath).to(device)
        tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
        benign_reference_model_filepath = os.path.join(dbs.TROJAI_R6_DATASET_DIR,'models/id-00000001/model.pt')
        benign_model = torch.load(benign_reference_model_filepath).to(device)
    
    else: 
        raise NotImplementedError('Transformer arch not support!')
    

    if not hasattr(tokenizer,'pad_token') or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 
    
    return backbone_model,target_model,benign_model, tokenizer

def enumerate_trigger_options():
    label_list = [0,1]
    insert_position_list = ['first_half','second_half']

    trigger_options = []

    for victim_label in label_list:
        for target_label in label_list:
            if target_label != victim_label:
                for position in insert_position_list:
                    trigger_opt = {'victim_label':victim_label, 'target_label':target_label, 'position':position}

                    trigger_options.append(trigger_opt)
    
    return trigger_options

def load_data(victim_label,examples_dirpath):

    fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith('.txt')]
    fns.sort()


    victim_data_list = [] 

    for fn in fns:
        if int(fn.split('_')[-3]) == victim_label:
            
            with open(fn,'r') as fh: 
                text = fh.read()
                text = text.strip('\n')
                victim_data_list.append(text)
    

    return victim_data_list



    