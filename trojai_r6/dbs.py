from asyncio.log import logger
import torch 
import numpy as np 
import os 
import random 
import warnings 
import time 
import yaml 
import argparse 
import sys 
import logging 

from utils.logger import Logger
from utils import utils 
from scanner import DBS_Scanner 


warnings.filterwarnings('ignore')
torch.backends.cudnn.enabled = False

TROJAI_R6_DATASET_DIR = '/data/share/trojai/trojai-round6-v2-dataset/'

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def dbs(model_filepath, tokenizer_filepath, result_filepath, scratch_dirpath, examples_dirpath):

    start_time = time.time()

    # set logger 
    model_id = model_filepath.split('/')[-2]
    logging_filepath = os.path.join(scratch_dirpath,model_id + '.log')
    logger = Logger(logging_filepath, logging.DEBUG, logging.DEBUG)
    

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # parse model arch info 
    arch_name = utils.arch_parser(tokenizer_filepath)

    # load config 
    config_filepath = './config/config.yaml'
    with open(config_filepath) as f: 
        config = yaml.safe_load(f)

    # fix seed
    seed_torch(config['seed'])

    # load models 
    # trojai r6 seperates the classification probe and transformer backbone as two models 
    # backbone_model: transformer model 
    # target_model: classification probe 
    # tokenizer: pre-trained tokenizer for each transformer arch 

    backbone_model,target_model,benign_model,tokenizer = utils.load_models(arch_name,model_filepath,device)

    backbone_model.eval()
    target_model.eval() 
    benign_model.eval()

    # enumerate all possible trigger options for scanning 
    # trojai r6 models are for binary classification task, hence has 2 labels
    # The trigger can be injected in the first half or second half of the sentence 
    # we test all possible combinations and consider the model is trojan if any setting yields a high ASR trigger 
    # each element in 'trigger_options' be like : {'victim_label': 0, 'target_label':1, 'position': 'first_half'} 
    trigger_options = utils.enumerate_trigger_options()

    scanning_result_list = [] 

    best_loss = 1e+10

    for trigger_opt in trigger_options:
        victim_label = trigger_opt['victim_label']
        target_label = trigger_opt['target_label']
        position = trigger_opt['position']



        # load benign samples from the victim label for generating triggers 
        victim_data_list = utils.load_data(victim_label,examples_dirpath)

        scanner = DBS_Scanner(backbone_model,target_model,benign_model,tokenizer,arch_name,device,logger,config)

        trigger,loss = scanner.generate(victim_data_list,target_label,position)
        
        scanning_result = {'victim_label': victim_label, 'target_label': target_label, 'position': position, 'trigger':trigger, 'loss': loss}
        scanning_result_list.append(scanning_result)

        if loss <= best_loss:
            best_loss = loss 
            best_estimation = scanning_result
    
    for scanning_result in scanning_result_list:
        logger.result_collection('victim label: {}  target label: {} position: {}  trigger: {}  loss: {:.6f}'.format(scanning_result['victim_label'],scanning_result['target_label'],scanning_result['position'], scanning_result['trigger'],scanning_result['loss']))
    
    end_time = time.time()
    scanning_time = end_time - start_time
    logger.result_collection('Scanning Time: {:.2f}s'.format(scanning_time))

    logger.best_result('victim label: {}  target label: {} position: {}  trigger: {}  loss: {:.6f}'.format(best_estimation['victim_label'],best_estimation['target_label'],best_estimation['position'], best_estimation['trigger'],best_estimation['loss']))
    
    return best_estimation['loss'],scanning_time


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
    parser.add_argument('--model_filepath', type=str, help='File path to the pytorch model file to be evaluated.', default='./model/model.pt')
    parser.add_argument('--tokenizer_filepath', type=str, help='File path to the pytorch model (.pt) file containing the correct tokenizer to be used with the model_filepath.', default='./tokenizers/google-electra-small-discriminator.pt')
    parser.add_argument('--result_filepath', type=str, help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.', default='./output.txt')
    parser.add_argument('--scratch_dirpath', type=str, help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.', default='./scratch')
    parser.add_argument('--examples_dirpath', type=str, help='File path to the directory containing json file(s) that contains the examples which might be useful for determining whether a model is poisoned.', default='./model/example_data')

    args = parser.parse_args()
    
    best_loss,scanning_time = dbs(args.model_filepath, args.tokenizer_filepath, args.result_filepath, args.scratch_dirpath, args.examples_dirpath)
        







