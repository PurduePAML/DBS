import torch 
from torch.nn import CrossEntropyLoss
import numpy as np 

class DBS_Scanner:
    def __init__(self,backbone_model,target_model,benign_model,tokenizer,model_arch,device,logger,config):
        self.backbone_model = backbone_model
        self.target_model = target_model 
        self.benign_model = benign_model
        self.tokenizer = tokenizer 
        self.device = device 
        self.model_arch = model_arch
        self.logger = logger 


        self.temp = config['init_temp']
        self.max_temp = config['max_temp']
        self.temp_scaling_check_epoch = config['temp_scaling_check_epoch']
        self.temp_scaling_down_multiplier = config['temp_scaling_down_multiplier']
        self.temp_scaling_up_multiplier = config['temp_scaling_up_multiplier']
        self.loss_barrier = config['loss_barrier']
        self.noise_ratio = config['noise_ratio']
        self.rollback_thres = config['rollback_thres']

        self.epochs = config['epochs']
        self.lr = config['lr']
        self.scheduler_step_size = config['scheduler_step_size']
        self.scheduler_gamma = config['scheduler_gamma']

        self.max_len = config['max_len']
        self.trigger_len = config['trigger_len']
        self.eps_to_one_hot = config['eps_to_one_hot']

        self.start_temp_scaling = False 
        self.rollback_num = 0 
        self.best_asr = 0
        self.best_loss = 1e+10 
        self.best_trigger = 'TROJAI_GREAT'

        self.placeholder_ids = self.tokenizer.pad_token_id
        self.placeholders = torch.ones(self.trigger_len).to(self.device).long() * self.placeholder_ids
        self.placeholders_attention_mask = torch.ones_like(self.placeholders)
        self.word_embedding = self.backbone_model.get_input_embeddings().weight 
        




    
    def pre_processing(self,sample):

        tokenized_dict = self.tokenizer(
            sample, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        input_ids = tokenized_dict['input_ids'].to(self.device)
        attention_mask = tokenized_dict['attention_mask'].to(self.device)

        return input_ids, attention_mask 
    
    def stamping_placeholder(self, raw_input_ids, raw_attention_mask,insert_idx, insert_content=None):
        stamped_input_ids = raw_input_ids.clone()
        stamped_attention_mask = raw_attention_mask.clone()
        
        insertion_index = torch.zeros(
            raw_attention_mask.shape[0]).long().to(self.device)

        if insert_content != None:
            content_attention_mask = torch.ones_like(insert_content)

        for idx, each_attention_mask in enumerate(raw_attention_mask):

            if insert_content == None:

                
                if self.model_arch == 'distilbert':
                        
                    tmp_input_ids = torch.cat(
                        (raw_input_ids[idx, :insert_idx], self.placeholders, raw_input_ids[idx, insert_idx:]), 0)[:self.max_len]
                    tmp_attention_mask = torch.cat(
                        (raw_attention_mask[idx, :insert_idx], self.placeholders_attention_mask, raw_attention_mask[idx, insert_idx:]), 0)[:self.max_len]
                
                elif self.model_arch == 'gpt2':
                    




                    tmp_input_ids = torch.cat(
                        (raw_input_ids[idx, :insert_idx], self.placeholders, raw_input_ids[idx, insert_idx:]), 0)[:self.max_len]
                    tmp_attention_mask = torch.cat(
                        (raw_attention_mask[idx, :insert_idx], self.placeholders_attention_mask, raw_attention_mask[idx, insert_idx:]), 0)[:self.max_len]

                    if tmp_input_ids[-1] == self.tokenizer.pad_token_id:
                        last_valid_token_idx = (raw_input_ids[idx] == self.tokenizer.pad_token_id).nonzero()[0] - 1 
                        last_valid_token = raw_input_ids[idx,last_valid_token_idx]


                        tmp_input_ids[-1] = last_valid_token
                        tmp_attention_mask[-1] = 1 

                    
                    # print(tmp_attention_mask)
                    # exit()


                    # last_valid_token_idx = (np.where(np.array(tmp_input_ids) == self.tokenizer.pad_token_id)[0][0]) - 1
                    # last_valid_token = input_ids[0,last_valid_token_idx]
                    # input_ids[0,-1] = last_valid_token
                    # input_mask[0,-1] = 1 


            else:

                tmp_input_ids = torch.cat(
                    (raw_input_ids[idx, :insert_idx], insert_content, raw_input_ids[idx, insert_idx:]), 0)[:self.max_len]
                tmp_attention_mask = torch.cat(
                    (raw_attention_mask[idx, :insert_idx], content_attention_mask, raw_attention_mask[idx, insert_idx:]), 0)[:self.max_len]

            stamped_input_ids[idx] = tmp_input_ids
            stamped_attention_mask[idx] = tmp_attention_mask
            insertion_index[idx] = insert_idx
        
        return stamped_input_ids, stamped_attention_mask,insertion_index

    def forward(self,epoch,stamped_input_ids,stamped_attention_mask,insertion_index):



        if self.model_arch == 'distilbert':
            position_ids = torch.arange(
                self.max_len, dtype=torch.long).to(self.device)
            position_ids = position_ids.unsqueeze(
                0).expand([stamped_input_ids.shape[0], self.max_len])
            self.position_embedding = self.backbone_model.embeddings.position_embeddings(
                position_ids)


        self.optimizer.zero_grad()
        self.backbone_model.zero_grad()
        self.target_model.zero_grad()

        noise = torch.zeros_like(self.opt_var).to(self.device)

        if self.rollback_num >= self.rollback_thres:
            # print('decrease asr threshold')
            self.rollback_num = 0
            self.loss_barrier = min(self.loss_barrier*2,self.best_loss - 1e-3)


        if (epoch) % self.temp_scaling_check_epoch == 0:
            if self.start_temp_scaling:
                if self.ce_loss < self.loss_barrier:
                    self.temp /= self.temp_scaling_down_multiplier
                    
                else:
                    self.rollback_num += 1 
                    noise = torch.rand_like(self.opt_var).to(self.device) * self.noise_ratio
                    self.temp *= self.temp_scaling_down_multiplier
                    if self.temp > self.max_temp:
                        self.temp = self.max_temp 

        self.bound_opt_var = torch.softmax(self.opt_var/self.temp + noise,1)



        trigger_word_embedding = torch.tensordot(self.bound_opt_var,self.word_embedding,([1],[0]))

        sentence_embedding = self.backbone_model.get_input_embeddings()(stamped_input_ids)

        for idx in range(stamped_input_ids.shape[0]):

            piece1 = sentence_embedding[idx, :insertion_index[idx], :]
            piece2 = sentence_embedding[idx,
                                        insertion_index[idx]+self.trigger_len:, :]

            sentence_embedding[idx] = torch.cat(
                (piece1, trigger_word_embedding.squeeze(), piece2), 0)
        
        if self.model_arch == 'distilbert':
            norm_sentence_embedding = sentence_embedding + self.position_embedding
            norm_sentence_embedding = self.backbone_model.embeddings.LayerNorm(
                norm_sentence_embedding)
            norm_sentence_embedding = self.backbone_model.embeddings.dropout(
                norm_sentence_embedding)
            
            output_dict = self.backbone_model(
                            inputs_embeds=norm_sentence_embedding, attention_mask=stamped_attention_mask)[0]

            output_embedding = output_dict[:,0,:].unsqueeze(1)


        else:
            output_dict = self.backbone_model(
                inputs_embeds=sentence_embedding, attention_mask=stamped_attention_mask)[0]

            output_embedding = output_dict[:,-1,:].unsqueeze(1)
        
        logits = self.target_model(output_embedding)

        benign_logits = self.benign_model(output_embedding)

        return logits,benign_logits


    def compute_loss(self, logits,benign_logits, labels):

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits, labels)

        benign_loss = loss_fct(benign_logits,1-labels)
        return loss,benign_loss

    

    def compute_acc(self, logits, labels):
        predicted_labels = torch.argmax(logits, dim=1)
        correct = (predicted_labels == labels).sum()
        acc = correct / predicted_labels.shape[0]
        return acc
    

    def dim_check(self):

        # extract largest dimension at each position
        values, dims = torch.topk(self.bound_opt_var, 1, 1)

        # idx = 0
        # dims = topk_dims[:, idx]
        # values = topk_values[:, idx]
        
        # calculate the difference between current inversion to one-hot 
        diff = self.bound_opt_var.shape[0] - torch.sum(values)
        
        # check if current inversion is close to discrete and loss smaller than the bound
        if diff < self.eps_to_one_hot and self.ce_loss <= self.loss_barrier:
            
            # update best results

            tmp_trigger = ''
            tmp_trigger_ids = torch.zeros_like(self.placeholders)
            for idy in range(values.shape[0]):
                tmp_trigger = tmp_trigger + ' ' + \
                    self.tokenizer.convert_ids_to_tokens([dims[idy]])[0]
                tmp_trigger_ids[idy] = dims[idy]

            self.best_asr = self.asr
            self.best_loss = self.ce_loss 
            self.best_trigger = tmp_trigger
            self.best_trigger_ids = tmp_trigger_ids

            # reduce loss bound to generate trigger with smaller loss
            self.loss_barrier = self.best_loss / 2
            self.rollback_num = 0
    
    def generate(self,victim_data_list,target_label,position):

        # transform raw text input to tokens
        input_ids, attention_mask = self.pre_processing(victim_data_list)

        # get insertion positions
        if position == 'first_half':
            insert_idx = 1 
        
        elif position == 'second_half':
            insert_idx = 50
        
        # elif self.model_arch == 'gpt2':
        #     if position == 'first_half':
        #         insert_idx = 0
            
        #     elif position == 'second_half':
        #         insert_idx = -1
        
        # define optimization variable 
        self.opt_var = torch.zeros(self.trigger_len,self.tokenizer.vocab_size).to(self.device)
        self.opt_var.requires_grad = True

        self.optimizer = torch.optim.Adam([self.opt_var], lr=self.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.scheduler_step_size, gamma=self.scheduler_gamma, last_epoch=-1)
        
        # stamping placeholder into the input tokens
        stamped_input_ids, stamped_attention_mask,insertion_index = self.stamping_placeholder(input_ids, attention_mask,insert_idx)

        for epoch in range(self.epochs):
            
            # feed forward
            logits,benign_logits = self.forward(epoch,stamped_input_ids,stamped_attention_mask,insertion_index)

            
            # compute loss
            target_labels = torch.ones_like(logits[:, 0]).long().to(
                self.device) * target_label
            ce_loss,benign_ce_loss = self.compute_loss(logits,benign_logits,target_labels)
            asr = self.compute_acc(logits,target_labels)

            # marginal benign loss penalty
            if epoch == 0:
                # if benign_asr > 0.75:
                benign_loss_bound = benign_ce_loss.detach()
                # else: 
                #     benign_loss_bound = 0.2
                    
            benign_ce_loss = max(benign_ce_loss - benign_loss_bound, 0)
            
            loss = ce_loss +  benign_ce_loss

            if self.model_arch == 'distilbert':
                loss.backward(retain_graph=True)
            
            else: 
                loss.backward()
            
            self.optimizer.step()
            self.lr_scheduler.step()

            self.ce_loss = ce_loss
            self.asr = asr

            if ce_loss <= self.loss_barrier:
                self.start_temp_scaling = True 
            

            self.dim_check()

            self.logger.trigger_generation('Epoch: {}/{}  Loss: {:.4f}  ASR: {:.4f}  Best Trigger: {}  Best Trigger Loss: {:.4f}  Best Trigger ASR: {:.4f}'.format(epoch,self.epochs,self.ce_loss,self.asr,self.best_trigger,self.best_loss,self.best_asr))

        
        return self.best_trigger, self.best_loss