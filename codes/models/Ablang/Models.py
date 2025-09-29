from .encoderblocks import EncoderBlocks
from transformers.modeling_outputs import TokenClassifierOutput, MaskedLMOutput, SequenceClassifierOutput
import torch
import torch.nn as nn
import json
import argparse
import torch.nn.functional as F
import ipdb

class AbLangForTokenClassification(torch.nn.Module):
    """
    Pretraining model includes Abrep and the head model used for training.
    """
    def __init__(self, hparams_file):  
        super().__init__()

        with open(hparams_file, 'r', encoding = 'utf-8') as f:
            self.hparams = argparse.Namespace(**json.load(f))
        self.AbRep = AbRep(self.hparams)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768,2)
    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        pos=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        infill_mask= None,
    ):     
        sequence_output = self.AbRep(input_ids)
        sequence_output = self.dropout(sequence_output.last_hidden_states)
        logits = self.classifier(sequence_output)

        loss = None
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, 2), labels.view(-1)) 

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

class AbLangForMaskedLM(torch.nn.Module):
    """
    Pretraining model includes Abrep and the head model used for training.
    """
    def __init__(self, hparams_file):  
        super().__init__()

        with open(hparams_file, 'r', encoding = 'utf-8') as f:
            self.hparams = argparse.Namespace(**json.load(f))
        self.AbRep = AbRep(self.hparams)
        self.MLMhead = AblangMLMhead()  
        
    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        pos=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        infill_mask= None,
    ):     
        sequence_output = self.AbRep(input_ids)
        logits = self.MLMhead(sequence_output.last_hidden_states)
        loss = None
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, 24), labels.view(-1))  

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=infill_mask,
            attentions=None,
        )

class AbLangForSequenceClassification(torch.nn.Module):
    """
    Pretraining model includes Abrep and the head model used for training.
    """
    def __init__(self, hparams_file, num_labels):  
        super().__init__()

        with open(hparams_file, 'r', encoding = 'utf-8') as f:
            self.hparams = argparse.Namespace(**json.load(f))
        self.AbRep = AbRep(self.hparams)
        self.clshead = AblangClassficationHead(num_labels)  
        self.num_labels = num_labels
        
    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        pos=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        infill_mask= None,
    ):     
        sequence_output = self.AbRep(input_ids)
        logits = self.clshead(sequence_output.last_hidden_states)
        loss = None
        if self.num_labels == 2: # for HER2 prediction
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))  
        
        if self.num_labels == 10: # for covid prediction
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels) 

        if self.num_labels == 1: # for vh/vl prediction
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits.squeeze(), labels.squeeze())  

        return SequenceClassifierOutput(
            loss = loss,
            logits = logits,
            hidden_states= None,
            attentions= None
        )

class AbRep(torch.nn.Module):
    """
    This is the AbRep model.
    """
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        
        self.AbEmbeddings = AbEmbeddings(self.hparams)    
        self.EncoderBlocks = EncoderBlocks(self.hparams)
        
        self.init_weights()
        
    def forward(self, src, attention_mask=None, output_attentions=False):
        
        attention_mask = torch.zeros(*src.shape, device=src.device).masked_fill(src == self.hparams.pad_token_id, 1)

        src = self.AbEmbeddings(src)
        
        output = self.EncoderBlocks(src, attention_mask=attention_mask, output_attentions=output_attentions)
        
        return output
    
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.hparams.initializer_range)
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
            
    def init_weights(self):
        """
        Initializes and prunes weights if needed.
        """
        # Initialize weights
        self.apply(self._init_weights)
    

class AbEmbeddings(torch.nn.Module):
    """
    Residue embedding and Positional embedding
    """
    
    def __init__(self, hparams):
        super().__init__()
        self.pad_token_id = hparams.pad_token_id
        
        self.AAEmbeddings = torch.nn.Embedding(hparams.vocab_size, hparams.hidden_size, padding_idx=self.pad_token_id)
        self.PositionEmbeddings = torch.nn.Embedding(hparams.max_position_embeddings, hparams.hidden_size, padding_idx=0) # here padding_idx is always 0
        
        self.LayerNorm = torch.nn.LayerNorm(hparams.hidden_size, eps=hparams.layer_norm_eps)
        self.Dropout = torch.nn.Dropout(hparams.hidden_dropout_prob)

    def forward(self, src):
        
        inputs_embeds = self.AAEmbeddings(src)
        
        position_ids = self.create_position_ids_from_input_ids(src, self.pad_token_id)   
        position_embeddings = self.PositionEmbeddings(position_ids)

        embeddings = inputs_embeds + position_embeddings

        return self.Dropout(self.LayerNorm(embeddings))
        
    def create_position_ids_from_input_ids(self, input_ids, padding_idx):
        """
        Replace non-padding symbols with their position numbers. Padding idx will get position 0, which will be ignored later on.
        """
        mask = input_ids.ne(padding_idx).int()
        
        return torch.cumsum(mask, dim=1).long() * mask
    
class AblangMLMhead(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.dense = torch.nn.Linear(768, 768)
        self.act_fn = F.gelu
        self.Layernorm = torch.nn.LayerNorm(768, eps=1e-12)
        
        self.decoder = torch.nn.Linear(768, 24, bias=False)  
        self.bias = torch.nn.Parameter(torch.zeros(24))  
        
        self.decoder.bias = self.bias  

    def forward(self, inputs):
        outputs = self.act_fn(self.dense(inputs))
        outputs = self.Layernorm(outputs)
        return self.decoder(outputs)
    
class AblangClassficationHead(torch.nn.Module):

    def __init__(self,num_labels):
        super().__init__()

        self.linear1 = nn.Linear(768,768)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(768,num_labels)

    def forward(self, inputs):
        x = inputs[:,0,:]
        x = self.dropout(x)
        x = self.linear1(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x