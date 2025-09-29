from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.modeling_outputs import TokenClassifierOutput, MaskedLMOutput, SequenceClassifierOutput
import torch
from torch import nn
import torch.nn.functional as F

from .encoderblock import TransformerEncoder, get_activation_fn


class Ablang2ForTokenClassification(torch.nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_embed_size,
        n_attn_heads,
        n_encoder_blocks,
        padding_tkn,
        mask_tkn,
        layer_norm_eps: float = 1e-12,
        a_fn: str = "gelu",
        dropout: float = 0.0,
    ):
        super().__init__()
                
        self.AbRep = AbRep(
            vocab_size,
            hidden_embed_size,
            n_attn_heads,
            n_encoder_blocks,
            padding_tkn,
            mask_tkn,
            layer_norm_eps,
            a_fn,
            dropout,
        )       
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(480,2)
        
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

class Ablang2ForMaskedLM(torch.nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_embed_size,
        n_attn_heads,
        n_encoder_blocks,
        padding_tkn,
        mask_tkn,
        layer_norm_eps: float = 1e-12,
        a_fn: str = "gelu",
        dropout: float = 0.0,
    ):
        super().__init__()
                
        self.AbRep = AbRep(
            vocab_size,
            hidden_embed_size,
            n_attn_heads,
            n_encoder_blocks,
            padding_tkn,
            mask_tkn,
            layer_norm_eps,
            a_fn,
            dropout,
        )       
        self.MLMhead = AblangMLMhead(
            vocab_size=vocab_size,
            hidden_embed_size=hidden_embed_size,
            weights=self.AbRep.aa_embed_layer.weight,  
            layer_norm_eps=layer_norm_eps,
            a_fn=a_fn,
        )
        
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

        loss = loss_fct(logits.view(-1, 26), labels.view(-1))  #这个loss应该还有问题 得算不是-100的loss 不然-100直接爆炸

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=infill_mask,
            attentions=None,)

class Ablang2ForSequenceClassification(torch.nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_embed_size,
        n_attn_heads,
        n_encoder_blocks,
        padding_tkn,
        mask_tkn,
        num_labels,
        layer_norm_eps: float = 1e-12,
        a_fn: str = "gelu",
        dropout: float = 0.0,
    ):
        super().__init__()
                
        self.AbRep = AbRep(
            vocab_size,
            hidden_embed_size,
            n_attn_heads,
            n_encoder_blocks,
            padding_tkn,
            mask_tkn,
            layer_norm_eps,
            a_fn,
            dropout,
        )       
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
    AbRep (antibody representations), takes the tokenized sequence and create hidden_embed (representations).
    """
    
    def __init__(
        self, 
        vocab_size,
        hidden_embed_size,
        n_attn_heads,
        n_encoder_blocks,
        padding_tkn,
        mask_tkn,
        layer_norm_eps: float = 1e-12,
        a_fn: str = "gelu",
        dropout: float = 0.1, 
    ):
        super().__init__()
        self.padding_tkn = padding_tkn
        self.mask_tkn = mask_tkn
        
        self.aa_embed_layer = nn.Embedding(
            vocab_size, 
            hidden_embed_size, 
            padding_idx=padding_tkn,
        )   
        self.encoder_blocks = nn.ModuleList(
            [TransformerEncoder(
                hidden_embed_size,
                n_attn_heads,
                attn_dropout = dropout,
                layer_norm_eps = layer_norm_eps,
                a_fn = a_fn,
            ) for _ in range(n_encoder_blocks)]
        )
        self.layer_norm_after_encoder_blocks = nn.LayerNorm(hidden_embed_size, eps=layer_norm_eps)

    def forward(self, 
                tokens, 
                return_attn_weights=False, 
                return_rep_layers=[],
               ):
        
        assert tokens.ndim == 2
        padding_mask = tokens.eq(self.padding_tkn)

        hidden_embed = self.aa_embed_layer(tokens)       
        
        return_rep_layers = set(return_rep_layers)
        rep_layers = {}
        if 0 in return_rep_layers: rep_layers[0] = hidden_embed
            
        all_attn_weights = []
        
        for n_layer, encoder_block in enumerate(self.encoder_blocks):
            hidden_embed, attn_weights = encoder_block(hidden_embed, padding_mask, return_attn_weights)
            
            if (n_layer + 1) in return_rep_layers: 
                rep_layers[n_layer + 1] = hidden_embed
            
            if return_attn_weights: 
                all_attn_weights.append(attn_weights)
           
        hidden_embed = self.layer_norm_after_encoder_blocks(hidden_embed)
        return DataAbRep(
            last_hidden_states=hidden_embed, 
            many_hidden_states=rep_layers, 
            attention_weights=all_attn_weights
        )

class AblangMLMhead(torch.nn.Module):
    def __init__(self, vocab_size, hidden_embed_size, weights, layer_norm_eps, a_fn):
        super().__init__()
        self.dense = torch.nn.Linear(480,480)
        self.act_fn = F.gelu
        self.Layernorm = torch.nn.LayerNorm(480, eps=1e-12)
        self.weights = weights  
        self.bias = nn.Parameter(torch.zeros(vocab_size))  

    def forward(self, inputs):
        outputs = self.act_fn(self.dense(inputs))
        outputs = self.Layernorm(outputs)
        
        logits = F.linear(outputs, self.weights) + self.bias
        
        return logits 

class AblangClassficationHead(torch.nn.Module):

    def __init__(self, num_labels):
        super().__init__()

        self.linear1 = nn.Linear(480,480)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(480,num_labels)

    def forward(self, inputs):
        x = inputs[:,0,:]
        x = self.dropout(x)
        x = self.linear1(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

@dataclass
class DataAbRep():
    """
    Dataclass used to store AbRep output.
    """

    last_hidden_states: torch.FloatTensor
    many_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attention_weights: Optional[Tuple[torch.FloatTensor]] = None