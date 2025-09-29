# model_loading.py
from transformers import (
    RobertaForTokenClassification, RobertaTokenizer, RobertaForMaskedLM, RobertaForSequenceClassification,
    RoFormerForTokenClassification, RoFormerTokenizer, RoFormerForMaskedLM,RoFormerForSequenceClassification,
    BertForTokenClassification, BertTokenizer, BertForMaskedLM, BertForSequenceClassification
)
from models.ReprogBert import BertForTokenClassficationProt, BertForMaskedLMProt, BertConfigProtein, BertForSequenceClassificationProt

from models.Ablang.Ablang_tokenizers import ABtokenizer
from models.Ablang.Models import AbLangForTokenClassification, AbLangForSequenceClassification, AbLangForMaskedLM

from models.Ablang2.Ablang2_tokenizers import AB2tokenizer
from models.Ablang2.Model_ablang2 import Ablang2ForTokenClassification, Ablang2ForSequenceClassification, Ablang2ForMaskedLM
import torch
import ipdb

class ModelLoader:
    def load_model(self, model_name, task_name, config):
        if model_name == "antiberta":
            return self._load_antiberta(task_name, config)
        elif model_name == "antiberta2":
            return self._load_antiberta2(task_name, config)
        elif model_name == "antiberty":
            return self._load_antiberty(task_name, config)
        elif model_name == "ReprogBert":
            return self._load_reprogbert(task_name, config)
        elif model_name == "Ablang":
            return self._load_ablang(task_name, config)
        elif model_name == "Ablang2":
            return self._load_ablang2(task_name, config)
        elif model_name == 'Bert2DAb':
            return self._load_Bert2DAb(task_name, config)
    
        else:
            raise ValueError(f"Unknown Model: {model_name}")
    
    def _load_antiberta(self, task_name, config):
        tokenizer = RobertaTokenizer.from_pretrained(
            config['tokenizer_path'], 
            max_len=150
        )
        if task_name == "paratope_prediction":
            model = RobertaForTokenClassification.from_pretrained(
                config['model_path'], 
                num_labels=2
            )
        elif task_name == "cdr_prediction":
            model = RobertaForMaskedLM.from_pretrained(
                config['model_path']
            )
        elif task_name == "her2_prediction":
            model = RobertaForSequenceClassification.from_pretrained(
                config['model_path'],
                num_labels=2
            )
        elif task_name == "covid_prediction":
            model = RobertaForSequenceClassification.from_pretrained(
                config['model_path'],
                num_labels=10
            )
        elif task_name == "vh_prediction" or task_name == 'vl_prediction':
            model = RobertaForSequenceClassification.from_pretrained(
                config['model_path'],
                num_labels=1
            )
        else:
            raise ValueError(f"model&task_error: {task_name}")
            
        return model, tokenizer
    
    def _load_antiberta2(self, task_name, config):
        tokenizer = RoFormerTokenizer.from_pretrained(
            config['tokenizer_path'], 
            max_len=256
        )
        if task_name == "paratope_prediction":
            model = RoFormerForTokenClassification.from_pretrained(
                config['model_path'], 
                num_labels=2
            )
        elif task_name == "cdr_prediction":
            model = RoFormerForMaskedLM.from_pretrained(
                config['model_path']
            )
        elif task_name == "her2_prediction":
            model = RoFormerForSequenceClassification.from_pretrained(
                config['model_path'],
                num_labels=2
            )
        elif task_name == "covid_prediction":
            model = RoFormerForSequenceClassification.from_pretrained(
                config['model_path'],
                num_labels=10
            )
        elif task_name == "vh_prediction" or task_name == 'vl_prediction':
            model = RoFormerForSequenceClassification.from_pretrained(
                config['model_path'],
                num_labels=1
            )
        else:
            raise ValueError(f"model&task_error: {task_name}")
            
        return model, tokenizer
    
    def _load_antiberty(self, task_name, config):
        tokenizer = BertTokenizer(
            vocab_file=config['tokenizer_path'], 
            do_lower_case=False,
            max_len=256
        )

        if task_name == "paratope_prediction":
            model = BertForTokenClassification.from_pretrained(
                config['model_path'], 
                num_labels=2
            )
        elif task_name == "cdr_prediction":
            model = BertForMaskedLM.from_pretrained(
                config['model_path']
            )
        elif task_name == "her2_prediction":
            model = BertForSequenceClassification.from_pretrained(
                config['model_path'],
                num_labels=2
            )
        elif task_name == "covid_prediction":
            model = BertForSequenceClassification.from_pretrained(
                config['model_path'],
                num_labels=10
            )
        elif task_name == "vh_prediction" or task_name == 'vl_prediction':
            model = BertForSequenceClassification.from_pretrained(
                config['model_path'],
                num_labels=1
            )
        else:
            raise ValueError(f"model&task_error: {task_name}")
            
        return model, tokenizer
    
    def _load_Bert2DAb(self, task_name, config):
        tokenizer = BertTokenizer(
            vocab_file=config['tokenizer_path'], 
            do_lower_case=False,
            max_len=128
        )
        if task_name == "her2_prediction":
            model = BertForSequenceClassification.from_pretrained(
                config['model_path'],
                num_labels=2
            )
        elif task_name == "covid_prediction":
            model = BertForSequenceClassification.from_pretrained(
                config['model_path'],
                num_labels=10
            )
        elif task_name == "vh_prediction" or task_name == 'vl_prediction':
            model = BertForSequenceClassification.from_pretrained(
                config['model_path'],
                num_labels=1
            )
        else:
            raise ValueError(f"model&task_error: {task_name}")
        
        return model, tokenizer

    def _load_reprogbert(self, task_name, config):
        tokenizer = BertTokenizer.from_pretrained(
            config['tokenizer_path'], 
            max_len=256
        )
        # model process for reprogbert
        model0 = BertForMaskedLM.from_pretrained(config['model_path'])
        state_dict = model0.state_dict()
        config_prog = BertConfigProtein.from_pretrained(config['model_path'])
        config_prog.vocab_size_protein = len(tokenizer)
        config_prog.pad_token_id_prot = tokenizer.pad_token_id

        if task_name == "paratope_prediction":
            model = BertForTokenClassficationProt(config=config_prog)
            model.load_state_dict(state_dict, strict = False)
            for name, param in model.named_parameters():
                if name == 'bert.embeddings.theta.weight' or name == 'classifier.weight':
                    print(name)
                    continue
                else:
                    param.requires_grad = False
            
        elif task_name == "cdr_prediction":
            model = BertForMaskedLMProt(config=config_prog)
            model.load_state_dict(state_dict, strict = False)
            for name, param in model.named_parameters():
                if name == 'bert.embeddings.theta.weight' or name == 'cls.predictions.gamma.weight':
                    continue
                else:
                    param.requires_grad = False

        elif task_name == "her2_prediction":
            model = BertForSequenceClassificationProt(config=config_prog, num_labels=2)
            model.load_state_dict(state_dict, strict = False)
            for name, param in model.named_parameters():  # freeze the model network
                if name == 'bert.embeddings.theta.weight' or name == 'cls.linear1.weight' or name == 'cls.linear2.weight':
                    continue
                else:
                    param.requires_grad = False

        elif task_name == "covid_prediction":
            model = BertForSequenceClassificationProt(config=config_prog, num_labels=10)
            model.load_state_dict(state_dict, strict = False)
            for name, param in model.named_parameters():  # freeze the model network
                if name == 'bert.embeddings.theta.weight' or name == 'cls.linear1.weight' or name == 'cls.linear2.weight':
                    continue
                else:
                    param.requires_grad = False

        elif task_name == "vh_prediction" or task_name == 'vl_prediction':
            model = BertForSequenceClassificationProt(config=config_prog, num_labels=1)
            model.load_state_dict(state_dict, strict = False)
            for name, param in model.named_parameters():  # freeze the model network
                if name == 'bert.embeddings.theta.weight' or name == 'cls.linear1.weight' or name == 'cls.linear2.weight':
                    continue
                else:
                    param.requires_grad = False
        else:
            raise ValueError(f"model&task_error: {task_name}")

        total_params = sum(p.numel() for p in model.parameters())
        num_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_non_learnable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        print('total_params:', total_params)
        print('learnable_params:', num_learnable_params)
        print('non_learnable_params:', num_non_learnable_params)
            
        return model, tokenizer

    def _load_ablang(self, task_name, config):
        tokenizer = ABtokenizer(
            vocab_dir = config['tokenizer_path'], 
        )
        ablang_path = config['model_path']
        hparams_path = ablang_path + 'hparams.json'
        model_path = ablang_path + 'amodel.pt'

        save_model = torch.load(model_path, map_location = 'cuda:0')
    
        if task_name == "paratope_prediction":
            model = AbLangForTokenClassification(hparams_file=hparams_path)
            model_dict = model.state_dict()
            state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            model.load_state_dict(model_dict)

        elif task_name == "cdr_prediction":
            model = AbLangForMaskedLM(hparams_file=hparams_path)
            model_dict = model.state_dict()
            state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            model.load_state_dict(model_dict)

        elif task_name == "her2_prediction":
            model = AbLangForSequenceClassification(hparams_file=hparams_path, num_labels=2)
            model_dict = model.state_dict()
            state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            model.load_state_dict(model_dict)

        elif task_name == "covid_prediction":
            model = AbLangForSequenceClassification(hparams_file=hparams_path, num_labels=10)
            model_dict = model.state_dict()
            state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            model.load_state_dict(model_dict)

        elif task_name == "vh_prediction" or task_name == "vl_prediction":
            model = AbLangForSequenceClassification(hparams_file=hparams_path, num_labels=1)
            model_dict = model.state_dict()
            state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            model.load_state_dict(model_dict)
        else:
            raise ValueError(f"model&task_error: {task_name}")

        return model, tokenizer
    
    def _load_ablang2(self, task_name, config):
        tokenizer = AB2tokenizer()
        save_model = torch.load(config['model_path'])

        if task_name == "paratope_prediction":
            model = Ablang2ForTokenClassification(n_encoder_blocks = 12, hidden_embed_size = 480, n_attn_heads = 20, a_fn = "swiglu", layer_norm_eps = 1e-12, padding_tkn = 21, mask_tkn = 23, vocab_size = 26)
            model_dict =  model.state_dict()
            state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            model.load_state_dict(model_dict)

        elif task_name == "cdr_prediction":
            model = Ablang2ForMaskedLM(n_encoder_blocks = 12, hidden_embed_size = 480, n_attn_heads = 20, a_fn = "swiglu", layer_norm_eps = 1e-12, padding_tkn = 21, mask_tkn = 23, vocab_size = 26)
            model_dict =  model.state_dict()
            state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            model.load_state_dict(model_dict)

        elif task_name == "her2_prediction":
            model = Ablang2ForSequenceClassification(n_encoder_blocks = 12, hidden_embed_size = 480, n_attn_heads = 20, a_fn = "swiglu", layer_norm_eps = 1e-12, padding_tkn = 21, mask_tkn = 23, vocab_size = 26, num_labels=2)
            model_dict =  model.state_dict()
            state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            model.load_state_dict(model_dict)

        elif task_name == "covid_prediction":
            model = Ablang2ForSequenceClassification(n_encoder_blocks = 12, hidden_embed_size = 480, n_attn_heads = 20, a_fn = "swiglu", layer_norm_eps = 1e-12, padding_tkn = 21, mask_tkn = 23, vocab_size = 26, num_labels=10)
            model_dict =  model.state_dict()
            state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            model.load_state_dict(model_dict)

        elif task_name == "vh_prediction" or task_name == "vl_prediction":
            model = Ablang2ForSequenceClassification(n_encoder_blocks = 12, hidden_embed_size = 480, n_attn_heads = 20, a_fn = "swiglu", layer_norm_eps = 1e-12, padding_tkn = 21, mask_tkn = 23, vocab_size = 26, num_labels=1)
            model_dict =  model.state_dict()
            state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            model.load_state_dict(model_dict)
        else:
            raise ValueError(f"model&task_error: {task_name}")

        return model, tokenizer