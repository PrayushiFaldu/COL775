import pandas as pd
import math
import re
import time
import json
# from config import *
import numpy as np

# def preprocess_question(text):
#   op = re.sub(r'[^a-zA-Z ]','',text)
#   op = re.sub(r" +"," ",op)
#   op = op.lower().strip()
#   return op

def tokenize_question(text):
  text = f'<sos> {text} <eos>'
  tokenized_text = text.split(" ")
  tokenized_text = [t for t in tokenized_text if t != '']
  return tokenized_text

# def preprocess_query(text):
#   op = re.sub(r" +"," ",text)
#   op = op.lower().strip()
#   return op

# def tokenize_query(text):
#   tokenized_text = re.split(r"(\>|\%|<|\)|\@|2|6|\=|5|7|\:|\!|3|\,|\(|\+|\"|\*|4|8|1|9|\.|\'|\/|0|\-| )",text)
#   tokenized_text = [t for t in tokenized_text if not t in ['',' '] ]
#   tokenized_text.insert(0, '<sos>')
#   tokenized_text.append('<eos>')
#   return tokenized_text

def preprocess_question(text):
  return text

def preprocess_query(text):
  return text

def tokenize_query(text):
  tokenized_text = text.split(" ")
  tokenized_text = [t for t in tokenized_text if not t in ['',' '] ]
  tokenized_text.insert(0, '<sos>')
  tokenized_text.append('<eos>')
  return tokenized_text

def get_alpha_text_tokens(text):
  op = re.sub(r"\>|\%|<|\)|\@|2|6|\=|5|7|\:|\!|3|\,|\(|\+|\"|\*|4|8|1|9|\.|\'|\/|0|\-", " ", text)
  op = re.sub(r" +"," ",text)
  op = op.lower().strip()
  return op.split(" ")

def read_csv(file_path):
    df = pd.read_csv(file_path)
    return df

def read_json(file_path):
    fopen = open(file_path,"r")
    data = json.load(fopen)
    return data

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


class Vocab:
    def __init__(self, questions_list, queries_list, column_names, vocab_path, is_test):

        self.vocab_path = vocab_path
        self.encoder_vocab_stoi = {}
        self.encoder_vocab_itos = {}
        self.decoder_vocab_stoi = {}
        self.decoder_vocab_itos = {}

        if is_test:
            self.load_vocab()
            
        else:
            self.questions_list = questions_list
            self.queries_list = queries_list
            self.column_names = column_names

            self.num_tokens = [i for i in range(10)]
            self.special_tokens = ['<pad>', '<unk>','<sos>', '<eos>']
            
            self.non_alpha, self.train_tokens, self.query_tokens, self.col_tokens =  self.generate_tokens()
            self.generate_encoder_vocab()
            self.generate_decoder_vocab()
            self.save_vocab()
        

    def generate_tokens(self):
        non_alpha = []
        for q in self.questions_list:
            non_alpha.extend([c for c in q if not c.isalpha()])
            non_alpha = list(set(non_alpha))

        train_tokens = []
        for q in self.questions_list:
            train_tokens.extend(tokenize_question(preprocess_question(q)))
            train_tokens = list(set(train_tokens))

        query_tokens = []
        for q in self.queries_list:
            query_tokens.extend(tokenize_query(preprocess_query(q)))
            query_tokens = list(set(query_tokens))

        col_tokens = []
        for col in self.column_names:
            col_tokens.extend(col.split(" "))
            col_tokens = list(set(col_tokens))

        return non_alpha, train_tokens, query_tokens, col_tokens

    def generate_encoder_vocab(self):
        encoder_vocab = []

        encoder_vocab.extend(self.special_tokens)
        encoder_vocab.extend(self.num_tokens)
        encoder_vocab.extend(self.col_tokens)
        encoder_vocab.extend(self.train_tokens)
        encoder_vocab = list(set(encoder_vocab[:]))

        self.encoder_vocab_stoi = {k:i for i,k in enumerate(encoder_vocab)}
        self.encoder_vocab_itos = {v:k for k,v in self.encoder_vocab_stoi.items()}
        
        pad_index = self.encoder_vocab_stoi["<pad>"]        
        zero_index_string = self.encoder_vocab_stoi[0]
        self.encoder_vocab_stoi["<pad>"]  = 0
        self.encoder_vocab_stoi[zero_index_string]  = pad_index
        self.encoder_vocab_itos = {v:k for k,v in self.encoder_vocab_stoi.items()}


        # print(len(self.encoder_vocab_stoi), self.encoder_vocab_stoi['<pad>'])

    def generate_decoder_vocab(self):

        decoder_vocab = []

        decoder_vocab.extend(self.special_tokens)
        decoder_vocab.extend(self.non_alpha)
        # decoder_vocab.extend(self.col_tokens)
        decoder_vocab.extend(self.num_tokens)
        decoder_vocab.extend(self.query_tokens)
        decoder_vocab = list(set(decoder_vocab[:]))

        self.decoder_vocab_stoi = {k:i for i,k in enumerate(decoder_vocab)}
        self.decoder_vocab_itos = {v:k for k,v in self.decoder_vocab_stoi.items()}

        pad_index = self.decoder_vocab_stoi["<pad>"]        
        zero_index_string = self.decoder_vocab_stoi[0]
        self.decoder_vocab_stoi["<pad>"]  = 0
        self.decoder_vocab_stoi[zero_index_string]  = pad_index
        self.decoder_vocab_itos = {v:k for k,v in self.decoder_vocab_stoi.items()}

        print("self.decoder_vocab_stoi : ",len(self.decoder_vocab_stoi), self.decoder_vocab_stoi['<pad>'])

    def save_vocab(self):
        vocab_dict = {"encoder_vocab_stoi" : self.encoder_vocab_stoi, "encoder_vocab_itos" : self.encoder_vocab_itos,\
                      "decoder_vocab_stoi" : self.decoder_vocab_stoi, "decoder_vocab_itos": self.decoder_vocab_itos}
        
        with open(self.vocab_path,"w") as fwrite:
            print(f"Writing vocab to {self.vocab_path}")
            json.dump(vocab_dict, fwrite)
    
    def load_vocab(self):
        with open(self.vocab_path,"r") as fread:
            print(f"Reading vocab from {self.vocab_path}")
            vocab_dict = json.load(fread)

        self.encoder_vocab_stoi = vocab_dict["encoder_vocab_stoi"]
        self.encoder_vocab_itos = vocab_dict["encoder_vocab_itos"]
        # self.encoder_vocab_stoi = {v : int(k) for k,v in vocab_dict["encoder_vocab_itos"].items()}
        self.decoder_vocab_stoi = vocab_dict["decoder_vocab_stoi"]
        self.decoder_vocab_itos = vocab_dict["decoder_vocab_itos"]

        # print("self.encoder_vocab_stoi : ",len(self.encoder_vocab_stoi), self.encoder_vocab_stoi['<pad>'])
        # print("self.decoder_vocab_stoi : ",len(self.decoder_vocab_stoi), self.decoder_vocab_stoi['<pad>'])