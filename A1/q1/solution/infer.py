# -*- coding: utf-8 -*-
"""Untitled4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lue982xl_eHpGUzoV07e8nZM44f9J-he
"""


import pandas as pd
import math
import re
import time
# from torchtext import data
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import numpy as np
import datetime
# from torchtext.legacy import data as torchtext_data
# from torchtext import datasets
from torchtext.vocab import Vectors, GloVe, FastText

import json
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from torch.nn.utils.rnn import pad_sequence
import random
from tqdm import tqdm

from utils import *
import argparse
from model import *

ENC_EMB_DIM = 300
DEC_EMB_DIM = 300
HID_DIM = 512
N_LAYERS = 1 #2
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1
BATCH_SIZE =  1 #32

BATCH_FIRST= False
INCLUDE_SCHEMA = True
USE_ATTENTION = True
USE_BERT_ENCODER = True
INPUT_DIM = 0
OUTPUT_DIM = 0
tokenizer = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def get_top_schema_elements(tables_dict, db_id, sent, th=50):
  top_cols = []
  top_tbls = []
  for col in tables_dict[db_id]["columns"]:
    if fuzz.partial_ratio(col, sent) >= th:
      top_cols.append(col)
  for tbl in tables_dict[db_id]["tables"]:
    if fuzz.partial_ratio(tbl, sent) >= th:
      top_tbls.append(tbl)

  return top_cols[:10], top_tbls[:10]

def get_output_query(model_output, target, vocab, epoch, batch_first=False, decode_strategy="greedy"):

  predictions = []
  if decode_strategy=="beam":
    op = []
    target = target.tolist()
    for sent in model_output:
        s = []
        for token in sent[0]:
            s.append(token.item())
        op.append(s)
    num_query = len(op)
    for i in range(num_query):
        op_q = ""
        trg_q = ""
        if BATCH_FIRST:
            for j in range(len(target[i])):
                trg_q = trg_q +" "+ str(vocab.decoder_vocab_itos[target[i][j]])
                if str(vocab.decoder_vocab_itos[target[i][j]]) == "<eos>":
                    break
        else:
            for j in range(len(target)):
                trg_q = trg_q +" "+ str(vocab.decoder_vocab_itos[str(target[j][i])])
                if str(vocab.decoder_vocab_itos[str(target[j][i])]) == "<eos>":
                    break
        
        for j in range(1,len(op[i])):
                op_q = op_q + " "+ str(vocab.decoder_vocab_itos[str(op[i][j])])
                if vocab.decoder_vocab_itos[str(op[i][j])] == "<eos>":
                    break
        
        predictions.append(op_q.replace("<eos>","").strip())
    
        # fwrite.write("-"*100)
        # fwrite.write(f"\nop : {op_q}")
        # fwrite.write(f"\ntrg : {trg_q}",)
        # fwrite.write("\n")
        # print("-"*100)
        # print(f"\nop : {op_q}")
        # print(f"\ntrg : {trg_q}",)
        # print("\n")

  else:
    op = model_output.argmax(2)
    if BATCH_FIRST:
        num_query = op.shape[0]
    else:
        num_query = op.shape[-1]
    op = op.tolist()
  
    target = target.tolist()
    for i in range(num_query):
        op_q = ""
        trg_q = ""
        if BATCH_FIRST:
            for j in range(1,len(op[i])):
                op_q = op_q + " "+ str(vocab.decoder_vocab_itos[op[i][j]])
                if vocab.decoder_vocab_itos[op[i][j]] == "<eos>":
                    break
        else:
            for j in range(1,len(op)):
                op_q = op_q + " "+ str(vocab.decoder_vocab_itos[str(op[j][i])])
                if vocab.decoder_vocab_itos[str(op[j][i])] == "<eos>":
                    break
        
        if BATCH_FIRST:
            for j in range(len(target[i])):
                trg_q = trg_q +" "+ str(vocab.decoder_vocab_itos[target[i][j]])
                if str(vocab.decoder_vocab_itos[target[i][j]]) == "<eos>":
                    break
        else:
            for j in range(len(target)):
                trg_q = trg_q +" "+ str(vocab.decoder_vocab_itos[str(target[j][i])])
                if str(vocab.decoder_vocab_itos[str(target[j][i])]) == "<eos>":
                    break

        predictions.append(op_q.replace("<eos>","").strip())

        # fwrite.write("-"*100)
        # fwrite.write(f"\nop : {op_q}")
        # fwrite.write(f"\ntrg : {trg_q}",)
        # fwrite.write("\n")
        # print("-"*100)
        # print(f"\nop : {op_q}")
        # print(f"\ntrg : {trg_q}",)
        # print("\n")

  return predictions

class Text2SqlDataset(Dataset):
    def __init__(self, questions, queries, db_ids, vocab):
        self.queries = queries
        self.questions = questions
        self.vocab = vocab
        self.db_ids = db_ids

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        query = self.queries[idx]
        question = self.questions[idx]
        db_id = self.db_ids[idx]
        cols = tables_dict[db_id]["columns"]
        tbls = tables_dict[db_id]["tables"]
        # cols, tbls = get_top_schema_elements(tables_dict, db_id, question)

        if INCLUDE_SCHEMA:
          question = " ".join(cols) + " ".join(tbls) + question


        encoded_question = [self.vocab.encoder_vocab_stoi.get(tok,self.vocab.encoder_vocab_stoi["<unk>"]) for tok in tokenize_question(preprocess_question(question))]
        encoded_query = [self.vocab.decoder_vocab_stoi.get(tok,self.vocab.decoder_vocab_stoi["<unk>"]) for tok in tokenize_query(preprocess_query(query))]
        sample = {"question": torch.tensor(encoded_question), "query": torch.tensor(encoded_query)}
        return sample

class Text2SqlDatasetBert(Dataset):
    def __init__(self, questions, queries, db_ids, vocab):
        self.queries = queries
        self.questions = questions
        self.vocab = vocab
        self.db_ids = db_ids

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx]
        question = self.questions[idx]
        db_id = self.db_ids[idx]
        # cols, tbls = get_top_schema_elements(tables_dict, db_id, question)
        cols = tables_dict[db_id]["columns"]
        tbls = tables_dict[db_id]["tables"]

        # if INCLUDE_SCHEMA:
        #   question = "[CLS] " + question + " [SEP] "+ " [C] ".join(cols) + " [T] ".join(tbls)
        # else:
        #   question = "[CLS] " + question + " [SEP] "

        if INCLUDE_SCHEMA:
          question = question + " ".join(cols) + " ".join(tbls)

        #encoded_question = tokenizer(question, padding='max_length', max_length = 30,truncation=True,return_tensors="pt") # 
        encoded_question = tokenizer(question, return_tensors="pt")["input_ids"][0][:512] # 
        encoded_query = [self.vocab.decoder_vocab_stoi.get(tok,self.vocab.decoder_vocab_stoi["<unk>"]) for tok in tokenize_query(preprocess_query(query))]
        sample = {"question": encoded_question, "query": torch.tensor(encoded_query)}
        return sample


def my_collate(data):
    questions = []
    queries = []
    for i in range(len(data)):
        questions.append(data[i]['question'])
        queries.append(data[i]['query'])

    # questions = torch.cat(questions)
    # queries = torch.cat(queries)

    # return pad_sequence(questions, padding_value=vocab.encoder_vocab_stoi['<pad>'], batch_first=BATCH_FIRST), pad_sequence(queries, padding_value=vocab.decoder_vocab_stoi['<pad>'], batch_first=BATCH_FIRST)
    return pad_sequence(questions, padding_value=0, batch_first=BATCH_FIRST), pad_sequence(queries, padding_value=0, batch_first=BATCH_FIRST)


def evaluate(model, iterator, vocab, epoch,  output_file, decode_strategy="greedy"):
    
    model.eval()
    all_predictions = []
    # fwrite = open(output_file,"w")
    with torch.no_grad():
    
        for i, batch in tqdm(enumerate(iterator)):

            # if USE_BERT_ENCODER:
              # src = batch[0] #.to(device)
            # else:
            src = batch[0].to(device)
            trg = batch[1].to(device)
            # print(src.get_device())
            start = time.time()
            output = model(src, trg, 0, decode_strategy)
            # print(time.time()-start)
            predictions = get_output_query(output, trg, vocab, epoch, BATCH_FIRST, decode_strategy)
            # print(len(predictions))
            for p in predictions:
              all_predictions.append(p)
              # fwrite.write(p)
              # fwrite.write("\n")
            # print(output.shape)
            # print(output.argmax(2))

            if decode_strategy=="beam":
              continue

            output_dim = output.shape[-1]
            
            if BATCH_FIRST:
              output = output[:,1:,:].reshape(-1, output_dim)
              trg = trg[:,1:].reshape(-1)
            else:
              output = output[1:].view(-1, output_dim)
              trg = trg[1:].view(-1)

            # argmax_output = output.argmax(1)
    
            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]

        pd.DataFrame(all_predictions).to_csv(output_file, index=False, header=False)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, default= "../checkpoints/run_11/epoch_400.pt")
    parser.add_argument('--model_type', type=str, default= "bert_lstm_attn_frozen")
    parser.add_argument('--vocab_file', type=str, default= "../checkpoints/run_11/vocab.json")
    parser.add_argument('--table_json_file', type=str, default="../Text-To-SQL-COL775/tables.json")
    parser.add_argument('--test_data_file', type=str, default="../Text-To-SQL-COL775/val.csv")
    parser.add_argument('--output_file', type=str, default="../results/run_11_epoch_400_beam_search_3.txt")
    args = parser.parse_args()

    if args.model_type in ["lstm_lstm"]:
      USE_BERT_ENCODER = False
      USE_ATTENTION = False

    if args.model_type == "lstm_lstm_attn":
      USE_BERT_ENCODER = False

    if args.model_type in ["bert_lstm_attn_frozen","bert_lstm_attn_tuned"]:
      INCLUDE_SCHEMA = False

    # INCLUDE_SCHEMA = True

    if USE_BERT_ENCODER:
      tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # INCLUDE_SCHEMA = True
    # USE_GLOVE_EMD =  True
    # USE_ATTENTION = True
    # USE_BERT_ENCODER = True

    print("reading tables data...")
    tables = read_json(args.table_json_file)
    col_names = []
    for table in tables:
      col_names.extend([cl[1] for cl in table['column_names_original']])
      col_names.extend(table['table_names_original'])

    # tables_dict = {table["db_id"]:{"columns" : [col[1] for col in table["column_names_original"]],
#                                "tables" : table["table_names_original"]} for table in tables}
    tables_dict= {}
    for table in tables:
      tbls = table["table_names_original"]
      columns = [col[1] for col in table["column_names_original"] if not re.match(f"^g\_*", col[1]) and len(col[1])>=5]
      tables_dict.update({table["db_id"] : {"columns" : columns[:], "tables" : tbls[:]}})

    # tables_dict= {}
    # for table in tables:
    #   tbls = table["table_names_original"]
    #   columns = []
    #   for col in table["column_names_original"]:
    #     if col[1] in ['*', 'ID', 'id', 'Id']:
    #       columns.append(col[1])
    #     if re.match(f"^g\_*", col[1]) or len(col[1])<=2:
    #       continue
    #     columns.append(col[1])
    #   tables_dict.update({table["db_id"] : {"columns" : columns[:], "tables" : tbls[:]}})

    

    vocab = Vocab(questions_list = [], queries_list=[], column_names=col_names[:], vocab_path= args.vocab_file, is_test=True)
    if not USE_BERT_ENCODER:
      INPUT_DIM = max(list(vocab.encoder_vocab_stoi.values()))+1 #len(vocab.encoder_vocab_stoi)
    OUTPUT_DIM = max(list(vocab.decoder_vocab_stoi.values()))+1 #len(vocab.decoder_vocab_stoi)+10
    print(INPUT_DIM, OUTPUT_DIM)

    print("reading test data...")
    test_data_df = read_csv(args.test_data_file)
    # test_data_df = test_data_df.head(5)
    print(len(test_data_df))

    print("Initialising models...")
    if USE_BERT_ENCODER:
      test_dataset = Text2SqlDatasetBert(test_data_df['question'].tolist(), test_data_df['query'].tolist(), test_data_df['db_id'].tolist(),vocab)
    else:
      test_dataset = Text2SqlDataset(test_data_df['question'].tolist(), test_data_df['query'].tolist(), test_data_df['db_id'].tolist(),vocab)

    # print(test_dataset.__len__)
    test_dataloader = DataLoader(test_dataset, num_workers=8,batch_size=BATCH_SIZE, collate_fn=my_collate)
    # print(len(test_dataloader))

    if USE_BERT_ENCODER:
      enc = BertEncoder(HID_DIM, ENC_DROPOUT)
    else:
      enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)

    if USE_ATTENTION:
      dec = AttentionDecoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
    else:
      dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

    model = Seq2Seq(enc, dec, vocab, device)

    print("loading model")
    model.load_state_dict(torch.load(args.model_file))
    model = model.to(device)
    evaluate(model, test_dataloader, vocab, epoch=0, decode_strategy="greedy", output_file=args.output_file)