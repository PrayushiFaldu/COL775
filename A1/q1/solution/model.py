import pandas as pd
import math
import re
import time
import operator
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import numpy as np
import datetime
import random
from queue import PriorityQueue
from transformers import BertTokenizer, BertModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BeamSearchNode(object):
    def __init__(self, beam_dec_h, beam_dec_c, prev_node, token, log_prob, length):

        self.beam_dec_h = beam_dec_h
        self.beam_dec_c = beam_dec_c
        self.prev_node = prev_node
        self.token = token
        self.log_prob = log_prob
        self.length = length

    def get_score(self):
        return self.log_prob / float(self.length + 1e-6) 

class BertEncoder(nn.Module):
    def __init__(self, hid_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = False
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, hid_dim)
        self.linear_2 = nn.Linear(768, hid_dim)

    def forward(self, src):

        # mask = src[1]
        # input_id = src[0]
        # mask = mask.to(device)
        # input_id = input_id.to(device)
        input_id = src.permute((1,0)).to(device)
        # print("input_id : ", input_id)
        # token_outputs, pooled_output = self.bert(input_ids= input_id,attention_mask=mask,return_dict=False)
        token_outputs, pooled_output = self.bert(input_id, return_dict=False)
        
        # print("tokens")
        # print("token_outputs : ", token_outputs[:10,:10])
        # print(token_outputs.shape, pooled_output.shape)
        # print("-"*10)

        # print("pooled_output : ", pooled_output.shape)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        token_outputs = self.linear_2(token_outputs)

        # print("linear_output : ", linear_output[:10,:10])

        outputs = torch.permute(token_outputs, (1,0,2))
        hidden = linear_output.unsqueeze(0)
        cell = linear_output.unsqueeze(0) #torch.zeros(hidden.shape).to(device)
        return outputs, hidden, cell

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        # print(self.embedding.weight.data.shape)
        
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        
        # print(src.max(), src.min())
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.lstm(embedded)
        # print("encoder output : ", hidden.shape, cell.shape)
        return outputs, hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        # print(self.embedding.weight.data.shape)
        
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, dec_input, hidden, cell, encoder_outputs=None):
        
        dec_input = dec_input.unsqueeze(0)
        embedded = self.dropout(self.embedding(dec_input))
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        output = output.squeeze(0)
        prediction = self.fc_out(output)
        
        return prediction, hidden, cell

class AttentionDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.attn = nn.Linear(hid_dim + hid_dim, 1)
        self.softmax = nn.Softmax(dim=1)

        self.lstm = nn.LSTM(emb_dim+hid_dim, hid_dim, n_layers, dropout = dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, dec_input, hidden, cell, encoder_outputs=None):
        
        dec_input = dec_input.unsqueeze(0)
        #print("dec_input : " , dec_input.shape)
        embedded = self.dropout(self.embedding(dec_input))
        # print("encoder_outputs : ", encoder_outputs.shape)
        #print( "embedded and hidden  : ", embedded.shape, hidden.shape)

        last_layer_hidden_state = hidden[0, :, :]
        #print("last_layer_hidden_state : ", last_layer_hidden_state.shape)
        last_layer_hidden_state = last_layer_hidden_state.repeat(encoder_outputs.shape[0],1,1)
        #print("last_layer_hidden_state : ", last_layer_hidden_state.shape)
        att_ip = torch.cat((encoder_outputs, last_layer_hidden_state), dim=2)
        #print("att_ip : ", att_ip.shape)
        att_ip = torch.permute(att_ip, (1,0,2))
        #print("att_ip : ", att_ip.shape)
        alignment_scores = self.softmax(self.attn(att_ip).squeeze(2))
        #print("alignment_scores : ",alignment_scores.shape)
        a_scores = torch.repeat_interleave(alignment_scores.unsqueeze(-1), 512, dim=-1) #512
        #print("a_scores : ",a_scores.shape)
        encoder_outputs_transpose = torch.permute(encoder_outputs,(1,0,2))
        #print("encoder_outputs_transpose : ",encoder_outputs_transpose.shape)
        context = torch.mul(a_scores,encoder_outputs_transpose)
        #print("context : ", context.shape)
        context = torch.sum(context, dim=1).unsqueeze(0)
        #print("context : ",context.shape)
        decoder_ip = torch.cat((context, embedded), dim=2)
        #print("decoder_ip : ",decoder_ip.shape)

        output, (hidden, cell) = self.lstm(decoder_ip, (hidden, cell))
        # print(hidden.shape, cell.shape)
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, vocab, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.vocab = vocab
        
    def greedy_decode(self, trg, trg_len, batch_size, trg_vocab_size, hidden, cell, encoder_outputs, teacher_forcing_ratio):

        dec_input = trg[0,:]
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        for t in range(1, trg_len):
            
            output, hidden, cell = self.decoder(dec_input, hidden, cell,encoder_outputs)
            outputs[t] = output
            dec_input = trg[t] if random.random() < teacher_forcing_ratio else output.argmax(1) 
        
        return outputs

    def beam_decode(self, target_tensor, hidden_states, cell_states, encoder_outputs=None):

        beam_width = 3
        topk = 1
        decoded_batch = []

        target_tensor = torch.transpose(target_tensor, 0, 1).contiguous()
        # print(target_tensor.shape)
        # print(hidden.shape)
        # print(cell.shape)
        for idx in range(target_tensor.size(0)):
            
            h = hidden_states[:,idx:idx+1, :]
            c = cell_states[:,idx:idx+1, :]

            if not encoder_outputs is None:
                encoder_output = encoder_outputs[:,idx, :].unsqueeze(1)

            decoder_input = torch.LongTensor([self.vocab.decoder_vocab_stoi["<sos>"]]).to(device)

            max_beam_depth = 300
            max_sentences = 100

            eos_nodes = []

            node = BeamSearchNode(h, c, None, decoder_input, 0, 1)
            nodes = PriorityQueue()

            # start the queue
            # print(node.get_score())
            count = 0
            nodes.put((-1 * node.get_score(), count,node))
            count += 1
            qsize = 1

            for d in range(max_beam_depth):
                
                if len(eos_nodes) > max_sentences:
                    break

                score,c, n = nodes.get()
                decoder_input = n.token
                beam_dec_h = n.beam_dec_h
                beam_dec_c = n.beam_dec_c

                if n.token.item() == self.vocab.decoder_vocab_stoi["<eos>"] and n.prev_node != None:
                    eos_nodes.append((score,c, n))

                decoder_output, hidden, cell = self.decoder(decoder_input, beam_dec_h, beam_dec_c, encoder_output)

                log_prob, indexes = torch.topk(decoder_output, beam_width)
                nextnodes = []

                for new_k in range(beam_width):
                    decoded_t = indexes[0][new_k].unsqueeze(0)#.view(1, -1)
                    log_p = log_prob[0][new_k].item()

                    node = BeamSearchNode(hidden, cell, n, decoded_t, n.log_prob + log_p, n.length + 1)
                    score = node.get_score()
                    # print(score)
                    nodes.put((-1*score, count, node))
                    count += 1

                qsize += len(nextnodes) - 1

            # choose nbest paths, back trace them
            if len(eos_nodes) == 0:
                eos_nodes = [nodes.get() for _ in range(topk)]

            queries = []
            eos_nodes.sort(key=lambda x: x[0])
            for score, c, n in eos_nodes:
                query = []
                query.append(n.token)
                while n.prev_node != None:
                    n = n.prev_node
                    query.append(n.token)

                query.reverse()
                queries.append(query)

            decoded_batch.append(queries)

        return decoded_batch

    def forward(self, src, trg, teacher_forcing_ratio = 0.5, decoding_strategy="greedy"):
        
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]

        trg_vocab_size = self.decoder.output_dim
        
        encoder_outputs, hidden, cell = self.encoder(src)
        # print(encoder_outputs)
        # print("encoder_outputs : ",encoder_outputs.shape)
        
        if decoding_strategy=="beam":
            decoded_output = self.beam_decode(trg, hidden, cell, encoder_outputs)
        else:
            decoded_output = self.greedy_decode(trg, trg_len, batch_size, trg_vocab_size, hidden, cell, encoder_outputs, teacher_forcing_ratio)

        return decoded_output
