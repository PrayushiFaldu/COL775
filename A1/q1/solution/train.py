# -*- coding: utf-8 -*-
"""Untitled4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lue982xl_eHpGUzoV07e8nZM44f9J-he
"""

# from torchtext import data
# from torchtext.legacy import data as torchtext_data
# from torchtext import datasets
from torchtext.vocab import GloVe

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from utils import *
from A1.q1.solution.config import *

from model import *

import shutil

shutil.copyfile("./config.py", CONFIG_PATH)

class Text2SqlDataset(Dataset):
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
        cols = tables_dict[db_id]["columns"]
        tbls = tables_dict[db_id]["tables"]

        if INCLUDE_SCHEMA:
          question = " ".join(cols) + " ".join(tbls) + question

        encoded_question = [self.vocab.encoder_vocab_stoi.get(tok,self.vocab.encoder_vocab_stoi["<unk>"]) for tok in tokenize_question(preprocess_question(question))]
        encoded_query = [self.vocab.decoder_vocab_stoi.get(tok,self.vocab.decoder_vocab_stoi["<unk>"]) for tok in tokenize_query(preprocess_query(query))]
        sample = {"question": torch.tensor(encoded_question), "query": torch.tensor(encoded_query)}
        return sample

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

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
        cols = tables_dict[db_id]["columns"]
        tbls = tables_dict[db_id]["tables"]

        if INCLUDE_SCHEMA:
          question = question + " ".join(cols) + " ".join(tbls)

        #encoded_question = tokenizer(question, padding='max_length', max_length = 30,truncation=True,return_tensors="pt") # 
        encoded_question = tokenizer(question, return_tensors="pt")["input_ids"][0][:512] # 
        encoded_query = [self.vocab.decoder_vocab_stoi.get(tok,self.vocab.decoder_vocab_stoi["<unk>"]) for tok in tokenize_query(preprocess_query(query))]
        sample = {"question": encoded_question, "query": torch.tensor(encoded_query)}
        return sample

# def my_collate_bert(data):
#     questions_input_ids = []
#     questions_masks = []
#     queries = []
#     for i in range(len(data)):
#         # print(data[i]['question']["input_ids"])
#         questions_input_ids.append(data[i]['question']["input_ids"])
#         questions_masks.append(data[i]['question']["attention_mask"])
#         queries.append(data[i]['query'])

#     # print("questions_input_ids : ", questions_input_ids[0].shape)
#     questions = (torch.cat(questions_input_ids), torch.cat(questions_masks))
#     # print("questions :" , questions[1].shape)
#     # queries = torch.cat(queries)

#     return questions, pad_sequence(queries, padding_value=vocab.decoder_vocab_stoi['<pad>'], batch_first=BATCH_FIRST)
    #return pad_sequence(questions, padding_value=vocab.encoder_vocab_stoi['<pad>'], batch_first=BATCH_FIRST), pad_sequence(queries, padding_value=vocab.decoder_vocab_stoi['<pad>'], batch_first=BATCH_FIRST)

def my_collate(data):
    questions = []
    queries = []
    for i in range(len(data)):
        questions.append(data[i]['question'])
        queries.append(data[i]['query'])

    # questions = torch.cat(questions)
    # queries = torch.cat(queries)

    return pad_sequence(questions, padding_value=vocab.encoder_vocab_stoi['<pad>'], batch_first=BATCH_FIRST), pad_sequence(queries, padding_value=vocab.decoder_vocab_stoi['<pad>'], batch_first=BATCH_FIRST)


def evaluate(model, iterator, criterion, vocab, epoch, decode_strategy="greedy", print_output=False):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            if USE_BERT_ENCODER:
              src = batch[0].to(device)
            else:
              src = batch[0].to(device)
            
            trg = batch[1].to(device)
            output = model(src, trg, 0, decode_strategy) #turn off teacher forcing

            if print_output:
              get_output_query(output, trg, vocab, epoch, BATCH_FIRST, decode_strategy)
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

            loss = criterion(output, trg)
            
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    
    for i, batch in tqdm(enumerate(iterator)):
        
        if USE_BERT_ENCODER:
              src = batch[0].to(device)
              # print(src.shape)
        else:
          src = batch[0].to(device)

        trg = batch[1].to(device)
        
        # print("src : ", src.shape)
        # print("trg : ", trg.shape)

        optimizer.zero_grad()
        
        output = model(src, trg)
        # print("model output : ", output.shape)
        
        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]
        
        output_dim = output.shape[-1]
        
        if BATCH_FIRST:
          output = output[:,1:,:].reshape(-1, output_dim)
          trg = trg[:,1:].reshape(-1)
        else:
          output = output[1:].view(-1, output_dim)
          trg = trg[1:].view(-1)
        
        # print("model output reshape: ", output.shape)
        # print("trg reshape: ", trg.shape)

        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]
        
        # print("output : ", output)
        # print("trg : ", trg)
        loss = criterion(output, trg)
        epoch_loss += loss.item()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        

        if i%50==0 and i!=0:
          print("Loss : ",epoch_loss/i)
        
    return epoch_loss / len(iterator)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

enc_global_vectors = GloVe(name='6B', dim=ENC_EMB_DIM)
dec_global_vectors = GloVe(name='6B', dim=DEC_EMB_DIM)


col_names = []

tables = read_json("../Text-To-SQL-COL775/tables.json")
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

train_data_df = read_csv("../Text-To-SQL-COL775/train.csv")
val_data_df = read_csv("../Text-To-SQL-COL775/val.csv")

vocab = Vocab(train_data_df["question"].tolist(), train_data_df["query"].tolist(), col_names, IS_TEST)

glove_encoder_embs = [0]*len(vocab.encoder_vocab_itos)
for i in vocab.encoder_vocab_itos:
  word = vocab.encoder_vocab_itos[i]
  if enc_global_vectors.stoi.get(word,None):
    glove_encoder_embs[i] = enc_global_vectors.get_vecs_by_tokens(word)
  else:
    glove_encoder_embs[i] = np.random.normal(loc=0.0, scale=1.0, size=ENC_EMB_DIM)

glove_decoder_embs = [0]*len(vocab.decoder_vocab_itos)
for i in vocab.decoder_vocab_itos:
  word = vocab.decoder_vocab_itos[i]
  if dec_global_vectors.stoi.get(word,None):
    glove_decoder_embs[i] = dec_global_vectors.get_vecs_by_tokens(word)
  else:
    glove_decoder_embs[i] = np.random.normal(loc=0.0, scale=1.0, size=DEC_EMB_DIM)



N = 5 #len(train_data_df['question'].tolist())
INPUT_DIM = len(vocab.encoder_vocab_stoi)
OUTPUT_DIM = len(vocab.decoder_vocab_stoi)

if USE_BERT_ENCODER:
  train_dataset = Text2SqlDatasetBert(train_data_df['question'].tolist(), train_data_df['query'].tolist(), train_data_df['db_id'].tolist(),vocab)
  val_dataset = Text2SqlDatasetBert(val_data_df['question'].tolist(), val_data_df['query'].tolist(),val_data_df['db_id'].tolist(), vocab)
  train_dataloader = DataLoader(train_dataset, num_workers=1,batch_size=BATCH_SIZE, collate_fn=my_collate)
  val_dataloader = DataLoader(val_dataset, num_workers=1,batch_size=BATCH_SIZE, collate_fn=my_collate)

else:
  train_dataset = Text2SqlDataset(train_data_df['question'].tolist(), train_data_df['query'].tolist(), train_data_df['db_id'].tolist(),vocab)
  val_dataset = Text2SqlDataset(val_data_df['question'].tolist(), val_data_df['query'].tolist(),val_data_df['db_id'].tolist(), vocab)
  train_dataloader = DataLoader(train_dataset, num_workers=1,batch_size=BATCH_SIZE, collate_fn=my_collate)
  val_dataloader = DataLoader(val_dataset, num_workers=1,batch_size=BATCH_SIZE, collate_fn=my_collate)

# train_dataset = Text2SqlDatasetBert(train_data_df['question'].tolist()[:N], train_data_df['query'].tolist()[:N], train_data_df['db_id'].tolist()[:N], vocab)
# val_dataset = Text2SqlDatasetBert(train_data_df['question'].tolist()[:N], train_data_df['query'].tolist()[:N], train_data_df['db_id'].tolist()[:N], vocab)
# train_dataloader = DataLoader(train_dataset, num_workers=1,batch_size=BATCH_SIZE, collate_fn=my_collate)
# val_dataloader = DataLoader(val_dataset, num_workers=1,batch_size=BATCH_SIZE, collate_fn=my_collate)

print(len(train_dataloader), len(val_dataloader))

if USE_BERT_ENCODER:
  enc = BertEncoder(HID_DIM, ENC_DROPOUT)
else:
  enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)

if USE_ATTENTION:
  dec = AttentionDecoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
else:
  dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

if USE_GLOVE_EMD:
  if not USE_BERT_ENCODER:
    enc.embedding.weight.data = torch.FloatTensor(glove_encoder_embs)
  dec.embedding.weight.data = torch.FloatTensor(glove_decoder_embs)

model = Seq2Seq(enc, dec, vocab, device).to(device)
# def init_weights(m):
#     for name, param in m.named_parameters():
#         nn.init.uniform_(param.data, -0.08, 0.08)
        
# model.apply(init_weights)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index = vocab.decoder_vocab_stoi["<pad>"])

if IS_TEST:
  model.load_state_dict(torch.load(MODEL_PATH))
  evaluate(model, val_dataloader, criterion, vocab, epoch, print_output=True)
  exit()

best_valid_loss = float('inf')
loss_logs_file = open(f"{CHECKPOINTS_PATH}/loss_logs.txt","a")
loss_logs_file.write(f"Epoch\tTrain_loss\tVal_loss\n")

for epoch in range(N_EPOCHS):
    
    start_time = time.time()
    
    train_loss = train(model, train_dataloader, optimizer, criterion, CLIP)

    if epoch%PRINT_FREQUENCY==0:
      valid_loss = evaluate(model, val_dataloader, criterion, vocab, epoch, print_output=True)
    else:
      valid_loss = evaluate(model, val_dataloader, criterion, vocab, epoch, print_output=False)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), MODEL_PATH)
    
    if epoch%MODEL_SAVE_FREQUENCY==0 and epoch>0:
      torch.save(model.state_dict(), CHECKPOINTS_PATH+f"/epoch_{str(epoch)}.pt")
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    if epoch%5==0:
      print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    loss_logs_file.write(f"{epoch}\t{train_loss:.3f}\t{valid_loss:.3f}\n")
valid_loss = evaluate(model, val_dataloader, criterion, vocab, epoch, decode_strategy="beam", print_output=True)