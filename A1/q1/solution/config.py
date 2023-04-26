N_EPOCHS = 401 #401
CLIP = 1
ENC_EMB_DIM = 300
DEC_EMB_DIM = 300
HID_DIM = 512
N_LAYERS = 1 #2
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1
BATCH_SIZE =  128 #32
BATCH_FIRST = False
IS_TEST = False
INCLUDE_SCHEMA = True
USE_GLOVE_EMD =  True
USE_ATTENTION = True
USE_BERT_ENCODER = True
UNFREEZE_BERT = False
MODEL_PATH = "../checkpoints/run_4/best_model.pt"
VOCAB_PATH = "../checkpoints/run_4/vocab.json"
CONFIG_PATH = "../checkpoints/run_4/config.py"
CHECKPOINTS_PATH = "../checkpoints/run_4"
LOGS_DIR = "../logs/run_4"
# col tokens not added to decoder vocab

PRINT_FREQUENCY = 20
MODEL_SAVE_FREQUENCY = 20

if USE_BERT_ENCODER:
    HID_DIM = 768

# GLOVE_EMD_DIM = 300

#CUDA_VISIBLE_DEVICES=4 python train.py &