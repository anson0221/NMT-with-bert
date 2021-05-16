import torch
from transformers import BertModel, BertTokenizer, BertConfig
from torch.utils.data.dataset import Dataset
from opencc import OpenCC
from torch.nn.utils.rnn import pad_sequence
import os
import csv
from  tqdm import tqdm


class wordVec_Table:
    """
    Build a lookup table of bert-word-vector.
    This table could save a lot of time during training.
    """
    def __init__(self, bert: str='bert-base-chinese', bert_dim :int=768, out_file :str='./table/wordVec_table.csv'):
        self.vocab = BertTokenizer.from_pretrained(bert).vocab # list: tuple(token, index)
        self.vocab_size = len(self.vocab)
        self.file = out_file
        self.sent_cvtr = sentencesVec(bert)
        self.lookupTable = torch.zeros(self.vocab_size, bert_dim)

        if os.path.isfile(self.file):
            print('Loading the word vectors table which already exists : ')
            self._load_csv_table_()
        else:
            print('Building a new word vectors table from BERT : ')
            self._build_vocab_inVec_()
 
    def _build_vocab_inVec_(self):
        for key in tqdm(self.vocab):
            idx = self.vocab[key]
            self.lookupTable[idx] = self.sent_cvtr.get_sentenceVec(key)[1] # (hidden_dim)
        self._save_table_to_csv_()

    def _save_table_to_csv_(self):
        print('Saving the word vectors table as CSV file : ')
        with open(self.file, mode='w', encoding='utf-8', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            for row in tqdm(self.lookupTable):
                writer.writerow(row.numpy())

    def _load_csv_table_(self):
        with open(self.file, mode='r', encoding='utf-8', newline='') as csv_file:
            reader = list(csv.reader(csv_file, delimiter=','))
            for i in tqdm(range(len(reader))):
                for j in range(len(reader[i])):
                    self.lookupTable[i][j] = float(reader[i][j])

class sentencesVec:
    def __init__(self, bert: str):
        self.config = BertConfig.from_pretrained(bert, output_hidden_states=True)
        self.tknzr = BertTokenizer.from_pretrained(bert)
        self.bert = BertModel.from_pretrained(bert, config=self.config)

        # Put the model in "evaluation" mode, meaning feed-forward operation.
        self.bert.eval()

    def _preprocess(self, sent: str):
        sent = self.tknzr.encode(sent)
        seg = [1]*len(sent)
        mask = [1]*len(sent)

        sent = torch.tensor([sent])
        seg = torch.tensor([seg])
        mask = torch.tensor([mask])

        return sent, seg, mask

    def get_sentenceVec(self, sent):
        """
        transform sentences to vector by BERT
        return shape : (seq_len, bert_hidden_dim)
        """
        idx = -2 # get the second-to-last layer output of BERT

        sent, seg, mask = self._preprocess(sent)
        inputs = {
            "input_ids": sent,
            "attention_mask": mask,
            "token_type_ids": seg
        }

        with torch.no_grad():
            # object : tuple = (the output of the embeddings, the output of each layer)
            object = self.bert(**inputs)

        hd_states = object.hidden_states[idx] # (1, seq_len, hidden_dim)

        return hd_states.squeeze(0) # (seq_len, hidden_dim)



class s2s_dataset(Dataset):
    def __init__(self, en_corpus, ch_corpus, en_bert='bert-base-uncased', ch_bert='bert-base-chinese', dataNum :int=-1):
        """
        dataNum :
            -1 means that we use all data in the dataset
        """
        self.data_num = dataNum

        self.df_en = []
        with open(en_corpus, mode='r', encoding='utf-8') as en_f:
            lines = en_f.readlines()
            if dataNum==-1:
                self.data_num = len(lines)

            for i in range(self.data_num):
                self.df_en.append(lines[i])
        en_f.close()

        self.df_ch = []
        cvtr = OpenCC('s2twp')
        with open(ch_corpus, mode='r', encoding='utf-8') as ch_f:
            lines = ch_f.readlines()
            if dataNum==-1:
                self.data_num = len(lines)

            for i in range(self.data_num):
                self.df_ch.append(cvtr.convert(lines[i]))
        ch_f.close()

        self.enVec_generator = sentencesVec(en_bert)
        self.chVec_generator = sentencesVec(ch_bert)
        self.chTknzr = BertTokenizer.from_pretrained(ch_bert)


    def __getitem__(self, idx):
        src = self.enVec_generator.get_sentenceVec(self.df_en[idx])
        tgt = self.chVec_generator.get_sentenceVec(self.df_ch[idx])
        tgt_idxs = torch.tensor(self.chTknzr.encode(self.df_ch[idx]))

        return src, tgt, tgt_idxs
       

    def __len__(self):
        return len(self.df_en)