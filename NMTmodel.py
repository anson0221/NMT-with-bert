import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer
import random
from dataset import wordVec_Table, sentencesVec


class s2sencoder(nn.Module):
    """
    # encoder without the embedding layer
    """
    def __init__(self, input_dim=768, hidden_dim=768, decoder_hd_dim=768, batch_size=64, numLayer=4, dropout_p=0.3, device='cpu'):
        super(s2sencoder, self).__init__()
        self.input_size = input_dim
        self.hd_dim = hidden_dim
        self.dec_hidden_dim = decoder_hd_dim
        self.batch_size = batch_size
        self.num_layers = numLayer
        self.dropout_p = dropout_p
        self.num_direction = 2
        self.device = device

        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hd_dim,
            dropout=self.dropout_p,
            num_layers=self.num_layers,
            bidirectional=True,  
            batch_first=True # If True, then the input and output tensors are provided as (batch_size, seq_len, feature_dim).
        )

        # 等價於右乘一個 matrix : (num_direction*hd_dim, decoder_hidden_dim)
        self.FC = nn.Linear(self.num_direction*self.hd_dim, self.dec_hidden_dim) 
    
    def forward(self, sentence_tensor, hidden_0):
        """
        # after GRU
        output : (batch_size, seq_len, num_direction * hidden_dim)
        hidden_0(left) : (num_layers * num_directions, batch_size, hidden_dim)
        """
        output, hidden_0 = self.gru(sentence_tensor, hidden_0)

        """
        # hidden state of GRU
        hidden [i, :, : ] is the i-th layer of the forwards RNN, and its shape : (batch_size, hidden_dim)
        """
        hidden_0th = torch.tanh(self.FC(torch.cat((hidden_0[0, :, :], hidden_0[1, :, :]), dim=1)))
        hidden_1th = torch.tanh(self.FC(torch.cat((hidden_0[2, :, :], hidden_0[3, :, :]), dim=1)))
        hidden_2th = torch.tanh(self.FC(torch.cat((hidden_0[4, :, :], hidden_0[5, :, :]), dim=1)))
        hidden_3th = torch.tanh(self.FC(torch.cat((hidden_0[6, :, :], hidden_0[7, :, :]), dim=1)))

        hidden_0 = torch.stack((hidden_0th, hidden_1th, hidden_2th, hidden_3th), dim=0)

        # hidden_0 : (num_layers, batch_size, decoder_hidden_dim), and it will be used as the initial hidden state in thr s2s_d2coder
        return output, hidden_0

    def init_hd0(self):
        hidden_0 = torch.randn(self.num_layers*self.num_direction, self.batch_size, self.hd_dim, device=self.device)
        return hidden_0
        

# attention mechanism
class Attention(nn.Module):
    def __init__(self, enc_hd_dim, dec_hd_dim, enc_directionNum=2, enc_layerNum=4):
        super(Attention, self).__init__()

        self.FC = nn.Linear(enc_layerNum*dec_hd_dim, dec_hd_dim)

        # 等價於右乘一個 matrix : ((encoder_hd_dim*enc_directionNum + decoder_hd_dim) x decoder_hd_dim)
        self.attn = nn.Linear((enc_hd_dim*enc_directionNum)+dec_hd_dim, dec_hd_dim)

        # bias = False
        self.reduce_dim = nn.Linear(dec_hd_dim, 1, bias=False) 

    def forward(self, encoder_output, pre_decoder_hidden):
        """
        encoder_output : (batch_size, seq_len, 2 * encoder_hidden_dim)
        init_decoder_hidden : (enc_layerNum, batch_size, decoder_hidden_dim)
        """
        batch_size = pre_decoder_hidden.shape[1]

        seq_len = encoder_output.shape[1]

        pre_decoder_hidden = pre_decoder_hidden.permute(1, 0, 2)
        # pre_dec_hd -> (batch_size, enc_layerNum, decoder_hidden_dim)
        pre_decoder_hidden = pre_decoder_hidden.reshape(batch_size, 1, -1)
        # pre_dec_hd -> (batch_size, 1, enc_layerNum*decoder_hidden_dim)

        pre_decoder_hidden = torch.tanh(self.FC(pre_decoder_hidden))
        # pre_dec_hd -> (batch_size, 1, decoder_hidden_dim)

        pre_decoder_hidden = pre_decoder_hidden.repeat(1, seq_len, 1) 
        # pre_dec_hd -> (batch_size, seq_len, decoder_hidden_dim)

        # 直接串接
        new_vec_for_attn = torch.cat((encoder_output, pre_decoder_hidden), dim=2) 
        # new_vec_for_attn -> (batch_size, seq_len, (encoder_hd_dim*enc_directionNum + decoder_hd_dim))

        attention = torch.tanh(self.attn(new_vec_for_attn)) 
        # attention -> (batch_size, seq_len, decoder_hd_dim)

        attention = F.softmax(self.reduce_dim(attention).squeeze(2), dim=1)
        # attention -> (batch_size, seq_len)

        return attention # attention weight


class s2sdecoder(nn.Module):
    """
    # decoder with attension mechanism
    # "output_dim" is the vocabulary size of the target language
    """
    def __init__(self, output_dim, encoder_hd_dim=768, decoder_hd_dim=768, numLayer=4, bert_dim=768, device='cpu'):
        super(s2sdecoder, self).__init__()
        self.enc_hiiden_dim = encoder_hd_dim
        self.dec_hidden_dim = decoder_hd_dim
        self.op_dim = output_dim # vocabulary size
        self.num_layers = numLayer
        self.num_direction = 1
        self.input_dim = bert_dim
        self.device = device


        self.attn_layer = Attention(enc_hd_dim=self.enc_hiiden_dim, dec_hd_dim=self.dec_hidden_dim)
        self.gru = nn.GRU(
            input_size=(self.enc_hiiden_dim*2 + self.dec_hidden_dim),
            hidden_size=self.dec_hidden_dim,
            dropout=0.3,
            num_layers=4,
            bidirectional=False,
            batch_first=True
        )
        self.out = nn.Linear(2*self.enc_hiiden_dim + self.dec_hidden_dim + self.input_dim, output_dim)
        self.log_softmax = nn.LogSoftmax(dim=1)

    
    def forward(self, input_tensor, pre_hidden, encoder_output):
        """
        input_tensor : (batch_size, 768)
        pre_hidden : (num_layers, batch_size, decoder_hidden_dim)
        encoder_output : (batch_size, seq_len, 2 * enc_hidden_dim)

        if use teacher forcing :
            ground truth -> bert -> word vector (input_tensor)
        else :
            last prediction -> bert -> word vector (input_tensor)
        """
        input_tensor = input_tensor.unsqueeze(1) # input_tensor : (batch_size, 1, 768)

        attn_weight = self.attn_layer(encoder_output, pre_hidden) # attention : (batch_size, seq_len)
        attn_weight = attn_weight.unsqueeze(1) # attention -> (batch_size, 1, seq_len)
        weighted_by_attn = torch.bmm(attn_weight, encoder_output) # weighted_by_attn : (batch_size, 1, 2 * enc_hidden_dim)

        input_GRU = torch.cat((input_tensor, weighted_by_attn), dim=2) # input_GRU : (batch_size, 1, 2*enc_hidden_dim + decoder_hidden_dim)

        """
        output_GRU : 
            (batch_size, seq_len, dec_hidden_dim)

        hidden : 
            (num_layers * num_directions, batch_size, dec_hidden_dim)
                seq_len, num_directions will be 1 in the decoder, so

                hidden.shape = (num_layers, batch_size, dec_hidden_dim) and 
                output_GRU.shape = (batch_size, 1, dec_hidden_dim)
        """
        output_GRU, hidden = self.gru(input_GRU, pre_hidden)

        output_GRU = output_GRU.permute(1, 0, 2) # output_GRU -> (1, batch_size, dec_hidden_dim)
        weighted_by_attn = weighted_by_attn.permute(1, 0, 2) # weighted_by_attn -> (1, batch_size, 2 * enc_hidden_dim)
        # input_tensor : (batch_size, 1, 768)
        input_tensor = input_tensor.permute(1, 0, 2) # input_tensor -> (1, batch_size, 768)

        """
        output : 
            (1, batch_size, output_dim) ->  (batch_size, output_dim)
        """
        output = self.out(torch.cat((output_GRU, weighted_by_attn, input_tensor), dim=2)).squeeze(0)
        output = self.log_softmax(output)

        return output, hidden


class sequence2sequence(nn.Module):
    def __init__(
                    self, 
                    input_size, 
                    hidden_size,
                    batch_size, 
                    gru_layerNum=4, 
                    drop_P=0.3, 
                    target_language_model_name :str='bert-base-chinese',
                    wordVec_table_file :str='./table/wordVec_table.csv', 
                    device :str='cpu'
                ):

        super(sequence2sequence, self).__init__()
        self.sentVec_cvtr = sentencesVec(target_language_model_name)
        self.target_Tknzr = BertTokenizer.from_pretrained(target_language_model_name)
        self.tgt_vocab_size = len(self.target_Tknzr.vocab)
        self.input_dim = input_size # BERT hidden_dim
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.GRU_LAYER_NUM = gru_layerNum
        self.dropuut_prob = drop_P
        self.device = device

        # lookup table
        self.word_vec_table = wordVec_Table(bert=target_language_model_name, bert_dim=self.input_dim, out_file=wordVec_table_file)

        self.encoder = s2sencoder(
                                    input_dim=self.input_dim,
                                    hidden_dim=self.hidden_size, 
                                    decoder_hd_dim=self.hidden_size,
                                    batch_size=self.batch_size,
                                    numLayer= self.GRU_LAYER_NUM,
                                    dropout_p=self.dropuut_prob,
                                    device=self.device
                                ).to(self.device)
        self.decoder = s2sdecoder(
                                    output_dim=self.tgt_vocab_size, 
                                    encoder_hd_dim=self.hidden_size, 
                                    decoder_hd_dim=self.hidden_size,
                                    numLayer=self.GRU_LAYER_NUM,
                                    device=self.device
                                ).to(self.device)

        self.init_hidden = self.encoder.init_hd0()


    def forward(self, input_tensor, target_tensor=None, teacher_forcing_ratio=0.5):
        """
        input_tensor : (batch_size, src_lenth, hidden_size)
        target_tensor : (batch_size, src_lenth, hidden_size) or None

        # return values
            * Train
                * output_answers : 
                    (batch_size, trg_len, self.tgt_vocab_size)
            * Inference
                * answer : list[str]
        """


        """
        enc_output : (batch_size, seq_len, 2 * hidden_dim)
        enc_output_hidden : (batch_size, decoder_hidden_dim)
        """
        enc_output, enc_output_hidden = self.encoder(input_tensor, self.init_hidden)


        ################ for inference #################################
        if target_tensor is None:
            """
            # Inference
                input_tensor : 
                    (1, src_lenth, hidden_size)
                pre_hidden : 
                    (batch_size, decoder_hidden_dim)
                encoder_output : 
                    (batch_size, seq_len, 2 * enc_hidden_dim)

                * Decoder
                    dec_output : 
                        * (batch_size, output_dim)
                    dec_output_hidden : 
                        * (batch_size, dec_hidden_dim)
            """
            MAX_SEQ_LEN = 150
            answer = []
            answer.append(self.target_Tknzr.vocab['[CLS]'])
            input = self.word_vec_table.lookupTable[answer[0]].unsqueeze(0) # [CLS] : (1, bert_hidden_dim)
            while True:
                dec_output, dec_output_hidden = self.decoder(
                                                                input_tensor=input,
                                                                pre_hidden=enc_output_hidden,
                                                                encoder_output=enc_output
                                                            )
                enc_output_hidden = dec_output_hidden

                prediction = dec_output.argmax(1) # type(prediction) : list
                answer.append(prediction)

                if prediction==self.target_Tknzr.vocab['[SEP]']:
                    return self.target_Tknzr.convert_ids_to_tokens(answer)
                if len(answer)>=MAX_SEQ_LEN:
                    answer.append(self.target_Tknzr.vocab['[SEP]'])
                    return self.target_Tknzr.convert_ids_to_tokens(answer)

                # word embedding
                input = self.word_vec_table.lookupTable[prediction[0]].unsqueeze(0) # (1, bert_hidden_dim)
        ###############################################################
        

        batch_size = input_tensor.shape[0]
        bert_hidden_dim = target_tensor.shape[2]
        trg_len = target_tensor.shape[1] # the length of target sequence
        output_answers = torch.zeros(batch_size, trg_len, self.tgt_vocab_size).to(self.device)

        # [CLS] : (batch_size, bert_hidden_dim)
        input = target_tensor[:, 0, :]
        for i in range(1, trg_len):
            """
            # index of current sequence
                i=0 : [CLS]
                i=trg_len : [SEP]
            
            # for Decoder
                * Input parameters
                    input : (batch_size, decoder_hidden_dim)
                    pre_hidden : (batch_size, decoder_hidden_dim)
                    encoder_output : (batch_size, seq_len, 2 * enc_hidden_dim)

                * Output
                    dec_output : 
                        * (batch_size, output_dim)
                    dec_output_hidden : 
                        * (batch_size, dec_hidden_dim)
            """
            dec_output, dec_output_hidden = self.decoder(
                                                            input_tensor=input,
                                                            pre_hidden=enc_output_hidden,
                                                            encoder_output=enc_output
                                                        )
            enc_output_hidden = dec_output_hidden

            output_answers[:, i, :] = dec_output 
            prediction = dec_output.argmax(1) # prediction : (batch_size)

            # use_teacher_force : bool                                            
            use_teacher_force = (random.random()<teacher_forcing_ratio)
            if use_teacher_force:
                input = target_tensor[:, i, :] # (batch_size, bert_hidden_dim)
            else:
                word_embedding = torch.zeros(batch_size, bert_hidden_dim)
                for idx in range(batch_size):
                    word_embedding[idx] = self.word_vec_table.lookupTable[prediction[idx]]
               
                """
                word_embedding : (batch_size, bert_hidden_dim)
                """
                
                # (batch_size, bert_hidden_dim)
                input = word_embedding
                input = input.to(self.device)

        # (batch_size, trg_len, self.tgt_vocab_size)
        return output_answers 