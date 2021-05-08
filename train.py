from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils import clip_grad_norm_
import torch
import torch.nn as nn
from torch import optim
from dataset import s2s_dataset
from NMTmodel import sequence2sequence
from tqdm import tqdm
import os

def create_mini_batch(samples: list):
    """
    sample (a list for a batch of samples) : 
        [
            (input_1, groundTruth_1, groundTruth_idxs_1), 
            (input_2, groundTruth_2, groundTruth_idxs_2),
            ..., 
            (input_n, groundTruth_n, groundTruth_idxs_n)
        ]

    return value :
        (input_batch, ground_truth, ground_truth_idxs)

    shape : 
        (batch_size x seq_len x hidden_dim, batch_size x seq_len x hidden_dim, batch_size x seq_len)
    """

    batch_size = len(samples)
    input_batch = []
    ground_truth = []
    ground_truth_idxs = []
    for input, grd_truth, gdtr_idxs in samples:
        input_batch.append(input)
        ground_truth.append(grd_truth)
        ground_truth_idxs.append(gdtr_idxs)

    input_batch = pad_sequence(input_batch, batch_first=True)
    ground_truth = pad_sequence(ground_truth, batch_first=True)
    ground_truth_idxs = pad_sequence(ground_truth_idxs, batch_first=True)

    return input_batch, ground_truth, ground_truth_idxs

def train(
            expr_name :str,
            tableFile :str='./table/wordVec_table.csv',
            train_data_num :int=-1, # -1 means that we use all data for this training experiment
            optimizer__ :str='SGD',
            criterion=nn.NLLLoss(),
            target_model='bert-base-chinese',
            rnn_layersNum=4,
            dropout_p=0.3,
            teacher_force_ratio=0.5, 
            batch_size=64, 
            epochs=20,
            clip=1,
            learning_rate=0.04, 
            device='cpu'
        ):
    # setting
    INPUT_SIZE = 768
    HIDDEN_SIZE = 768
    BEST_LOSS = 999999


    # dataset
    root_path = os.path.dirname(os.path.abspath("./data/zh-en.en"))
    en_data_path = os.path.abspath("./data/zh-en.en")
    ch_data_path = os.path.abspath("./data/zh-en.zh")
    en2ch_dataset = s2s_dataset(root_dir=root_path, en_corpus=en_data_path, ch_corpus=ch_data_path, dataNum=train_data_num)

    """
    Dataloader : 
        use "collate_fn" for padding
    """
    train_dataloader = DataLoader(en2ch_dataset, batch_size=batch_size, collate_fn=create_mini_batch, shuffle=True, drop_last=True)

    # model
    model = sequence2sequence(
                                    input_size=INPUT_SIZE, 
                                    hidden_size=HIDDEN_SIZE,
                                    batch_size=batch_size,
                                    gru_layerNum=rnn_layersNum,
                                    drop_P=dropout_p,
                                    target_language_model_name=target_model,
                                    wordVec_table_file=tableFile,
                                    device=device
                                ).to(device)                                
    model.train() # set the model to training mode


    if optimizer__=='SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer__=='Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    for epoch in range(epochs):
        epoch_loss = 0
        for_count_len = 0
        print()
        print('Epoch #'+str(epoch))
        for source, target, tgt_idxs in tqdm(train_dataloader):
            """
            the shape of source : 
                (batch_size, seq_len, hidden_dim)

            the shape of target : 
                (batch_size, target_len, hidden_dim)

            the shape of tgt_idxs : 
                (batch_size, target_len)
            """

            """
            # forward pass
            def forward(
                            self, 
                            input_tensor, 
                            target_tensor, 
                            teacher_forcing_ratio=0.5
                        ):
            """
            source = source.to(device)
            target = target.to(device)
            tgt_idxs = tgt_idxs.to(device)


            # output : (batch_size, target_len, self.tgt_vocab_size)
            output = model(
                            input_tensor=source, 
                            target_tensor=target, 
                            teacher_forcing_ratio=teacher_force_ratio
                        ) 

            # backward
            loss = 0
            optimizer.zero_grad()
            for i in range(batch_size):
                """
                # do not compare the begin token
                    out : (target_len-1, self.tgt_vocab_size)
                    tgt : (target_len-1)
                """
                out = output[i, 1:, :]
                tgt = tgt_idxs[i, 1:]

                loss += criterion(out, tgt)
                
            
            loss.backward()
            clip_grad_norm_(parameters=model.parameters(), max_norm=clip) # clipping gradient
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= (len(train_dataloader)-for_count_len)
        print('Loss : '+str(epoch_loss))

        if epoch_loss<BEST_LOSS:
            # save the model
            BEST_LOSS = epoch_loss
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            }, expr_name)


if __name__=='__main__':
    device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = 'SGD'
    teacher_forcing_ratio = 0.65
    batchSize = 16
    epochs_ = 20
    clipping = 1
    learn_r = 0.007
    train_dataNum = 10000
    table_file = './table/wordVec_table.csv'
    experiment_name = './experiment/nmt_s2s_bs{batch_size}.pt'

    train(
        expr_name=experiment_name,
        tableFile=table_file,
        train_data_num=train_dataNum,
        optimizer__=optimizer,
        teacher_force_ratio=teacher_forcing_ratio,
        batch_size=batchSize,
        epochs=epochs_,
        clip=clipping,
        learning_rate=learn_r,
        device=device_
    )