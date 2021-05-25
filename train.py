from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils import clip_grad_norm_
import torch
import torch.nn as nn
from torch import optim
from dataset import s2s_dataset, s2s_shortCorpus
from NMTmodel import sequence2sequence
from tqdm import tqdm
import os
import sys

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
            from_ck_point :bool,
            model_path :str,
            tableFile :str='./table/wordVec_table.csv',
            dataset :str='small',
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
    if dataset=='small':
        short_cp_path = os.path.abspath("./data/cmn.txt")
        en2ch_dataset = s2s_shortCorpus(file_path=short_cp_path)
    elif dataset=='big':
        en_data_path = os.path.abspath("./data/zh-en.en")
        ch_data_path = os.path.abspath("./data/zh-en.zh")
        en2ch_dataset = s2s_dataset(en_corpus=en_data_path, ch_corpus=ch_data_path, dataNum=train_data_num)
    else:
        print('Invalid dataset!')
        return


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
    path = os.path.join('./experiment/', model_path)
    if from_ck_point:  
        check_point = torch.load(path, map_location=device)
        model.load_state_dict(check_point['model_state_dict'])
        BEST_LOSS = check_point['loss'] 

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.92)
    if optimizer__=='SGD':
        pass
    elif optimizer__=='ck_point':
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.92)
            optimizer.load_state_dict(check_point['optimizer_state_dict'])

    model.train() # set the model to training mode
    for epoch in range(epochs):
        epoch_loss = 0
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

        epoch_loss /= (len(train_dataloader))
        print('Loss : '+str(epoch_loss))

        if epoch_loss<BEST_LOSS:
            # save the model
            BEST_LOSS = epoch_loss
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            }, path)


if __name__=='__main__':
    """
    python3 train.py [model name] [dataset] [train_dataNum]
        * dataset:
            * small
            * big   
        * train_dataNum:
            * -1 means all 
            * the size of this 'big' dataset is about 240000
            * the size of this 'small' dataset is about 24360
    """
    device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = 'ck_point'
    teacher_forcing_ratio = 0.7
    batchSize = 8
    epochs_ = 30
    clipping = 1
    learn_r = 0.007 
    modelPath = sys.argv[1]
    table_file = './table/wordVec_table.csv'
    dataset_ = sys.argv[2]
    train_dataNum = int(sys.argv[3])
    from_checkpoint = True

    train(
        from_ck_point=from_checkpoint,
        model_path=modelPath,
        tableFile=table_file,
        dataset=dataset_,
        train_data_num=train_dataNum,
        optimizer__=optimizer,
        teacher_force_ratio=teacher_forcing_ratio,
        batch_size=batchSize,
        epochs=epochs_,
        clip=clipping,
        learning_rate=learn_r,
        device=device_
    )