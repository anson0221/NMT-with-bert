import torch
from NMTmodel import sequence2sequence as s2s
from dataset import sentencesVec
import sys
import os


def make_inference(model :s2s, converter :sentencesVec, input_sent :str)->str:
    sent = converter.get_sentenceVec(input_sent) # (seq_len, hidden_dim)
    sent = sent.unsqueeze(0) # (1, seq_len, hidden_dim)
    with torch.no_grad():
        out = model(input_tensor=sent)
    return out

if __name__=='__main__':
    path = os.path.join('./experiment/', sys.argv[1])
    input_bert = 'bert-base-uncased'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = s2s(
                input_size=768,
                hidden_size=768,
                batch_size=1 # one sentence at a time
                )
    check_point = torch.load(path, map_location=device)
    model.load_state_dict(check_point['model_state_dict'])
    model.eval()

    cvtr = sentencesVec(bert=input_bert)
    while True:
        print('Please input a sentence in English.')
        sent = input()
        print(make_inference(model=model, converter=cvtr, input_sent=sent))