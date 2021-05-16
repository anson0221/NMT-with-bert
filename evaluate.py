from nltk.translate.bleu_score import sentence_bleu
from transformers import BertTokenizer
from dataset import  s2s_dataset, sentencesVec
import torch
from NMTmodel import sequence2sequence as s2s
from dataset import sentencesVec
import sys
import os


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num = sys.argv[2]
    dataset = s2s_dataset(en_corpus='./data/zh_en.en', ch_corpus='./data/zh_en.zh', dataNum=num)

    input_bert = 'bert-base-uncased'
    source_cvtr = sentencesVec(bert=input_bert)

    output_bert = 'bert-base-chinese'
    target_tknzr = BertTokenizer.from_pretrained(output_bert)

    model_path = os.path.join('./experiment/', sys.argv[1])
    model = s2s(
                input_size=768,
                hidden_size=768,
                batch_size=1 # one sentence at a time
                )
    check_point = torch.load(model_path, map_location=device)
    model.load_state_dict(check_point['model_state_dict'])
    model.eval()

    score = 0.0
    for i in range(len(dataset)):
        with torch.no_grad():
            # prediction : List[str]
            prediction = model(source_cvtr.get_sentenceVec(dataset.df_en[i]).unsqueeze(0))

        score += sentence_bleu(
            references=[target_tknzr.encode(dataset.df_ch[i])],
            hypothesis=prediction,
            weights=(0.2, 0.2, 0.2, 0.2, 0.2)
        )
    score /= len(dataset)

    print('average BLEU score : '+str(score))