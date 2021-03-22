import argparse
import torch
import jsonlines
import os

from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
from typing import List
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, balanced_accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument('--corpus', type=str, default='./data/corpus.jsonl')
parser.add_argument('--claim_train', type=str, default='./data/claims_train.jsonl')
parser.add_argument('--claim_dev', type=str, default='./data/claims_dev.jsonl')
parser.add_argument('--abstract_train', type=str, default='./data/abstract_retrieval_train.jsonl')
parser.add_argument('--abstract_dev', type=str, default='./data/abstract_retrieval_dev.jsonl')
parser.add_argument('--dest', type=str, default='./data_rationale_training')
parser.add_argument('--model', type=str, default='/home/zeng/model/pre/biobert-base')
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--batch-size-gpu', type=int, default=8, help='The batch size to send through GPU')
parser.add_argument('--batch-size-accumulated', type=int, default=128, help='The batch size for each gradient update')
parser.add_argument('--lr-base', type=float, default=5e-5)
parser.add_argument('--lr-linear', type=float, default=1e-3)
parser.add_argument('--model_base', type=str, default='bert')


args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device "{device}"')


class SciFactRationaleSelectionDataset(Dataset):
    def __init__(self, corpus: str, dataset: str, abstracts):
        self.samples = []
        abstract_retrieval = jsonlines.open(abstracts)
        dataset = jsonlines.open(dataset)
        corpus = {doc['doc_id']: doc for doc in jsonlines.open(corpus)}
        for data, retrieval in tqdm(list(zip(dataset, abstract_retrieval))):
            # assert data['id'] == retrieval['claim_id']
            assert data['id'] == retrieval['id']

            # for doc_id in retrieval['doc_ids']:
            for doc_id in retrieval['retrieved_doc_ids']:
                doc_id = str(doc_id)
                doc = corpus[int(doc_id)]
                #if the doc is correctly retrieved
                if doc_id in list(data['evidence'].keys()):
                    evidence_sentence_idx = {s for es in data['evidence'][doc_id] for s in es['sentences']}
                    # print(evidence_sentence_idx)
                #if not
                else:
                    evidence_sentence_idx = {}

                for i, sentence in enumerate(doc['abstract']):
                    self.samples.append({
                        'claim': data['claim'],
                        'sentence': sentence,
                        'evidence': i in evidence_sentence_idx
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def encode(claims: List[str], sentences: List[str]):
    encoded_dict = tokenizer.batch_encode_plus(
        zip(sentences, claims),
        pad_to_max_length=True,
        return_tensors='pt')
    if encoded_dict['input_ids'].size(1) > 512:
        # Too long for the model. Truncate it
        encoded_dict = tokenizer.batch_encode_plus(
            zip(sentences, claims),
            max_length=512,
            truncation_strategy='only_first',
            pad_to_max_length=True,
            return_tensors='pt')
    encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
    return encoded_dict

def evaluate(model, dataset):
    model.eval()
    targets = []
    outputs = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=args.batch_size_gpu):
            encoded_dict = encode(batch['claim'], batch['sentence'])
            logits = model(**encoded_dict)[0]
            targets.extend(batch['evidence'].float().tolist())
            outputs.extend(logits.argmax(dim=1).tolist())
    return f1_score(targets, outputs, zero_division=0),\
           precision_score(targets, outputs, zero_division=0),\
           recall_score(targets, outputs, zero_division=0), \
           accuracy_score(targets, outputs), \
           balanced_accuracy_score(targets, outputs)

if __name__ == '__main__':

    trainset = SciFactRationaleSelectionDataset(args.corpus, args.claim_train, args.abstract_train)
    devset = SciFactRationaleSelectionDataset(args.corpus, args.claim_dev, args.abstract_dev)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)
    # model base are either bert or roberta
    if args.model_base == 'bert':
        optimizer = torch.optim.Adam([
            {'params': model.bert.parameters(), 'lr': args.lr_base},
            {'params': model.classifier.parameters(), 'lr': args.lr_linear}
        ])
    elif args.model_base == 'roberta':
        optimizer = torch.optim.Adam([
            {'params': model.roberta.parameters(), 'lr': args.lr_base},
            {'params': model.classifier.parameters(), 'lr': args.lr_linear}
        ])

    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, 128)

    # training
    for e in range(args.epochs):
        model.train()
        t = tqdm(DataLoader(trainset, batch_size=args.batch_size_gpu, shuffle=True))
        for i, batch in enumerate(t):
            encoded_dict = encode(batch['claim'], batch['sentence'])
            loss, logits = model(**encoded_dict, labels=batch['evidence'].long().to(device))
            loss.backward()
            if (i + 1) % (args.batch_size_accumulated // args.batch_size_gpu) == 0:
                optimizer.step()
                optimizer.zero_grad()
                t.set_description(f'Epoch {e}, iter {i}, loss: {round(loss.item(), 4)}')
        scheduler.step()
        train_score = evaluate(model, trainset)
        print(f'Epoch {e}, train f1: %.4f, precision: %.4f, recall: %.4f, acc: %.4f, balanced_acc: %.4f' % train_score)
        dev_score = evaluate(model, devset)
        print(f'Epoch {e}, dev f1: %.4f, precision: %.4f, recall: %.4f, acc: %.4f, balanced_acc: %.4f' % dev_score)
        # Save
        save_path = os.path.join(args.dest, f'epoch-{e}-f1-{int(dev_score[0] * 1e4)}')
        os.makedirs(save_path)
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)
