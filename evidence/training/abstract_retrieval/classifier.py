import argparse
import torch
import jsonlines
import os
import numpy as np

from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, balanced_accuracy_score


parser = argparse.ArgumentParser()
parser.add_argument('--corpus', type=str, default='./data/corpus.jsonl')
parser.add_argument('--claim-train', type=str, default='./data/claims_train.jsonl')
parser.add_argument('--claim-dev', type=str, default='./data/claims_dev.jsonl')
parser.add_argument('--dest_dir', type=str, default='./data_abstract_training')
parser.add_argument('--model', type=str, default='/home/zeng/model/pre/biobert-base')
parser.add_argument('--epochs', type=int, default=11)
parser.add_argument('--batch-size-gpu', type=int, default=16, help='The batch size to send through GPU')
parser.add_argument('--batch-size-accumulated', type=int, default=128, help='The batch size for each gradient update')
parser.add_argument('--lr-base', type=float, default=1e-5)
parser.add_argument('--lr-linear', type=float, default=1e-3)
parser.add_argument('--model-base', type=str, default='bert')
parser.add_argument('--k', type=int, default=10)


args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device "{device}"')


class SciFactAbstractDataset(Dataset):
    def __init__(self, corpus: str, dataset: str, vectorizer, doc_vectors):
        self.samples = []
        all_doc_ids = tfidf_scope(vectorizer, doc_vectors, dataset, k=30)
        corpus = {doc['doc_id']: doc for doc in jsonlines.open(corpus)}

        for i, data in enumerate(jsonlines.open(dataset)):
            scope = all_doc_ids[i]
            abstract_ids = [int(doc_id) for doc_id in list(data['evidence'].keys()) if int(doc_id) in scope]
            cited_ids = [int(doc_id) for doc_id in data['cited_doc_ids'] if doc_id in scope]
            non_abstract_ids = set(cited_ids) - set(abstract_ids)

            # add positive samples
            for doc_id in abstract_ids:
                doc = corpus[doc_id]
                title = doc['title']
                self.samples.append({
                    'claim': data['claim'],
                    'title': title,
                    'evidence': 1
                })
            # add negative samples
            if len(non_abstract_ids) > 0:
                for doc_id in non_abstract_ids:
                    doc = corpus[doc_id]
                    title = doc['title']
                    self.samples.append({
                        'claim': data['claim'],
                        'title': title,
                        'evidence': 0
                    })

            # use up the rest of top 30 tfidf retrieval as negative samples
            rest = [doc_id for doc_id in scope if doc_id not in cited_ids]
            for doc_id in rest:
                doc = corpus[doc_id]
                title = doc['title']
                self.samples.append({
                    'claim': data['claim'],
                    'title': title,
                    'evidence': 0
                })


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def tfidf_scope(vectorizer, doc_vectors, dataset, k=30):
    corpus = list(jsonlines.open(args.corpus))
    doc_id_ranks = []
    for data in jsonlines.open(dataset):
        claim = data['claim']
        claim_vector = vectorizer.transform([claim]).todense()
        doc_scores = np.asarray(doc_vectors @ claim_vector.T).squeeze()
        doc_indices_rank = doc_scores.argsort()[::-1].tolist()[:k]
        doc_id_rank = [corpus[idx]['doc_id'] for idx in doc_indices_rank]
        doc_id_ranks.append(doc_id_rank)

    return doc_id_ranks


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
            encoded_dict = encode(batch['claim'], batch['title'])
            logits = model(**encoded_dict)[0]
            targets.extend(batch['evidence'].long().tolist())
            outputs.extend(logits.argmax(dim=1).long().tolist())

    return f1_score(targets, outputs, zero_division=0),\
            precision_score(targets, outputs, zero_division=0),\
            recall_score(targets, outputs, zero_division=0)

if __name__ == '__main__':
    vectorizer = TfidfVectorizer(stop_words='english',
                                 ngram_range=(1, 2))
    doc_vectors = vectorizer.fit_transform([doc['title'] + ' ' + ' '.join(doc['abstract'])
                                            for doc in list(jsonlines.open(args.corpus))])


    trainset = SciFactAbstractDataset(args.corpus, args.claim_train, vectorizer, doc_vectors)
    devset = SciFactAbstractDataset(args.corpus, args.claim_dev, vectorizer, doc_vectors)


    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)
    # model base are either bert or roberta
    if args.model_base=='bert':
        optimizer = torch.optim.Adam([
            {'params': model.bert.parameters(), 'lr': args.lr_base},
            {'params': model.classifier.parameters(), 'lr': args.lr_linear}
        ])
    elif args.model_base=='roberta':
        optimizer = torch.optim.Adam([
            {'params': model.roberta.parameters(), 'lr': args.lr_base},
            {'params': model.classifier.parameters(), 'lr': args.lr_linear}
        ])

    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, 20)

    for e in range(args.epochs):
        model.train()
        t = tqdm(DataLoader(trainset, batch_size=args.batch_size_gpu, shuffle=True))
        for i, batch in enumerate(t):
            encoded_dict = encode(batch['claim'], batch['title'])
            loss, logits = model(**encoded_dict, labels=batch['evidence'].long().to(device))

            loss.backward()
            if (i + 1) % (args.batch_size_accumulated // args.batch_size_gpu) == 0:
                optimizer.step()
                optimizer.zero_grad()
                t.set_description(f'Epoch {e}, iter {i}, loss: {round(loss.item(), 4)}')
        scheduler.step()
        train_score = evaluate(model, trainset)
        print(f'Epoch {e}, train f1: %.4f, precision: %.4f, recall: %.4f' % train_score)

        dev_score = evaluate(model, devset)
        print(f'Epoch {e}, dev f1: %.4f, precision: %.4f, recall: %.4f' % dev_score)
        # Save
        save_path = os.path.join('{}/abstract_training'.format(args.dest_dir), f'epoch-{e}-f1-{int(dev_score[0] * 1e4)}')
        os.makedirs(save_path)
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)
