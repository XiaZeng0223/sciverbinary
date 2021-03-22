import argparse
import torch
import jsonlines
import random
import os

from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
from tqdm import tqdm
from typing import List
from sklearn.metrics import f1_score, precision_score, recall_score

parser = argparse.ArgumentParser()
parser.add_argument('--corpus', type=str, default='./data/corpus.jsonl')
parser.add_argument('--train', type=str, default='./data/claims_train.jsonl')
parser.add_argument('--dev', type=str, default='./data/claims_dev.jsonl')
parser.add_argument('--dest', type=str, default='./model/test')
parser.add_argument('--model', type=str, default='prajjwal1/bert-tiny')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch-size-gpu', type=int, default=10, help='The batch size to send through GPU')
parser.add_argument('--batch-size-accumulated', type=int, default=128, help='The batch size for each gradient update')
parser.add_argument('--lr-base', type=float, default=5e-5)
parser.add_argument('--lr-linear', type=float, default=1e-4)
parser.add_argument('--model_base', type=str, default='bert')
parser.add_argument('--warmup_steps', type=int, default=128)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device "{device}"')


class LabelPredictionDataset(Dataset):
    def __init__(self, corpus: str, claims: str):
        self.samples = []

        corpus = {doc['doc_id']: doc for doc in jsonlines.open(corpus)}
        # three-way classification
        # label_encodings = {'CONTRADICT': 0, 'NOT_ENOUGH_INFO': 1, 'SUPPORT': 2}

        # binary classification: merge CONTRADICT and SUPPORT as 0, NOT_ENOUGH_INFO as 1.
        # unknown detection: detecting the truth value of 'unknown'/'NOT_ENOUGH_INFO
        label_encodings = {'CONTRADICT': 0, 'NOT_ENOUGH_INFO': 1, 'SUPPORT': 0}


        for claim in jsonlines.open(claims):
            if claim['evidence']:
                for doc_id, evidence_sets in claim['evidence'].items():
                    doc = corpus[int(doc_id)]

                    # Add individual evidence set as samples:
                    for evidence_set in evidence_sets:
                        rationale = [doc['abstract'][i].strip() for i in evidence_set['sentences']]
                        self.samples.append({
                            'claim': claim['claim'],
                            'rationale': ' '.join(rationale),
                            'label': label_encodings[evidence_set['label']]
                        })

                    # Add all evidence sets as positive samples
                    rationale_idx = {s for es in evidence_sets for s in es['sentences']}
                    rationale_sentences = [doc['abstract'][i].strip() for i in sorted(list(rationale_idx))]
                    self.samples.append({
                        'claim': claim['claim'],
                        'rationale': ' '.join(rationale_sentences),
                        'label': label_encodings[evidence_sets[0]['label']]  # directly use the first evidence set label
                        # because currently all evidence sets have
                        # the same label
                    })

                    # Add negative samples
                    non_rationale_idx = set(range(len(doc['abstract']))) - rationale_idx
                    non_rationale_idx = random.sample(non_rationale_idx,
                                                      k=min(random.randint(1, 2), len(non_rationale_idx)))
                    non_rationale_sentences = [doc['abstract'][i].strip() for i in sorted(list(non_rationale_idx))]
                    self.samples.append({
                        'claim': claim['claim'],
                        'rationale': ' '.join(non_rationale_sentences),
                        'label': label_encodings['NOT_ENOUGH_INFO']
                    })
            else:
                # Add negative samples
                for doc_id in claim['cited_doc_ids']:
                    doc = corpus[int(doc_id)]
                    non_rationale_idx = random.sample(range(len(doc['abstract'])), k=random.randint(1, 2))
                    non_rationale_sentences = [doc['abstract'][i].strip() for i in non_rationale_idx]
                    self.samples.append({
                        'claim': claim['claim'],
                        'rationale': ' '.join(non_rationale_sentences),
                        'label': label_encodings['NOT_ENOUGH_INFO']
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]





def encode(claim: List[str], rationale: List[str]):
    encoding = tokenizer(claim, rationale, padding=True, truncation=True, max_length=512, return_tensors="pt")
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    return input_ids, attention_mask


def evaluate(model, dataset):
    model.eval()
    targets = []
    outputs = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=args.batch_size_gpu):
            input_ids, attention_mask = encode(batch['claim'], batch['rationale'])
            logits = model(input_ids.to(device)).logits
            targets.extend(batch['label'].float().tolist())
            outputs.extend(logits.argmax(dim=1).tolist())
    return {
        'macro_f1': f1_score(targets, outputs, zero_division=0, average='macro'),
        'f1': tuple(f1_score(targets, outputs, zero_division=0, average=None)),
        'precision': tuple(precision_score(targets, outputs, zero_division=0, average=None)),
        'recall': tuple(recall_score(targets, outputs, zero_division=0, average=None))
    }


if __name__ == '__main__':

    trainset = LabelPredictionDataset(args.corpus, args.train)
    devset = LabelPredictionDataset(args.corpus, args.dev)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    config = AutoConfig.from_pretrained(args.model, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, config=config).to(device)
    # model base may be distilbert, bert or roberta, or other model bases from huggingface library, see https://huggingface.co/transformers/pretrained_models.html
    if args.model_base == 'distilbert':
        optimizer = torch.optim.Adam([
            {'params': model.distilbert.parameters(), 'lr': args.lr_base},
            {'params': model.classifier.parameters(), 'lr': args.lr_linear}
        ])
    elif args.model_base == 'bert':
        optimizer = torch.optim.Adam([
            {'params': model.bert.parameters(), 'lr': args.lr_base},
            {'params': model.classifier.parameters(), 'lr': args.lr_linear}
        ])
    elif args.model_base == 'roberta':
        optimizer = torch.optim.Adam([
            {'params': model.roberta.parameters(), 'lr': args.lr_base},
            {'params': model.classifier.parameters(), 'lr': args.lr_linear}
        ])

    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.epochs * args.batch_size_accumulated)

    for e in range(args.epochs):
        model.train()
        t = tqdm(DataLoader(trainset, batch_size=args.batch_size_gpu, shuffle=True))
        for i, batch in enumerate(t):
            input_ids, attention_mask = encode(batch['claim'], batch['rationale'])
            outputs = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device), labels=batch['label'].long().to(device))
            loss = outputs.loss
            loss.backward()
            if (i + 1) % (args.batch_size_accumulated // args.batch_size_gpu) == 0:
                optimizer.step()
                optimizer.zero_grad()
                t.set_description(f'Epoch {e}, iter {i}, loss: {round(loss.item(), 4)}')
        scheduler.step()

        # Eval
        train_score = evaluate(model, trainset)
        print(f'Epoch {e} train score:')
        print(train_score)
        dev_score = evaluate(model, devset)
        print(f'Epoch {e} dev score:')
        print(dev_score)

        # Save
        save_path = os.path.join(args.dest, f'epoch-{e}-f1-{int(dev_score["macro_f1"] * 1e4)}')
        os.makedirs(save_path)
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)