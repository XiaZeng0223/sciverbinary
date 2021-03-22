import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import argparse
import jsonlines
import numpy as np
import logging
import transformers
transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)
from sklearn.feature_extraction.text import TfidfVectorizer


parser = argparse.ArgumentParser()
parser.add_argument('--corpus', type=str, default='/home/zeng/allenai/scifact/data/corpus.jsonl')
parser.add_argument('--dataset', type=str, default='/home/zeng/allenai/scifact/data/claims_dev.jsonl')
parser.add_argument('--model', type=str, default='/home/zeng/allenai/scifact/abstract_retrival_training/epoch-19-f1-8813')
parser.add_argument('--output', type=str, default='/home/zeng/allenai/scifact/test_abstract_transformers/abstract_retrieval.jsonl')
parser.add_argument('--doc_section', type=str, default='title', choices=['both', 'title', 'abstract'])
parser.add_argument('--batch-size-gpu', type=int, default=30, help='The batch size to send through GPU')



args = parser.parse_args()

class SciFactDataset(Dataset):
    def __init__(self, corpus: str, dataset: str, vectorizer, doc_vectors):
        self.samples = []
        all_doc_ids = tfidf_scope(vectorizer, doc_vectors, dataset, k=30)
        corpus = {doc['doc_id']: doc for doc in jsonlines.open(corpus)}

        for i, data in enumerate(jsonlines.open(dataset)):
            scope = all_doc_ids[i]
            # add testing samples
            for doc_id in scope:
                title = corpus[doc_id]['title']
                self.samples.append({
                    'id': data['id'],
                    'claim': data['claim'],
                    'title': title
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

def encode(claims, sentences):
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

def predict(model, dataset):
    all_doc_ids = tfidf_scope(vectorizer, doc_vectors, args.dataset, k=30)
    model.eval()
    preds = []
    with torch.no_grad():
        for i, batch in enumerate(DataLoader(dataset, batch_size=args.batch_size_gpu)):
            # print(i)
            # print("batch['claim']", batch['claim'])
            # print("batch['title]", batch['title'])
            encoded_dict = encode(batch['claim'], batch['title'])
            logits = model(**encoded_dict)[0]
            output = logits.argmax(dim=1).long().tolist()
            # print(output)
            if 1 in output:
                indices = np.argwhere(np.array(output) == 1)
                preds.append([all_doc_ids[i][indice[0]] for indice in indices])
            else:
                preds.append([])
    # print(len(preds), preds[0])
    return preds


if __name__ == '__main__':
    output = jsonlines.open(args.output, 'w')
    vectorizer = TfidfVectorizer(stop_words='english',
                                 ngram_range=(1, 2))
    doc_vectors = vectorizer.fit_transform([doc['title'] + ' ' + ' '.join(doc['abstract'])
                                            for doc in list(jsonlines.open(args.corpus))])

    dataset = SciFactDataset(args.corpus, args.dataset, vectorizer, doc_vectors)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device "{device}"')

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device).eval()

    preds = predict(model, dataset)
    for i, data in enumerate(jsonlines.open(args.dataset)):
        data['retrieved_doc_ids'] = preds[i]
        output.write(data)

