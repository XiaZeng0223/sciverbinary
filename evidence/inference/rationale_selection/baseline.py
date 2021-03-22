import argparse
import jsonlines
import numpy as np


import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--corpus', type=str, default='/home/zeng/allenai/scifact/data/corpus.jsonl')
parser.add_argument('--dataset', type=str, default='/home/zeng/allenai/scifact/data/claims_dev.jsonl')
parser.add_argument('--abstract-retrieval', type=str, default='/home/zeng/allenai/scifact/test_abstract_bertscore/abstract_retrieval.jsonl')
parser.add_argument('--model', type=str, default='/home/zeng/allenai/scifact/model/rationale_roberta_large_scifact')
parser.add_argument('--threshold', type=float, default=0.5, required=False)
parser.add_argument('--output', type=str, default='/home/zeng/allenai/scifact/test_rationale/rationale_selection.jsonl')
parser.add_argument('--only-rationale', action='store_true')

args = parser.parse_args()

def update_dict(key, value, d):
    if key not in d:
        d[key] = [value]
    else:
        d[key].append(value)
    return d

def top_k(keys, values, k):

    evidence = {}
    len_doc_0 = len(values[0])
    len_doc_1 = len(values[1])
    len_doc_2 = len(values[2])

    scores = np.concatenate(values)
    inds = [int(id) for id in np.argpartition(scores, -k)[-k:]]
    # print(inds)
    for ind in inds:
        if ind < len_doc_0:
            evidence = update_dict(keys[0], ind, evidence)
        elif ind < len_doc_0 + len_doc_1:
            evidence = update_dict(keys[1], ind - len_doc_0, evidence)
        elif ind < len_doc_0 + len_doc_1 + len_doc_2:
            evidence = update_dict(keys[2], ind - len_doc_0 - len_doc_1, evidence)
    # print(evidence)
    return evidence


def output_k(results, output_path, threshold=None, k=None):

    output = jsonlines.open(output_path, 'w')

    for result in results:
        evidence = {}
        if k and result['evidence_scores'] != {}:
            evidence = top_k(keys=list(result['evidence_scores'].keys()), values=list(result['evidence_scores'].values()), k=k)
        else:
            for doc_id, sentence_scores in result['evidence_scores'].items():
                doc_indices_sorted = sentence_scores.argsort()[::-1]
                doc_indices_above_threthold = np.argwhere(sentence_scores > threshold)
                evidence[doc_id] = [int(i) for i in doc_indices_sorted if [i] in doc_indices_above_threthold]

        output.write({
            'claim_id': result['claim_id'],
            'evidence': evidence
        })

if __name__ == '__main__':

    corpus = {doc['doc_id']: doc for doc in jsonlines.open(args.corpus)}
    dataset = jsonlines.open(args.dataset)
    abstract_retrieval = jsonlines.open(args.abstract_retrieval)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device "{device}"')

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device).eval()

    results = []

    with torch.no_grad():
        for data, retrieval in tqdm(list(zip(dataset, abstract_retrieval))):
            assert data['id'] == retrieval['claim_id']
            claim = data['claim']

            evidence_scores = {}
            for doc_id in retrieval['doc_ids']:
                doc = corpus[doc_id]
                sentences = doc['abstract']

                encoded_dict = tokenizer.batch_encode_plus(
                    zip(sentences, [claim] * len(sentences)) if not args.only_rationale else sentences,
                    pad_to_max_length=True,
                    return_tensors='pt'
                )
                encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
                sentence_scores = torch.softmax(model(**encoded_dict)[0], dim=1)[:, 1].detach().cpu().numpy()
                evidence_scores[doc_id] = sentence_scores
            results.append({
                'claim_id': retrieval['claim_id'],
                'evidence_scores': evidence_scores
            })



    output_k(results=results, output_path=args.output, threshold=args.threshold)

