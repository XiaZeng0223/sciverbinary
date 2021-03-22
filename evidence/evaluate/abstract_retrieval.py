

"""
Computes the abstract retrieval F1 as in the paper.
"""


import argparse
import jsonlines
from collections import Counter

from lib.metrics import compute_f1


parser = argparse.ArgumentParser()
# parser.add_argument('--corpus', type=str, default='./data/corpus.jsonl')
# parser.add_argument('--dataset', type=str, default='./data/claims_dev.jsonl')
parser.add_argument('--abstract-retrieval', type=str, default='/home/zeng/allenai/scifact/test_abstract_bertscore/abstract_retrieval_P.jsonl')
args = parser.parse_args()

# corpus = {doc['doc_id']: doc for doc in jsonlines.open(args.corpus)}
# dataset = jsonlines.open(args.dataset)
abstract_retrieval = jsonlines.open(args.abstract_retrieval)

counts = Counter()

for retrieval in abstract_retrieval:
    true_doc_ids = set(map(int, retrieval["evidence"].keys()))

    counts["relevant"] += len(true_doc_ids)

    for pred_doc_id in retrieval["retrieved_doc_ids"]:
        counts["retrieved"] += 1
        if pred_doc_id in true_doc_ids:
            counts["correct"] += 1

f1 = compute_f1(counts)
print(f1)
