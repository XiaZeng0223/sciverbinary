'''
Evaluate full-pipeline predictions. Example usage:

python verisci/evaluate/pipeline.py \
    --gold data/claims_dev.jsonl \
    --corpus data/corpus.jsonl \
    --prediction prediction/merged_predictions.jsonl \
    --output predictions/metrics.json
'''

import argparse
import pandas as pd

from lib.data import GoldDataset, PredictedDataset
from lib import metrics


def get_args():
    parser = argparse.ArgumentParser(
        description='Evaluate SciFact predictions.'
    )
    parser.add_argument('--gold', type=str, required=True,
                        help='The gold labels.')
    parser.add_argument('--corpus', type=str, required=True,
                        help='The corpus of documents.')
    parser.add_argument('--prediction', type=str, required=True,
                        help='The predictions.')
    parser.add_argument('--output', type=str, default=None,
                        help='If provided, save metrics to this file.')
    args = parser.parse_args()
    return args


def main():
    pd.set_option('display.max_columns', None)

    args = get_args()

    data = GoldDataset(args.corpus, args.gold)
    predictions = PredictedDataset(data, args.prediction)

    res = metrics.compute_metrics(predictions)
    print(res)
    if args.output is not None:
        res.to_csv(args.output, sep='\t')



if __name__ == "__main__":
    main()
