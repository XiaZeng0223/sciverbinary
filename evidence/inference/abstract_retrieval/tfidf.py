import argparse
import jsonlines
import numpy as np
from statistics import mean, median
from sklearn.feature_extraction.text import TfidfVectorizer

parser = argparse.ArgumentParser()
parser.add_argument('--testing', type=bool, default=False)
parser.add_argument('--min-gram', type=int, default=1)
parser.add_argument('--max-gram', type=int, default=2)
parser.add_argument('--corpus', type=str, default='/home/zeng/allenai/scifact/data/corpus.jsonl')
parser.add_argument('--dataset', type=str, default='/home/zeng/allenai/scifact/data/claims_dev.jsonl')
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--output_dir', type=str, default='/home/zeng/allenai/scifact/test_abstract_tfidf')
parser.add_argument('--output', type=str, default='/home/zeng/allenai/scifact/prediction/abstract_retrieval.jsonl')

args = parser.parse_args()
if args.testing:
    # k = args.k
    for k in [1, 2, 3, 4, 5, 10, 20, 30, 50, 100]:
        corpus = list(jsonlines.open(args.corpus))
        dataset = list(jsonlines.open(args.dataset))
        output = jsonlines.open('{}/abstract_retrieval_tfidf_top_{}.jsonl'.format(args.output, k), 'w')

        vectorizer = TfidfVectorizer(stop_words='english',
                                     ngram_range=(args.min_gram, args.max_gram))

        doc_vectors = vectorizer.fit_transform([doc['title'] + ' ' + ' '.join(doc['abstract'])
                                                for doc in corpus])

        doc_ranks = []
        correct_retrival=0
        yes_info=0
        no_info = 0
        ground_truth_counts = 0
        for data in dataset:
            if data['evidence'] != {}:
                yes_info+=1

                claim = data['claim']
                claim_vector = vectorizer.transform([claim]).todense()
                doc_scores = np.asarray(doc_vectors @ claim_vector.T).squeeze()
                doc_indices_rank = doc_scores.argsort()[::-1].tolist()
                doc_id_rank = [corpus[idx]['doc_id'] for idx in doc_indices_rank][:k]

                data['retrieved_doc_ids'] = doc_id_rank
                output.write(data)

                truth = list(data['evidence'].keys())
                ground_truth_counts+=len(truth)
                # print('ground truth: {}'.format(truth))
                # print('predictions: {} '.format(doc_id_rank))
                for prediction in doc_id_rank:
                    if str(prediction) in truth:
                        correct_retrival+=1
                        # print('correct prediction of doc {} for claim {}'.format(prediction, data['id']))

            else:
                no_info+=1
        print(k)
        print('{} correct abstracts retrived'.format(correct_retrival))
        print('{} claims with retrivable abstract, {} without evidence'.format(yes_info, no_info))
        print('Ground truth counts {}'.format(ground_truth_counts))
else:
    k = args.k
    corpus = list(jsonlines.open(args.corpus))
    dataset = list(jsonlines.open(args.dataset))
    output = jsonlines.open('{}/abstract_retrieval_tfidf_top_{}.jsonl'.format(args.output, k), 'w')

    vectorizer = TfidfVectorizer(stop_words='english',
                                 ngram_range=(args.min_gram, args.max_gram))

    doc_vectors = vectorizer.fit_transform([doc['title'] + ' ' + ' '.join(doc['abstract'])
                                            for doc in corpus])

    doc_ranks = []
    correct_retrival = 0
    yes_info = 0
    no_info = 0
    ground_truth_counts = 0
    for data in dataset:
        if data['evidence'] != {}:
            yes_info += 1

            claim = data['claim']
            claim_vector = vectorizer.transform([claim]).todense()
            doc_scores = np.asarray(doc_vectors @ claim_vector.T).squeeze()
            doc_indices_rank = doc_scores.argsort()[::-1].tolist()
            doc_id_rank = [corpus[idx]['doc_id'] for idx in doc_indices_rank][:k]


            data['retrieved_doc_ids'] = doc_id_rank
            output.write(data)

            truth = list(data['evidence'].keys())
            ground_truth_counts += len(truth)
            # print('ground truth: {}'.format(truth))
            # print('predictions: {} '.format(doc_id_rank))
            for prediction in doc_id_rank:
                if str(prediction) in truth:
                    correct_retrival += 1
                    # print('correct prediction of doc {} for claim {}'.format(prediction, data['id']))

        else:
            no_info += 1
    print(k)
    print('{} correct abstracts retrived'.format(correct_retrival))
    print('{} claims with retrivable abstract, {} without evidence'.format(yes_info, no_info))
    print('Ground truth counts {}'.format(ground_truth_counts))





