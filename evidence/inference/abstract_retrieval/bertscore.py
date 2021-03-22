import argparse
import jsonlines
import numpy as np
from tqdm import tqdm
from bert_score import BERTScorer
import logging
import transformers
transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.parsing.preprocessing import remove_stopwords


parser = argparse.ArgumentParser()
parser.add_argument('--corpus', type=str, default='/home/zeng/allenai/scifact/data/corpus.jsonl')
parser.add_argument('--dataset', type=str, default='/home/zeng/allenai/scifact/data/claims_dev.jsonl')
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--threshold', type=float, default=0.55)
parser.add_argument('--model_type', type=str, default='/home/zeng/model/biobert-base')
parser.add_argument('--num_layers', type=int, default=12)
parser.add_argument('--output', type=str, default='/home/zeng/allenai/scifact/test_abstract_bertscore/abstract_retrieval.jsonl')
parser.add_argument('--min-gram', type=int, default=1)
parser.add_argument('--max-gram', type=int, default=2)
parser.add_argument('--doc_section', type=str, default='both', choices=['both', 'title', 'abstract'])
parser.add_argument('--remove_stopwords', type=bool, default=False)
parser.add_argument('--remove_claim_stopwords', type=bool, default=True)



args = parser.parse_args()

def tfidf(vectorizer, doc_vectors, data, k=30):

    saved = False

    claim = data['claim']
    claim_vector = vectorizer.transform([claim]).todense()
    doc_scores = np.asarray(doc_vectors @ claim_vector.T).squeeze()
    doc_indices_rank = doc_scores.argsort()[::-1].tolist()[:k]
    doc_id_rank = [corpus[idx]['doc_id'] for idx in doc_indices_rank]

    truth = list(data['evidence'].keys())
    # print('ground truth: {}'.format(truth))
    # print('reservations: {}'.format(doc_id_rank))
    for prediction in doc_id_rank:
        if str(prediction) in truth:
            # print('correct reservation of doc {} for claim {}'.format(prediction, data['id']))
            saved = True

    return doc_indices_rank, doc_id_rank, saved

if __name__ == '__main__':
    corpus = np.array(list(jsonlines.open(args.corpus)))
    dataset = np.array(list(jsonlines.open(args.dataset)))
    output = jsonlines.open(args.output, 'w')
    k = args.k
    doc_section = args.doc_section

    vectorizer = TfidfVectorizer(stop_words='english',
                                 ngram_range=(args.min_gram, args.max_gram))

    doc_vectors = vectorizer.fit_transform([doc['title'] + ' ' + ' '.join(doc['abstract'])
                                            for doc in corpus])


    if doc_section == 'both':
        docs = [[doc['title']] + doc['abstract'] for doc in corpus]
    elif doc_section =='title':
        docs = [doc['title'] for doc in corpus]
    elif doc_section == 'abstract':
        docs = [doc['abstract'] for doc in corpus]

    if args.remove_stopwords:
        print('removing stopwords')
        docs_removed = []
        for doc in docs:
            doc_removed = [remove_stopwords(sent) for sent in doc]
            docs_removed.append(doc_removed)
        docs = docs_removed

    remove_claim_stopwords = args.remove_claim_stopwords

    scorer = BERTScorer(model_type=args.model_type, num_layers=args.num_layers, rescale_with_baseline=False, device='cuda', batch_size=1, nthreads=16)


    i=0
    doc_ranks = []
    correct_retrival = 0
    correct_none_retrival = 0
    ground_truth_counts = 0
    ground_none_counts = 0
    for data in tqdm(dataset):
        if remove_claim_stopwords:
            claim = remove_stopwords(data['claim'])
        else:
            claim = data['claim']
        i+=1

        # use a tfidf filter to get the top 30 docs for target claim
        doc_indices, doc_id, saved = tfidf(vectorizer, doc_vectors, data)
        # print(type(doc_indices), len(doc_indices), doc_indices[0])

        # use a bertscorer to get the top 3 out of the top 30
        doc_indices_scores = []
        for doc_indice in doc_indices:
            P, R, F1 = scorer.score([claim], [docs[doc_indice]])
            doc_indices_scores.append(P)
        scores = np.array(doc_indices_scores)
        doc_indices_top_k = scores.argsort()[-k:][::-1]  #get the top k doc
        doc_indices_above_threthold = np.argwhere(scores > args.threshold)

        doc_id_rank = [doc_id[idx] for idx in doc_indices_top_k if [idx] in doc_indices_above_threthold]

        data['retrieved_doc_ids'] = doc_id_rank
        output.write(data)

        # if ground truth exits
        if data['evidence'] != {}:
            truth = list(data['evidence'].keys())
            ground_truth_counts += len(truth)

            for prediction in doc_id_rank:
                if str(prediction) in truth:
                    correct_retrival += 1

        # if ground truth is no evidence
        elif data['evidence'] == {}:
            ground_none_counts += 1
            if doc_id_rank == []:
                correct_none_retrival+=1



    print('{} correct abstracts retrived'.format(correct_retrival))
    print('Ground truth counts {}'.format(ground_truth_counts))
    print('{} correct none retrieval'.format(correct_none_retrival))
    print('grount none counts {}'.format(ground_none_counts))
