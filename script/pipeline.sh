#!/bin/bash
#
#
# Usgae: bash script/pipeline.sh [abstract] [rationale] [label] [dataset] [model_dir] [data_dir]
# [abstract] options: "oracle", "tfidf", "bertscore", "classifier"
# [rationale] options: "oracle", "baseline", "classifier"
# [label] options: "baseline", "classifier"
# [dataset] options: "dev", "test"
# [model_dir]: dir to your trained models, e.g. ./model/biobertb
# [data_dir]: dir to your downloaded data, e.g. ./data

abstract = $1
rationale = $2
label = $3
dataset = $4
model_dir = $5
data_dir = $6

# Please set the following path to your local path to conda.sh
# This is set to use different virtual environments for different components
source /home/zeng/anaconda3/etc/profile.d/conda.sh

echo "Running pipeline on ${dataset} set."

####################

# Create a prediction folder to store results.
rm -rf prediction
mkdir -p prediction

###################
# Run abstract retrieval.
echo; echo "Retrieving abstracts."
if [ $retrieval == "oracle" ]
then
    python3 evidence/inference/abstract_retrieval/oracle.py \
        --dataset ${data_dir}/claims_${dataset}.jsonl \
        --output prediction/abstract_retrieval.jsonl
elif [ $retrieval == "tfidf" ]
then
    python3 evidence/inference/abstract_retrieval/tfidf.py \
        --corpus ${data_dir}/corpus.jsonl \
        --dataset ${data_dir}/claims_${dataset}.jsonl \
        --k ${k} \
        --min-gram 1 \
        --max-gram 2 \
        --output prediction/abstract_retrieval.jsonl
elif [ $retrieval == "bertscore" ]
then
    conda activate two-step
    python3 evidence/inference/abstract_retrieval/bertscore.py \
        --corpus ${data_dir}/corpus.jsonl \
        --dataset ${data_dir}/claims_${dataset}.jsonl \
        --k ${k} \
        --min-gram 1 \
        --max-gram 2 \
        --output prediction/abstract_retrieval.jsonl \
fi
elif [ $retrieval == "classifier" ]
then
    conda activate evidence
    python3 evidence/inference/abstract_retrieval/classifier.py \
        --corpus ${data_dir}/corpus.jsonl \
        --dataset ${data_dir}/claims_${dataset}.jsonl \
        --model ${model_dir}/abstract \
        --output prediction/abstract_retrieval.jsonl
fi


######################
conda deactivate
conda activate evidence
# Run rationale selection
echo; echo "Selecting rationales."
if [ $model == "oracle" ]
then
    python3 evidence/inference/rationale_selection/oracle.py \
        --dataset ${data_dir}/claims_${dataset}.jsonl \
        --abstract-retrieval prediction/abstract_retrieval.jsonl \
        --output prediction/rationale_selection.jsonl
elif [ $model == "baseline" ]
then
    python3 evidence/inference/rationale_selection/transformer.py \
        --corpus ${data_dir}/corpus.jsonl \
        --dataset ${data_dir}/claims_${dataset}.jsonl \
        --abstract-retrieval prediction/abstract_retrieval.jsonl \
        --model "$(dirname "${model_dir}")/rationale_roberta_large_scifact" \
        --threshold 0.5 \
        --output prediction/rationale_selection.jsonl
elif [ $model == "classifier" ]
then
    python3 evidence/inference/rationale_selection/classifier.py \
        --corpus ${data_dir}/corpus.jsonl \
        --dataset ${data_dir}/claims_${dataset}.jsonl \
        --abstract prediction/abstract_retrieval.jsonl \
        --model ${model_dir}/rationale_continue \
        --output prediction/rationale_selection.jsonl
fi
####################

#conda deactivate
# Run label prediction, using the selected rationales.
echo; echo "Predicting labels."
if [ $model == "baseline" ]
then
    conda activate evidence
    python3 label/inference/label_prediction/baseline.py \
      --corpus ${data_dir}/corpus.jsonl \
      --dataset ${data_dir}/claims_${dataset}.jsonl \
      --rationale-selection prediction/rationale_selection.jsonl \
      --model "$(dirname "${model_dir}")/label_roberta_large_fever_scifact" \
      --output prediction/label_prediction.jsonl
elif [ $model == "classifier" ]
then
    conda activate two-step
    python3 label/inference/label_prediction/two-step.py \
        --corpus ${data_dir}/corpus.jsonl \
        --dataset ${data_dir}/claims_${dataset}.jsonl \
        --rationale-selection prediction/rationale_selection.jsonl \
        --model_n ${model_dir}/label/neutral_detector \
        --model_s ${model_dir}/label/entail_detector \
        --output prediction/label_prediction.jsonl
fi
####################

# Merge rationale and label predictions.
echo; echo "Merging predictions."

python3 label/inference/merge_predictions.py \
    --rationale-file prediction/rationale_selection.jsonl \
    --label-file prediction/label_prediction.jsonl \
    --result-file prediction/merged_predictions.jsonl

####################


# Evaluate final predictions
echo; echo "Evaluating."
python3 label/evaluate/pipeline.py \
    --gold ${data_dir}/claims_${dataset}.jsonl \
    --corpus ${data_dir}/corpus.jsonl \
    --prediction prediction/merged_predictions.jsonl \
    --output prediction/results.csv
