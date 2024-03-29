# sciverbinary README

This repository presents team QMUL-SDS's participation on SCIVER shared task. Compared to the baseline system, we achieve substantial improvements on the dev set. As a result, our team is the No. 4 team on the leaderboard at the time. 

## Approach
We propose an approach that performs scientific claim verification by doing binary classifications step-by-step. 

### Evidence Retrieval
We trained a BioBERT-large classifier to select abstracts based on relevance assessments for each sentence-pair <claim, title of the abstract> and continued to train it to select rationales out of each retrieved abstract. 

### Label Prediction
We propose a two-step setting for label prediction, i.e. first predicting "NOT\_ENOUGH\_INFO" or "ENOUGH\_INFO", then label those marked as "ENOUGH\_INFO" as either "SUPPORT" or "CONTRADICT". 

## Dependencies
Due to time limits, we only updated label prediction module to use more up-to-date transformers library and kept using the older version of it on our evidence retrieval module as it is inherited from scifact baseline repository.

To do evidence retrieval, please create a virtual environment named `evidence` with the following command.
```bash
conda env create -f evidence/evidence.yml
```
To do two-step label prediction, please create a virtual environment named `two-step` with the following command.
```bash
conda env create -f label/two-step.yml
```

Please use `evidence` environment if you want to reproduce results from the baseline system; please use `two-step` environment if you want to reproduce BERTscore related results.
 
## Dataset
For your convenience, the proprecessed data is available [here](https://www.dropbox.com/sh/gwq9bkto4bpq7bz/AADj8Zj1gx9ew4xWTN7cS_0oa?dl=0); please see detailed descriptions of the SCIFACT dataset [here](https://github.com/allenai/scifact).

## Download trained Models
All of the model checkpoints are available on Huggingface model hub under the user name `xiazeng`. In general, the naming pattern is `xiazeng/sciverbinary-model_${training_data}-${base_model}-${task}`, where `training_data` may be `train_data` or `train_dev_data`, `base_model` may be `biobertb` (BioBERT-base), `biobertb_wo` (BioBERT-base without continued pertaining on SciFact corpus) `biobertl` (BioBERT-large), `robertal` (RoBERTa-large), `task` may be `abstract` (trained on the task of abstract retrieval), `rationale` (trained on the task of rationale selection), `rationale_continue`  (first trained on the task of abstract retrieval and the continued the training on the task of rationale selection), `label-neutral_detector` (trained on the task of detecting whether there is enough info or not, i.e, the first step of the two-step label prediction task), `label-entail_detector` (trained on the task of detecting whether there is entailment or not, i.e, the second step of the two-step label prediction task).

The models we used to generate final submissions are have the model id `xiazeng/sciverbinary-model_train_dev_data-biobertl-abstract`, `xiazeng/sciverbinary-model_train_dev_data-biobertl-rationale_continue`, `xiazeng/sciverbinary-model_train_dev_data-robertal-label-neutral_detector` and `xiazeng/sciverbinary-model_train_dev_data-robertal-label-entail_detector` (order is the pipeline order). They are trained with both train set and dev set.

Our paper reported model performance of various BERT variants that are trained on train set and evaluated on dev set. They are also available with coresponding model ids.


## Reproduce evaluation metrics

After downloading the dataset and trained models, please use `script/pipeline.sh` to run the whole pipeline and get evaluation metrics. See detailed usage instructions in the script.

## Training
Please use `evidence/training/abstract_retrieval/classifier.py` to train a model to do abstract retrieval.

Please use `evidence/training/rationale_selection/classifier.py` to train a model to do rationale selection.

Please use `label/training/neutral_detector.py` to train a model to detect "NOT\_ENOUGH\_INFO".

Please use `label/training/support_detector.py` to train a model to detect "SUPPORT".

## Contact
Email: `x.zeng@qmul.ac.uk`
