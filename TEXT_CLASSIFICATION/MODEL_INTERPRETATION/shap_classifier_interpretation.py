from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline
import torch
import shap
import pandas as pd
import numpy as np
import random
from tqdm import tqdm

device = 'cuda'

# Initialize the model and tokenizer
model_name = "textdetox/xlmr-large-toxicity-classifier"
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline(model = model, tokenizer=tokenizer, task = "sentiment-analysis", top_k=None)
explainer = shap.Explainer(classifier)

def generate_random_indices(input_list, seed, n):
    random.seed(seed)
    random_indices = random.sample(range(len(input_list)), n)
    return random_indices

def extract_positive_clusters2(values, data):

    # Find the indices where values are positive
    positive_indices = np.where(values > 0)[0]

    # Find clusters of consecutive numbers
    clusters = np.split(positive_indices, np.where(np.diff(positive_indices) != 1)[0]+1)


    # Extract groups of words from the data array using the clusters
    groups = [' '.join(data[cluster].tolist()) for cluster in clusters if cluster.size > 0]

    groups = [i for i in groups if i !='']

    return [''.join(i.split())  for i in groups], groups

def group_token_stat(groups_list):
    group_stat = list()

    for i in groups_list:
        group_stat.append(len(i.split()))

    return group_stat


def get_most_toxic_tokens_stats(lang: str, how_many: str, seed: int):

    toxic_sents = data.loc[data['lang'] == lang]['toxic_sentence'].to_list()
    inds = generate_random_indices(toxic_sents, seed, how_many)

    input_sents = [toxic_sents[i] for i in inds]
    all_groups_out = list()
    all_groups_stats = list()

    shap_values = explainer(input_sents)
    tox_obj = shap_values[:, :, "toxic"]

    for i in tqdm(range(len(input_sents)), desc = f'current language: {lang}'):

        vals_i = tox_obj.values[i]
        data_i = tox_obj.data[i]
        groups_merged, groups_in_tokens = extract_positive_clusters2(vals_i, data_i)
        all_groups_out.append(groups_merged)
        all_groups_stats.append(group_token_stat(groups_in_tokens))

    return lang, input_sents, all_groups_out, all_groups_stats

def get_most_comm(lang):
  toxic_spans = data.loc[data['lang'] == lang]['toxic_spans'].to_list()
  toxic_spans = [ast.literal_eval(i) for i in toxic_spans]
  toxic_spans_out = list()
  for i in toxic_spans:
    toxic_spans_out.extend(i)
  return Counter(toxic_spans_out).most_common()
