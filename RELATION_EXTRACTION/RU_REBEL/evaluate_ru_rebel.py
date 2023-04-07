from datasets import load_dataset, Dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, AutoConfig, EarlyStoppingCallback, IntervalStrategy
import numpy as np
import torch
import os
import pandas as pd
from transformers.optimization import Adafactor, AdafactorSchedule
from tqdm.auto import tqdm
import argparse
import sys
from collections import Counter
import numpy as np
import random
from random import randint

os.environ["WANDB_DISABLED"] = "true"

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
set_random_seed(42)

print('\nDOWNLOADING MODEL...')
# sberbank-ai/ruT5-base
# memyprokotow/rut5-REBEL-base
# google/flan-t5-base TOKENIZATION PROBLEM
# google/mt5-base
# cointegrated/rut5-base
# model_path = 'sberbank-ai/ruT5-base'
# model_path = r'F:\e_stakovskii\output\5_ep_sber_rut5\checkpoint-167040'

model_path = r'memyprokotow/rut5-REBEL-base'
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
config = AutoConfig.from_pretrained(model_path)
# num_added_toks = tokenizer.add_tokens(['<obj>', '<subj>', '<triplet>'], special_tokens=True)

model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# model.resize_token_embeddings(len(tokenizer))

# config.vocab_size = tokenizer.vocab_size
model = model.to('cuda')

print(f'MODEL USED: {model.config.name_or_path}\n')

# model = nn.DataParallel(model,device_ids = [0, 1])

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

print('\nPREPING DATA...')

# df_train = pd.read_csv(r'C:\Temp_Workspace\TRAINING\data\ru_rebel_data_train.csv', encoding = 'utf-8')
# df_dev = pd.read_csv(r'C:\Temp_Workspace\TRAINING\data\ru_rebel_data_dev.csv', encoding = 'utf-8')
# df_test = pd.read_csv(r'C:\Temp_Workspace\TRAINING\data\ru_rebel_data_test.csv', encoding = 'utf-8')


df_train = pd.read_csv(r'C:\Temp_Workspace\TRAINING\ru_rebel_big\ru_rebel_data_train.csv', encoding = 'utf-8')
df_dev = pd.read_csv(r'C:\Temp_Workspace\TRAINING\ru_rebel_big\ru_rebel_data_dev.csv', encoding = 'utf-8')
df_test = pd.read_csv(r'C:\Temp_Workspace\TRAINING\ru_rebel_big\ru_rebel_data_test.csv', encoding = 'utf-8')

def lim_by_tokens(df_in, tok_lim: int = 512):
    
    print(f'\nDiscarding samples longer than {tok_lim} tokens...')
    
    train_split_context = list()
    train_split_triplets = list()

    for i in tqdm(range(len(df_in))):
        trp_len = len(tokenizer.tokenize(df_in['triplets'][i]))
        cnt_len = len(tokenizer.tokenize(df_in['context'][i]))
        if any([trp_len>tok_lim, cnt_len>tok_lim]):
            continue
        else:
            train_split_triplets.append(df_in['triplets'][i])
            train_split_context.append(df_in['context'][i])

    assert len(train_split_context) == len(train_split_triplets)
    
    df_train = pd.DataFrame(list(zip(train_split_context, train_split_triplets)), columns = ['context', 'triplets'])
    
    print(f'Number of sample discarded: {len(df_in) - len(df_train)}')
    
    return df_train

df_train_lim = lim_by_tokens(df_train)
df_dev_lim = lim_by_tokens(df_dev)
df_test_lim = lim_by_tokens(df_test)

train_ds = Dataset.from_pandas(df_train_lim, split="train")
dev_ds = Dataset.from_pandas(df_dev_lim, split="dev")
test_ds = Dataset.from_pandas(df_test_lim, split="test")

class relation_extraction_dataset(torch.utils.data.Dataset):
    
    def __init__(self, data, tokenizer):

        self.data = data
        self.tokenizer = tokenizer
        self.max_input_length = 512

    def __getitem__(self, idx):

        source = self.tokenizer(
            self.data[idx]['context'],
            max_length=self.max_input_length,
            truncation=True,
            padding='max_length',
            # padding='longest',
            # padding = True,
            return_tensors="pt",
        )
        target = self.tokenizer(
            self.data[idx]['triplets'],
            max_length=self.max_input_length,
            truncation=True,
            padding='max_length',
            # padding='longest',
            # padding = True,
            return_tensors="pt",
        )
        source["labels"] = target["input_ids"]

        return {k: v.squeeze(0) for k, v in source.items()}

    def __len__(self):
        return len(self.data)

trainset = relation_extraction_dataset(train_ds, tokenizer)
tuneset = relation_extraction_dataset(dev_ds, tokenizer)
testset = relation_extraction_dataset(test_ds, tokenizer)

def extract_triplets(text):
    # text = re.sub(r'(\b\>\<\b)', '> <', text)
    text = ' '.join(text.replace('>', '> ').replace('<', ' <').split())
    # text = text[:text.find('</s>')]
    
    triplets = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
    return triplets

NO_RELATION = "no relation"

relations = ['no_relation',
'org:alternate_names',
'org:city_of_headquarters',
'org:country_of_headquarters',
'org:dissolved',
'org:founded',
'org:founded_by',
'org:member_of',
'org:members',
'org:number_of_employees/members',
'org:parents',
'org:political/religious_affiliation',
'org:shareholders',
'org:stateorprovince_of_headquarters',
'org:subsidiaries',
'org:top_members/employees',
'org:website',
'per:age',
'per:alternate_names',
'per:cause_of_death',
'per:charges',
'per:children',
'per:cities_of_residence',
'per:city_of_birth',
'per:city_of_death',
'per:countries_of_residence',
'per:country_of_birth',
'per:country_of_death',
'per:date_of_birth',
'per:date_of_death',
'per:employee_of',
'per:origin',
'per:other_family',
'per:parents',
'per:religion',
'per:schools_attended',
'per:siblings',
'per:spouse',
'per:stateorprovince_of_birth',
'per:stateorprovince_of_death',
'per:stateorprovinces_of_residence',
'per:title']

def parse_arguments():
    parser = argparse.ArgumentParser(description='Score a prediction file using the gold labels.')
    parser.add_argument('gold_file', help='The gold relation file; one relation per line')
    parser.add_argument('pred_file', help='A prediction file; one relation per line, in the same order as the gold file.')
    args = parser.parse_args()
    return args

def score(key, prediction, verbose=True):
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation    = Counter()

    # Loop over the data to compute a score
    for row in range(len(prediction)):
        gold = key[row]
        guess = prediction[row]
         
        if gold == NO_RELATION and guess == NO_RELATION:
            pass
        elif gold == NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
        elif gold != NO_RELATION and guess == NO_RELATION:
            gold_by_relation[gold] += 1
        elif gold != NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1

    # Print verbose information
    if verbose:
        print("Per-relation statistics:")
        relations = gold_by_relation.keys()
        longest_relation = 0
        for relation in sorted(relations):
            longest_relation = max(len(relation), longest_relation)
        for relation in sorted(relations):
            # (compute the score)
            correct = correct_by_relation[relation]
            guessed = guessed_by_relation[relation]
            gold    = gold_by_relation[relation]
            prec = 1.0
            if guessed > 0:
                prec = float(correct) / float(guessed)
            recall = 0.0
            if gold > 0:
                recall = float(correct) / float(gold)
            f1 = 0.0
            if prec + recall > 0:
                f1 = 2.0 * prec * recall / (prec + recall)
            # (print the score)
            sys.stdout.write(("{:<" + str(longest_relation) + "}").format(relation))
            sys.stdout.write("  P: ")
            if prec < 0.1: sys.stdout.write(' ')
            if prec < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(prec))
            sys.stdout.write("  R: ")
            if recall < 0.1: sys.stdout.write(' ')
            if recall < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(recall))
            sys.stdout.write("  F1: ")
            if f1 < 0.1: sys.stdout.write(' ')
            if f1 < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(f1))
            sys.stdout.write("  #: %d" % gold)
            sys.stdout.write("\n")
        print("")

    # Print the aggregate score
    if verbose:
        print("Final Score:")
    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro   = float(sum(correct_by_relation.values())) / float(sum(guessed_by_relation.values()))
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(sum(gold_by_relation.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    print("Precision (micro): {:.3%}".format(prec_micro))
    print("   Recall (micro): {:.3%}".format(recall_micro))
    print("       F1 (micro): {:.3%}".format(f1_micro))
    return prec_micro, recall_micro, f1_micro

# Adapted from: https://github.com/btaille/sincere/blob/6f5472c5aeaf7ef7765edf597ede48fdf1071712/code/utils/evaluation.py
def re_score(pred_relations, gt_relations, relation_types, mode="boundaries"):
    """Evaluate RE predictions
    Args:
        pred_relations (list) :  list of list of predicted relations (several relations in each sentence)
        gt_relations (list) :    list of list of ground truth relations
            rel = { "head": (start_idx (inclusive), end_idx (exclusive)),
                    "tail": (start_idx (inclusive), end_idx (exclusive)),
                    "head_type": ent_type,
                    "tail_type": ent_type,
                    "type": rel_type}
        vocab (Vocab) :         dataset vocabulary
        mode (str) :            in 'strict' or 'boundaries' """

    assert mode in ["strict", "boundaries"]
    relation_types = relations if relation_types is None else relation_types
    # relation_types = [v for v in relation_types if not v == "None"]
    scores = {rel: {"tp": 0, "fp": 0, "fn": 0} for rel in relation_types + ["ALL"]}

    # Count GT relations and Predicted relations
    n_sents = len(gt_relations)
    n_rels = sum([len([rel for rel in sent]) for sent in gt_relations])
    n_found = sum([len([rel for rel in sent]) for sent in pred_relations])

    # Count TP, FP and FN per type
    for pred_sent, gt_sent in zip(pred_relations, gt_relations):
        for rel_type in relation_types:
            # strict mode takes argument types into account
            if mode == "strict":
                pred_rels = {(rel["head"], rel["head_type"], rel["tail"], rel["tail_type"]) for rel in pred_sent if
                             rel["type"] == rel_type}
                gt_rels = {(rel["head"], rel["head_type"], rel["tail"], rel["tail_type"]) for rel in gt_sent if
                           rel["type"] == rel_type}

            # boundaries mode only takes argument spans into account
            elif mode == "boundaries":
                pred_rels = {(rel["head"], rel["tail"]) for rel in pred_sent if rel["type"] == rel_type}
                gt_rels = {(rel["head"], rel["tail"]) for rel in gt_sent if rel["type"] == rel_type}

            scores[rel_type]["tp"] += len(pred_rels & gt_rels)
            scores[rel_type]["fp"] += len(pred_rels - gt_rels)
            scores[rel_type]["fn"] += len(gt_rels - pred_rels)

    # Compute per relation Precision / Recall / F1
    for rel_type in scores.keys():
        if scores[rel_type]["tp"]:
            scores[rel_type]["p"] = 100 * scores[rel_type]["tp"] / (scores[rel_type]["fp"] + scores[rel_type]["tp"])
            scores[rel_type]["r"] = 100 * scores[rel_type]["tp"] / (scores[rel_type]["fn"] + scores[rel_type]["tp"])
        else:
            scores[rel_type]["p"], scores[rel_type]["r"] = 0, 0

        if not scores[rel_type]["p"] + scores[rel_type]["r"] == 0:
            scores[rel_type]["f1"] = 2 * scores[rel_type]["p"] * scores[rel_type]["r"] / (
                    scores[rel_type]["p"] + scores[rel_type]["r"])
        else:
            scores[rel_type]["f1"] = 0

    # Compute micro F1 Scores
    tp = sum([scores[rel_type]["tp"] for rel_type in relation_types])
    fp = sum([scores[rel_type]["fp"] for rel_type in relation_types])
    fn = sum([scores[rel_type]["fn"] for rel_type in relation_types])

    if tp:
        precision = 100 * tp / (tp + fp)
        recall = 100 * tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

    else:
        precision, recall, f1 = 0, 0, 0

    scores["ALL"]["p"] = precision
    scores["ALL"]["r"] = recall
    scores["ALL"]["f1"] = f1
    scores["ALL"]["tp"] = tp
    scores["ALL"]["fp"] = fp
    scores["ALL"]["fn"] = fn

    # Compute Macro F1 Scores
    scores["ALL"]["Macro_f1"] = np.mean([scores[ent_type]["f1"] for ent_type in relation_types])
    scores["ALL"]["Macro_p"] = np.mean([scores[ent_type]["p"] for ent_type in relation_types])
    scores["ALL"]["Macro_r"] = np.mean([scores[ent_type]["r"] for ent_type in relation_types])

    print(f"RE Evaluation in *** {mode.upper()} *** mode")

    print(
        "processed {} sentences with {} relations; found: {} relations; correct: {}.".format(n_sents, n_rels, n_found,
                                                                                             tp))
    print(
        "\tALL\t TP: {};\tFP: {};\tFN: {}".format(
            scores["ALL"]["tp"],
            scores["ALL"]["fp"],
            scores["ALL"]["fn"]))
    print(
        "\t\t(m avg): precision: {:.2f};\trecall: {:.2f};\tf1: {:.2f} (micro)".format(
            precision,
            recall,
            f1))
    print(
        "\t\t(M avg): precision: {:.2f};\trecall: {:.2f};\tf1: {:.2f} (Macro)\n".format(
            scores["ALL"]["Macro_p"],
            scores["ALL"]["Macro_r"],
            scores["ALL"]["Macro_f1"]))

    for rel_type in relation_types:
        print("\t{}: \tTP: {};\tFP: {};\tFN: {};\tprecision: {:.2f};\trecall: {:.2f};\tf1: {:.2f};\t{}".format(
            rel_type,
            scores[rel_type]["tp"],
            scores[rel_type]["fp"],
            scores[rel_type]["fn"],
            scores[rel_type]["p"],
            scores[rel_type]["r"],
            scores[rel_type]["f1"],
            scores[rel_type]["tp"] +
            scores[rel_type][
                "fp"]))

    return scores, precision, recall, f1

def compute_metrics(eval_preds) -> dict:
    
#     list_of_preds = list()
    
#     for i in eval_preds.predictions[0]:
#         temp_ids_list = list()
#         for j in i:
#             temp_ids_list.append(np.argmax(j, axis=-1))
#         list_of_preds.append(tokenizer.decode(temp_ids_list, skip_special_tokens=False))
    
    list_of_preds_t = tokenizer.batch_decode(eval_preds.predictions, skip_special_tokens=False)
    list_of_preds_t = list(map(lambda x: x[:x.find('</s>')+4] if x.find('</s>') != -1 else x, list_of_preds_t))
    
    list_of_labels = tokenizer.batch_decode(eval_preds.label_ids, skip_special_tokens=False)
    list_of_labels = list(map(lambda x: x[:x.find('</s>')+4] if x.find('</s>') != -1 else x, list_of_labels))
    
#     del(eval_preds)
    
#     print(list_of_labels[:2])
#     print(list_of_preds_t[:2])
    
    gold_final = list()
    labels_final = list()

    for i in range(len(list_of_preds_t)):

        gold_labels = [i['type'] for i in extract_triplets(list_of_labels[i])]

        preds = [i['type'] for i in extract_triplets(list_of_preds_t[i])]

        ch_num = len(gold_labels) - len(preds)

        if ch_num == 0:
            gold_final.extend(gold_labels)
            labels_final.extend(preds)
        elif ch_num > 0:
            preds.extend([None for i in range(ch_num)])
            gold_final.extend(gold_labels)
            labels_final.extend(preds)
        else:
            preds = preds[:len(gold_labels)]
            gold_final.extend(gold_labels)
            labels_final.extend(preds)

    assert len(gold_final) == len(labels_final)
    
    ret_tuple = score(gold_final, labels_final, verbose=True)
    
    rand_num = randint(0, len(gold_final)-1)
    
    print(f'SANITY CHECK: \n{gold_final[rand_num]}\n{labels_final[rand_num]}')
    
    out_metrics_dict = {'Precision': ret_tuple[0], 'Recall': ret_tuple[1], 'F1': ret_tuple[2]}
    
    return out_metrics_dict

optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
lr_scheduler = AdafactorSchedule(optimizer)

batch_size = 8

ep_num = 5

logging_steps_num = len(train_ds) // batch_size

check_in_on_steps = logging_steps_num // 4

args = Seq2SeqTrainingArguments(
    output_dir = fr'F:\e_stakovskii\output\{ep_num}_ep_sber_rut5',
    
    do_train=True,
    do_eval=True,
    
    learning_rate = 2e-5,
    per_device_train_batch_size = batch_size,
    per_device_eval_batch_size = batch_size * 2,
    weight_decay = 2e-3,
    seed = 42,
    
    disable_tqdm = False,
    push_to_hub = False,
    report_to = 'none',


    save_steps = check_in_on_steps,
    logging_steps = check_in_on_steps,
    eval_steps = check_in_on_steps,
    num_train_epochs = ep_num,

    evaluation_strategy = IntervalStrategy.STEPS,
    save_strategy = IntervalStrategy.STEPS,
    
    metric_for_best_model = 'F1',
    load_best_model_at_end = True,    
    
    predict_with_generate = True,
    # eval_accumulation_steps=100,
    # fp16=True
)

trainer = Seq2SeqTrainer(
    model = model,
    args = args,
    train_dataset = trainset,
    eval_dataset = tuneset,
    tokenizer = tokenizer,
    data_collator = data_collator,
    compute_metrics = compute_metrics,
    optimizers = (optimizer, lr_scheduler),
    callbacks = [EarlyStoppingCallback(early_stopping_patience = 10, early_stopping_threshold = 0.05)]
)

print(trainer.evaluate(testset))