import os
import json
import yaml

import torch
from torch import nn
from datasets import load_from_disk
import evaluate 
import numpy as np
from transformers import AutoTokenizer, DataCollatorForTokenClassification, TrainingArguments, EarlyStoppingCallback, IntervalStrategy
from adapters import AutoAdapterModel, AdapterTrainer
from argparse import ArgumentParser

from utils import set_random_seed

def do_the_adapter_training(target_ner_group_in: str, output_path_in: str, model, tokenizer, data_collator, metric):
    """
    DESCRIPTION:

    Perform the adapter training for the given NER group
    
    ARGS:

    target_ner_group_in: the string indicating which ner group will be trained
    output_path_in: the path where you want to store the resulting adapter; note that we store multiple checkpoints
    model: the initialized model object where you have already loaded the base model, e.g. ruBert, mBert etc
    tokenizer: the initialized tokenizer object, same as above
    data_collator: the initialized data collator object, same as above
    metric: the initialized seqeval object

    OUTPUT:

    adapter_name: the formal adapter name, e.g. 'ner_entity', 'ner_group', 'ner_.....
    best_checkpoint_location: the local path to the best checkpoint
    best_checkpoint_metric: the best checkpoint's metric, i.e. 'overall_f1'

    """ 
    # TODO: save the elapsed time
    # TODO: log and save the training
    def compute_metrics(p):
        """
        the main function responsible for metric calculation
        """
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_names[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        flattened_results = {
            "overall_precision": results["overall_precision"],
            "overall_recall": results["overall_recall"],
            "overall_f1": results["overall_f1"],
            "overall_accuracy": results["overall_accuracy"],
        }
        for k in results.keys():
            if(k not in flattened_results.keys()):
                flattened_results[k+"_f1"]=results[k]["f1"]

        return flattened_results

    adapter_name = f"ner_{target_ner_group_in}"
    model.add_adapter(adapter_name)
    model.add_tagging_head(adapter_name, num_labels=len(label_names), id2label=id2label)

    model.set_active_adapters([[adapter_name]])
    model.train_adapter([adapter_name])

    training_args = TrainingArguments(
        output_dir=output_path_in,
        learning_rate=current_experiment_info['learning_rate'],
        per_device_train_batch_size=current_experiment_info['batch_size'],
        per_device_eval_batch_size=current_experiment_info['batch_size'],
        weight_decay=current_experiment_info['weight_decay'],
        save_total_limit = 5,
        run_name = current_experiment_info['experiment_name'],
        evaluation_strategy=IntervalStrategy.STEPS,
        save_strategy=IntervalStrategy.STEPS,
        save_steps=current_experiment_info['check_every_x_steps'],
        logging_steps=current_experiment_info['check_every_x_steps'],
        eval_steps=current_experiment_info['check_every_x_steps'],
        max_steps=current_experiment_info['max_steps'],
        seed=current_experiment_info['seed'],
        report_to='none',
        disable_tqdm = True,
        metric_for_best_model = 'overall_f1',
        load_best_model_at_end=True,
        # remove_unused_columns=False
    )
    # if target_ner_group_in == 'entity':
    class CustomTrainerWithWeights(AdapterTrainer): # overwrite the loss function to be able to use weights for NER classes
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            # forward pass
            outputs = model(**inputs)
            logits = outputs.get('logits')
            # compute custom loss with class weights for all BIO tags
            loss_fct = nn.CrossEntropyLoss(weight=weights_tensor)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    trainer = CustomTrainerWithWeights(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience = current_experiment_info['early_stopping_patience'], 
                                         early_stopping_threshold = current_experiment_info['early_stopping_threshold'])]
    )
    # else:
    #     trainer = AdapterTrainer(
    #         model=model,
    #         args=training_args,
    #         train_dataset=dataset["train"],
    #         eval_dataset=dataset["validation"],
    #         data_collator=data_collator,
    #         tokenizer=tokenizer,
    #         compute_metrics=compute_metrics,
    #         callbacks=[EarlyStoppingCallback(early_stopping_patience = 4, early_stopping_threshold = 0.05)]
    #     ) 
    # print(f'\nTRAINING THE ADAPTER FOR THE NER GROUP "{target_ner_group_in}"')

    trainer.train()
    
    test_results = trainer.evaluate(dataset["test"])

    best_checkpoint_location = trainer.state.best_model_checkpoint
    best_checkpoint_metric = trainer.state.best_metric

    # Deactivate all adapters
    model.set_active_adapters(None)
    # Delete the added adapter
    model.delete_adapter(adapter_name)
    model.delete_head(adapter_name)

    return adapter_name, best_checkpoint_location, best_checkpoint_metric, test_results

def align_labels_with_tokens(labels, word_ids, id2label):
    """
    this function responsible for fixing up the raw tagging data after in went through the tokenizer;
    in particular it fixes up the BIO tagging in the cases when the tokenizer splitting does match with
    the original splitting logic, i.e. split by empty spaces.
    note the part where we assign -100 to subtokens - that can be changed: see the commented out lines 
    """
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # if current_experiment_info['ignore_subtoken']:
            label = - 100 # Uncomment these two lines to start ignoring subtokens
            new_labels.append(label)
            # else: 
            #     # Same word as previous token
            #     label = labels[word_id]
            #     # If the label is B-XXX we change it to I-XXX
            #     if id2label[label].startswith('B-'):
            #         label += 1
            #     new_labels.append(label)
    return new_labels

def tokenize_and_align_labels(examples):
    """
    basically applies the align_labels_with_tokens function but this time actually using the tokenizer
    """
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids, id2label))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--preprocessed_data_path',
        required=True,
        type=str, 
        help="Path to the prerocessed data in .hf format"
    )
    parser.add_argument(
        '--to_save_path',
        required=True,
        type=str, 
        help="Path to save the adapters checkpoints"
    )
    parser.add_argument(
        '--output_test_results_path',
        required=True,
        type=str, 
        help="Path to save the test split evaluation results"
    )

    args = parser.parse_args()

    curr_dir = os.getcwd() 

    preprocessed_data_path = os.path.join(curr_dir, args.preprocessed_data_path)

    output_path = os.path.join(curr_dir, args.to_save_path)
    os.makedirs(output_path, exist_ok=True)

    output_test_results_path = args.output_test_results_path
    os.makedirs(output_test_results_path, exist_ok=True)
    current_test_info = dict() # extra info about the test split
    metrics_out = dict() # the output metrics for dvc; here we record the overall_f1 metrics after evaluation on the test split

    params = yaml.safe_load(open("params.yaml"))[f"train_stage_{curr_domain}"] # we get this from the params.yaml file

    seqeval_path = os.path.join(curr_dir, r'src\scripts\seqeval.py') 

    current_experiment_info = {"model_name": params['model_name'], # at the end we output this file
                               "seed": params['seed'],
                                "learning_rate": float(params['learning_rate']), 
                                "weight_decay": float(params['weight_decay']), 
                                "batch_size": params['batch_size'],
                                "max_steps":params['max_number_of_steps'],
                                "check_every_x_steps":params['checkpoint_every_x_steps'],
                                "early_stopping_patience":params['early_stopping_patience'],
                                "early_stopping_threshold":params['early_stopping_threshold'],
                                "experiment_name":params['experiment_name']}
    
    os.environ["WANDB_DISABLED"] = "true"
    set_random_seed(current_experiment_info['seed'])

    model_name = current_experiment_info['model_name']
  
    tokenizer = AutoTokenizer.from_pretrained(downloaded_model_path)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    metric = evaluate.load(seqeval_path)

    model = AutoAdapterModel.from_pretrained(downloaded_model_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Start the main training part
    path_to_ner_group = os.path.join(preprocessed_data_path, f'{NER_GROUP}.hf') # note the .hf extensions

    dataset = load_from_disk(path_to_ner_group) # TODO: load only the train and validation part -> can we do it?? -> need to check

    label_names = dataset['info']['labels']

    # if NER_GROUP == 'entity':
    weights = dataset['info']['weights'] # note the weights we got from the preprocessing stage
    weights_tensor = torch.tensor(weights, dtype=torch.float)
    
    label2id = {val:ind for ind, val in enumerate(label_names)}
    id2label = {ind:val for ind, val in enumerate(label_names)}

    dataset['train'] = dataset['train'].map( # apply the tokenize_and_align_labels on the split data 
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset['train'].column_names,
    )

    dataset['validation'] = dataset['validation'].map( # apply the tokenize_and_align_labels on the split data 
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset['validation'].column_names,
        )
    
    # here we want to save some statistics about our test split
    label_names_for_stats = [i if i == 'O' else i[2:] for i in label_names]
    label_names_for_stats = list(set(label_names_for_stats))

    # Note that we use all labels present; this way we will be see if any labels were represented with a count==0 in the test split, i.e. unlucky seed/poor shuffle
    # at the preprocessing stage
    labels_stats = {i:0 for i in label_names_for_stats} # TODO: make all splits evenly balanced instead of relying on the shuffle during the preprocessing
    
    for i in dataset['test']: # iterate through the test split to collect the tags
        for j in i['ner_tags']:
            curr_tag = id2label[j]
            if curr_tag == 'O':
                labels_stats[curr_tag] += 1
            elif curr_tag.startswith('B-'):
                out_tag = curr_tag[2:]
                labels_stats[out_tag] += 1
    # here we finish getting the frequency statistics

    dataset['test'] = dataset['test'].map( # apply the tokenize_and_align_labels on the split data 
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset['test'].column_names,
    )
    
    output_path_final = os.path.join(output_path, NER_GROUP)

    # do the training
    name_output, best_checkpoint_path, metric_output, test_results = do_the_adapter_training(NER_GROUP, output_path_final, model, tokenizer, data_collator, metric)

    temp_dict = {'overall_f1': metric_output, # insert extra info into the current_experiment_info dict
                'best_checkpoint':best_checkpoint_path}
    current_experiment_info[NER_GROUP] = temp_dict
    current_experiment_info['data_used'] = path_to_ner_group

    temp_dict_test = {'test_results': test_results, 'data_used': path_to_ner_group, 'adapter_used': best_checkpoint_path, 'test_split_statistics': labels_stats}
    
    current_test_info[NER_GROUP] = temp_dict_test
    metrics_out[NER_GROUP] = test_results["eval_overall_f1"]


    out_json_path = os.path.join(output_path, f'experiment_results.json')

    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(current_experiment_info, f, ensure_ascii=False, indent=4)

    with open(os.path.join(output_test_results_path, 'test_results.json'), 'w', encoding='utf-8') as f: # this is more detailed report on testing
        json.dump(current_test_info, f, ensure_ascii=False, indent=4)

    with open(os.path.join(output_test_results_path, 'metrics.json'), 'w', encoding='utf-8') as f: # here we output only the 'overall_f1' values for each NER group and mostly just for dvc
        json.dump(metrics_out, f, ensure_ascii=False, indent=4)
