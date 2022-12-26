from from_conll_to_hf import *

from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, TrainingArguments, Trainer, EarlyStoppingCallback, IntervalStrategy
from datasets import load_metric

import os
import wandb

import numpy as np

import argparse

import random
import torch

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--file_path", type=str, help="path to your conll file", required=True
    )
    parser.add_argument(
        "--tags", type=str, help="your NE tags", required=True
    )
    parser.add_argument(
        "--export", type=bool, default=False, help="export the splits locally"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="batch size (8 or 16)"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1_000,
        help="maximum training steps (1000, 3000, 5000 or 10000)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="weight decay better be 1e-5",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="learning rate"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="rubert",
        help="name for trained models folder"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="your model name or its path",
    )
    parser.add_argument(
        "--nickname",
        default="temp_model",
        type=str,
        help="friendly model name",
    )
    parser.add_argument(
        "--say_when",
        type=int,
        default=5,
        help="Number of epochs of no improvement after which the training must stop ", required=False
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Number of epochs of no improvement after which the training must stop ", required=False
    )
    args = parser.parse_args()
    
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
    
    main_path_in = args.file_path

    tg_in_t = args.tags
    
    tg_in = tg_in_t.split(',')

    exp_bool_in = args.export
    
    data_name = args.file_path.split('/') if '/' in args.file_path else args.file_path.split('\\')
    
    print(f'\nCREATING DATASET FROM {data_name[-1]}\n')
    
    dataset = HF_NER_dataset(mp = main_path_in, tg = tg_in, exp_bool=exp_bool_in).dataset
    print(dataset['train'])
    print(dataset['test'])
    print(dataset['validation'])

    print("List of tags: ", dataset['train'].features['ner_tags'].feature.names)
    print(f'\nFINISHED CREATING DATASET\n')
    
    label_names = dataset["train"].features["ner_tags"].feature.names

    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {v: k for k, v in id2label.items()}
    
    print(f'\nDOWNLOADING MODEL AND TOKENIZER\n')
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, model_max_length=512, truncation=True)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model, 
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
        )
    
    print(f'\nMODEL USED: {args.model}\n')
    
    print(f'\nFINISHED DOWNLOADING\n')
    
    def tokenize_adjust_labels(all_samples_per_split):
        
        tokenized_samples = tokenizer.batch_encode_plus(all_samples_per_split["tokens"], max_length=512, truncation=True, is_split_into_words=True)
        total_adjusted_labels = []
        print(len(tokenized_samples["input_ids"]))
        for k in range(0, len(tokenized_samples["input_ids"])):
            prev_wid = -1
            word_ids_list = tokenized_samples.word_ids(batch_index=k)
            existing_label_ids = all_samples_per_split["ner_tags"][k]
            i = -1
            adjusted_label_ids = []
        
            for wid in word_ids_list:
                if(wid is None):
                    adjusted_label_ids.append(-100)
                elif(wid!=prev_wid):
                    i = i + 1
                    adjusted_label_ids.append(existing_label_ids[i])
                    prev_wid = wid
                else:
                    label_name = label_names[existing_label_ids[i]]
                    adjusted_label_ids.append(existing_label_ids[i]) 
            total_adjusted_labels.append(adjusted_label_ids)
        tokenized_samples["labels"] = total_adjusted_labels
        return tokenized_samples

    tokenized_dataset = dataset.map(tokenize_adjust_labels, batched=True)

    data_collator = DataCollatorForTokenClassification(tokenizer)

    
    metric = load_metric("seqeval")

    def compute_metrics(p):
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

    training_args = TrainingArguments(
        output_dir=f"{args.output_dir}/{args.nickname}_{args.max_steps}",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        save_total_limit = 5,
        run_name = f"{args.max_steps}_{args.nickname}",
        evaluation_strategy=IntervalStrategy.STEPS,
        save_strategy=IntervalStrategy.STEPS,
        save_steps=200,
        logging_steps=100,
        eval_steps=100,
        max_steps=args.max_steps,
        seed=42,
        report_to='none',
        metric_for_best_model = 'overall_f1',
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience = args.say_when, early_stopping_threshold = args.threshold)]
    )
    
    print("\nTRAINING STARTED!\n")
    
    trainer.train()
    
    print("\nTRAINING STARTED!\n")
    
    trainer.evaluate(tokenized_dataset["test"])
    
    print("\nTRAINING FINISHED!\n")
