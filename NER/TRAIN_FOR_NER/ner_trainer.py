from from_conll_to_hf import *

from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import load_metric

import os
import wandb

import numpy as np

import argparse


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
        default=5_000,
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
        help="name for trained models folder",
    )

    args = parser.parse_args()

    main_path_in = args.file_path

    tg_in_t = args.tags
    
    tg_in = tg_in_t.split(',')

    exp_bool_in = args.export

    dataset = HF_NER_dataset(mp = main_path_in, tg = tg_in, exp_bool=exp_bool_in).dataset

    label_names = dataset["train"].features["ner_tags"].feature.names

    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {v: k for k, v in id2label.items()}

    tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased", model_max_length=512, truncation=True)
    model = AutoModelForTokenClassification.from_pretrained(
        "DeepPavlov/rubert-base-cased", 
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
        )

    #Get the values for input_ids, token_type_ids, attention_mask
    def tokenize_adjust_labels(all_samples_per_split):
        
        tokenized_samples = tokenizer.batch_encode_plus(all_samples_per_split["tokens"], max_length=512, truncation=True, is_split_into_words=True)
        #tokenized_samples is not a datasets object so this alone won't work with Trainer API, hence map is used 
        #so the new keys [input_ids, labels (after adjustment)]
        #can be added to the datasets dict for each train test validation split
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

    os.environ["WANDB_API_KEY"]="42277c5edc2cce1067dce1109dec6b3001d25270"
    os.environ["WANDB_ENTITY"]="EISTAKOVSKII"
    os.environ["WANDB_PROJECT"]="finetune_bert_ner_test"
    
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
        output_dir=f"{args.output_dir}/rubert_{args.max_steps}",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        save_total_limit = 2,
        report_to="wandb",
        run_name = f"{args.max_steps}_steps_rubert",
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=1000,
        logging_steps=1000,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    print("\nTRAINING STARTED!\n")
    
    trainer.train()
    
    wandb.finish()

    print("\nTRAINING FINISHED!\n")
