from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback, IntervalStrategy, DataCollatorWithPadding
from datasets import load_metric

from tqdm.auto import tqdm

import os

import numpy as np

import argparse

import random
import torch

from torch import nn

from prep_data import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--file_path", type=str, help="path to your csv file", required=True
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="batch size (8 or 16)", required=False
    )
    parser.add_argument(
        "--max_epoch_num",
        type=int,
        default=10,
        help="maximum number of training epochs", required=False
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="weight decay better be 1e-5",required=False
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="learning rate", required=False
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="trained_model",
        help="name for trained models folder", required=True
    )
    parser.add_argument(
        "--model",
        type=str,
        help="your model name or its path", required=True,
    )
    parser.add_argument(
        "--nickname",
        default="temp_model",
        type=str,
        help="friendly model name", required=True,
    )
    parser.add_argument(
        "--say_when",
        type=int,
        default=5,
        help="Number of epochs of no improvement after which the training must stop ", required=True
    )
    parser.add_argument(
        "--zero_means",
        type=str,
        default='LABEL_0',
        help="what the label 0 means for your task", required=False
    )
    parser.add_argument(
        "--one_means",
        type=str,
        default='LABEL_1',
        help="what the label 1 means for your task", required=False
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
    
    print(f'\nCREATING DATASET FROM {args.file_path}\n')
    
    train_ds, val_ds, test_ds = prep_and_split_data(args.file_path)

    print(f'DOWNLOADING MODEL AND TOKENIZER\n')
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def preprocess_function(examples):
       return tokenizer(examples["text"], truncation=True, max_length=512)

    tokenized_train = train_ds.map(preprocess_function, batched=True).remove_columns('text')

    tokenized_val = val_ds.map(preprocess_function, batched=True).remove_columns('text')

    tokenized_test = test_ds.map(preprocess_function, batched=True).remove_columns('text')
    
    id2label = {
        "0": args.zero_means,
        "1": args.one_means
    }
    label2id = {
        args.zero_means: 0,
        args.one_means: 1
    }

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    model = AutoModelForSequenceClassification.from_pretrained(args.model, 
                                                               num_labels=2,
                                                               id2label=id2label,
                                                               label2id=label2id)
    
    model.to(device)

    print(f'MODEL USED: {model.config.name_or_path}\n')

    def compute_metrics(eval_pred):
        load_accuracy = load_metric("accuracy")
        load_f1 = load_metric("f1")

        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
        f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
        
        return {"accuracy": accuracy, "f1": f1}

    repo_name = args.output_dir

    batch_size = args.batch_size

    num_ep = args.max_epoch_num

    logging_steps_num = len(train_ds) // batch_size

    check_in_on_steps = logging_steps_num // 3

    training_args = TrainingArguments(
        output_dir=f"{args.output_dir}/{args.nickname}_{args.max_epoch_num}ep",
        learning_rate = args.learning_rate,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size = batch_size*4,
        weight_decay = args.weight_decay,
        seed=42,

        disable_tqdm = False,
        push_to_hub = False,
        report_to = 'none',
        warmup_steps = 200,

        save_total_limit = 5,
        evaluation_strategy = IntervalStrategy.STEPS,
        save_strategy = IntervalStrategy.STEPS,

        save_steps = check_in_on_steps,
        logging_steps = check_in_on_steps,
        eval_steps = check_in_on_steps,
        max_steps = logging_steps_num*num_ep,
        metric_for_best_model = 'f1',
        load_best_model_at_end = True
        )
    
    class_weights = train_ds.num_rows / (2 * np.bincount(train_ds['label']))
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    class_weights = class_weights.to(torch.device('cuda'))
    
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss_fct = nn.CrossEntropyLoss(weight=class_weights)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            
            return (loss, outputs) if return_outputs else loss
    
    trainer = CustomTrainer(
        model = model,
        args = training_args,
        train_dataset = tokenized_train,
        eval_dataset = tokenized_val,
        tokenizer = tokenizer,
        data_collator = data_collator,
        compute_metrics = compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience = args.say_when, early_stopping_threshold = 0.15)]
        )

    print("\nTRAINING STARTED!")

    trainer.train()

    print("\nEVALUATING ON THE TEST SPLIT...")

    print(trainer.evaluate(tokenized_test))

    print("\nTRAINING FINISHED!\n")

    print(f"\nTHE BEST MODEL LOCATED AT {trainer.state.best_model_checkpoint} WITH F1 {trainer.state.best_metric}\n")
