import pandas as pd
from datasets import Dataset
import os
import torch
from transformers import DataCollatorForLanguageModeling, AutoModelForCausalLM, TrainingArguments, Trainer, IntervalStrategy, EarlyStoppingCallback, AutoTokenizer
import argparse
import random
import numpy as np
import math

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--file_path", type=str, help="path to your csv file", required=True
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="batch size (8 or 16)", required=False
    )
    parser.add_argument(
        "--max_steps_num",
        type=int,
        default=10000,
        help="maximum number of training steps", required=False
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="weight decay better be 1e-5",required=False
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-5, help="learning rate", required=False
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="mlm_model",
        help="name for trained models folder", required=False
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
        help="friendly model name", required=False,
    )
    parser.add_argument(
        "--say_when",
        type=int,
        default=5,
        help="Number of epochs of no improvement after which the training must stop ", required=True
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

    print(f'\nCREATING DATASET FROM {args.file_path}\n')

    df = pd.read_csv(args.file_path, names=['text'], encoding='utf-8', index_col=None)
    main_data = Dataset.from_pandas(df)

    main_data = main_data.train_test_split(test_size=0.1)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    tokenized_data = main_data.map(
        preprocess_function,
        batched=True,
        num_proc=4,
        remove_columns=main_data["train"].column_names,
    )

    block_size = 128


    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_dataset = tokenized_data.map(group_texts, batched=True, num_proc=4)

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    device = torch.device("cuda") if torch.cuda.is_available else "cpu"
    
    print(f'DOWNLOADING MODEL\n')
    
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model = model.to(device)

    print(f'MODEL USED: {model.config.name_or_path}\n')

    batch_size = args.batch_size

    k_steps = args.max_steps_num // 1000
    
    training_args = TrainingArguments(
        output_dir = f"{args.output_dir}/{args.nickname}_{k_steps}k_steps",
        learning_rate = args.learning_rate,
        push_to_hub=False,
        report_to = 'none',
        save_total_limit = 3,
        warmup_steps = 10,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size = batch_size,
        evaluation_strategy = IntervalStrategy.STEPS,
        save_strategy = IntervalStrategy.STEPS,
        save_steps = 100,
        logging_steps = 50,
        eval_steps = 50,
        max_steps = args.max_steps_num,
        load_best_model_at_end = True,
        metric_for_best_model ='loss', 
        greater_is_better = False,
        weight_decay = args.weight_decay,
        seed = 42
    )

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = lm_dataset["train"],
        eval_dataset = lm_dataset["test"],
        data_collator = data_collator,
        tokenizer = tokenizer,
        callbacks = [EarlyStoppingCallback(early_stopping_patience = args.say_when, early_stopping_threshold = 0.01)]
    )

    print("\nTRAINING STARTED!")

    trainer.train()

    def compute_metrics(eval_pred):
        perpl = math.exp(eval_pred['eval_loss'])

        return {"Perplexity": perpl}

    print("\nEVALUATING ON THE TEST SPLIT...")

    metrics = trainer.evaluate()

    print(metrics)

    print('\n')

    print(compute_metrics(metrics))

    print("\nTRAINING FINISHED!\n")

    print(f"THE BEST MODEL LOCATED AT {trainer.state.best_model_checkpoint}\n")
