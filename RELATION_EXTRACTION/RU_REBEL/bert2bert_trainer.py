import pandas as pd
from datasets import Dataset
import os
from transformers import BertTokenizerFast, Seq2SeqTrainer, Seq2SeqTrainingArguments, EncoderDecoderModel, DataCollatorForSeq2Seq
from transformers.optimization import Adafactor, AdafactorSchedule
from tqdm.auto import tqdm
import random
from random import randint
from utils_compute_metrics import *
import torch

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

print('\nPREPING DATA...')

df_train = pd.read_csv(r'C:\Temp_Workspace\TRAINING\ru_rebel_big\ru_rebel_data_train.csv', encoding = 'utf-8')
df_dev = pd.read_csv(r'C:\Temp_Workspace\TRAINING\ru_rebel_big\ru_rebel_data_dev.csv', encoding = 'utf-8')
df_test = pd.read_csv(r'C:\Temp_Workspace\TRAINING\ru_rebel_big\ru_rebel_data_test.csv', encoding = 'utf-8')

train_ds = Dataset.from_pandas(df_train, split="train")
dev_ds = Dataset.from_pandas(df_dev, split="dev")
test_ds = Dataset.from_pandas(df_test, split="test")

tokenizer = BertTokenizerFast.from_pretrained('sberbank-ai/ruBert-base')
tokenizer.add_tokens(['<obj>', '<subj>', '<triplet>'], special_tokens=True)

encoder_max_length = 128
decoder_max_length = 128

def process_data_to_model_inputs(batch):
  # tokenize the inputs and labels
  inputs = tokenizer(batch["context"], padding="max_length", truncation=True, max_length=encoder_max_length)
  outputs = tokenizer(batch["triplets"], padding="max_length", truncation=True, max_length=decoder_max_length)

  batch["input_ids"] = inputs.input_ids
  batch["attention_mask"] = inputs.attention_mask
  batch["decoder_input_ids"] = outputs.input_ids
  batch["decoder_attention_mask"] = outputs.attention_mask
  batch["labels"] = outputs.input_ids.copy()

  # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`. 
  # We have to make sure that the PAD token is ignored
  batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]
  
  return batch

batch_size = 8

train_data = train_ds.map(
    process_data_to_model_inputs, 
    batched=True, 
    batch_size=batch_size, 
    remove_columns=["context", "triplets"]
)
train_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

dev_data = dev_ds.map(
    process_data_to_model_inputs, 
    batched=True, 
    batch_size=batch_size, 
    remove_columns=["context", "triplets"]
)
dev_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

test_data = test_ds.map(
    process_data_to_model_inputs, 
    batched=True, 
    batch_size=batch_size, 
    remove_columns=["context", "triplets"]
)
test_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

print('\nDOWNLOADING MODEL...')
bert2bert = EncoderDecoderModel.from_encoder_decoder_pretrained("sberbank-ai/ruBert-base", "sberbank-ai/ruBert-base")
bert2bert = bert2bert.to('cuda')
print(f'MODEL USED: {bert2bert.config.name_or_path}\n')

bert2bert.decoder.resize_token_embeddings(len(tokenizer))

bert2bert.config.decoder_start_token_id = tokenizer.cls_token_id
bert2bert.config.eos_token_id = tokenizer.sep_token_id
bert2bert.config.pad_token_id = tokenizer.pad_token_id
bert2bert.config.vocab_size = bert2bert.config.encoder.vocab_size

# bert2bert.config.max_length = 142 ###### CHANGES THE SPEED OF VALIDATION FROM 1 HOUR TO 2 MINS
# bert2bert.config.min_length = 56
# bert2bert.config.no_repeat_ngram_size = 3
# bert2bert.config.early_stopping = True
# bert2bert.config.length_penalty = 2.0
# bert2bert.config.num_beams = 4

def compute_metrics(eval_preds) -> dict:
    
    labels_ids = eval_preds.label_ids
    pred_ids = eval_preds.predictions

    list_of_preds_t = tokenizer.batch_decode(pred_ids, skip_special_tokens=False)
    list_of_preds_t = list(map(lambda x: x[:x.find('[SEP]')-1] if x.find('[SEP]') != -1 else x, list_of_preds_t))

    labels_ids[labels_ids == -100] = tokenizer.pad_token_id

    list_of_labels = tokenizer.batch_decode(labels_ids, skip_special_tokens=False)
    list_of_labels = list(map(lambda x: x[:x.find('[SEP]')-1] if x.find('[SEP]') != -1 else x, list_of_labels))
    
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
    
    rand_num_p = randint(0, len(list_of_preds_t)-1)
    rand_num = randint(0, len(gold_final)-1)
    
    rand_pred_str = list_of_preds_t[rand_num_p][:49] if len(list_of_preds_t[rand_num_p])>50 else list_of_preds_t[rand_num_p]
    
    print(f'SANITY CHECK: \nTRUE LABEL: {gold_final[rand_num]}\nPREDICTION: {labels_final[rand_num]}\nRANDOM PREDICTION RAW: {rand_pred_str}')
    
    out_metrics_dict = {'Precision': ret_tuple[0], 'Recall': ret_tuple[1], 'F1': ret_tuple[2]}
    
    return out_metrics_dict

ep_num = 10

logging_steps_num = len(train_ds) // batch_size

check_in_on_steps = logging_steps_num // 4

optimizer = Adafactor(bert2bert.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
lr_scheduler = AdafactorSchedule(optimizer)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=bert2bert)

training_args = Seq2SeqTrainingArguments(
    
    do_train=True,
    do_eval=True,
       
    output_dir = fr'F:\e_stakovskii\output\{ep_num}_ep_bert2bert',
    
    learning_rate = 2e-5,
    per_device_train_batch_size = batch_size,
    per_device_eval_batch_size = batch_size,
    weight_decay = 2e-3,
    seed = 42,
    
    predict_with_generate=True,
    # fp16=True,
    
    evaluation_strategy="steps",
    logging_steps=check_in_on_steps,
    save_steps=check_in_on_steps,
    eval_steps= check_in_on_steps,

    num_train_epochs = ep_num,

    push_to_hub = False,
    report_to = 'none',

    save_total_limit=10,
)

trainer = Seq2SeqTrainer(
    model = bert2bert,
    args = training_args,
    compute_metrics = compute_metrics,
    data_collator = data_collator,
    train_dataset = train_data,
    eval_dataset = dev_data,
    optimizers = (optimizer, lr_scheduler)
)

print('\nSTARTING TRAINING...')
# trainer.train()
trainer.train(resume_from_checkpoint=r'F:\e_stakovskii\output\5_ep_bert2bert\checkpoint-167130')

print("\nEVALUATING ON THE TEST SPLIT...")

print(trainer.evaluate(test_data))

print("\nTRAINING FINISHED!\n")
