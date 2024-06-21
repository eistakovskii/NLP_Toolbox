from transformers import MarianMTModel, MarianTokenizer
from fuzzywuzzy import fuzz
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random

def backtranslate(texts, source_model_name, target_model_name, temperature=2.1):
    # Check if CUDA (GPU support) is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the tokenizer and model for Russian to English translation and move the model to the GPU
    src_tokenizer = MarianTokenizer.from_pretrained(source_model_name)
    src_model = MarianMTModel.from_pretrained(source_model_name).to(device)

    # Translate texts from Russian to English with temperature control
    translated_to_english = [
        src_model.generate(
            **src_tokenizer(text, return_tensors="pt", padding=True).to(device),
            do_sample=True,
            temperature=temperature
        ) for text in texts
    ]
    english_texts = [src_tokenizer.decode(t[0], skip_special_tokens=True) for t in translated_to_english]

    # Load the tokenizer and model for English to Russian translation and move the model to the GPU
    tgt_tokenizer = MarianTokenizer.from_pretrained(target_model_name)
    tgt_model = MarianMTModel.from_pretrained(target_model_name).to(device)

    # Translate texts from English back to Russian with temperature control
    back_translated_texts = [
        tgt_model.generate(
            **tgt_tokenizer(text, return_tensors="pt", padding=True).to(device),
            do_sample=True,
            temperature=temperature
        ) for text in english_texts
    ]
    russian_texts = [tgt_tokenizer.decode(t[0], skip_special_tokens=True) for t in back_translated_texts]

    return russian_texts

def apply_typos(text, num_changes):
    """
    Apply a specified number of Levenshtein modifications (typos) to the input text.
    :param text: The original text string.
    :param num_changes: The number of typo changes to apply.
    :return: The modified text with typos.
    """
    chars = list(text)
    for _ in range(num_changes):
        change_type = random.choice(['insert', 'delete', 'substitute'])
        index = random.randint(0, len(chars) - 1)

        if change_type == 'insert':
            chars.insert(index, random.choice('абвгдеёжзийклмнопрстуфхцчшщъыьэюя'))
        elif change_type == 'delete' and len(chars) > 1:
            chars.pop(index)
        elif change_type == 'substitute':
            chars[index] = random.choice('абвгдеёжзийклмнопрстуфхцчшщъыьэюя')
    
    return ''.join(chars)

def augment_dataframe(df, backtranslate_func, apply_typos_func):
    # Split the dataframe into 20% for backtranslation and 80% remaining
    df_bt, df_remaining = train_test_split(df, test_size=0.8, random_state=42)
    
    print('\nBACKTRANSLATING...')
    # Apply backtranslation to the 20% split
    df_bt['samples'] = df_bt['samples'].apply(lambda x: backtranslate_func([x], 'Helsinki-NLP/opus-mt-ru-en', 'Helsinki-NLP/opus-mt-en-ru')[0])

    # Append the backtranslated samples to the original dataframe
    df_augmented = pd.concat([df, df_bt])
    
    print('\nAPPLYING TYPOS...')
    # Split the remaining dataframe again to get a fresh 20% for typo application
    df_typos, _ = train_test_split(df_remaining, test_size=0.8, random_state=42)

    # Apply typos to the 20% split
    df_typos['samples'] = df_typos['samples'].apply(lambda x: apply_typos_func(x, num_changes=2))

    # Append the typo samples to the augmented dataframe
    df_augmented = pd.concat([df_augmented, df_typos])

    return df_augmented.reset_index(drop=True)
