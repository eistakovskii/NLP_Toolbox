from sklearn.model_selection import train_test_split
import pandas as pd
from datasets import Dataset

def prep_splits(path_to_data: str, split_ratio: float) -> list:
    
    df = pd.read_csv(path_to_data, names=['labels', 'text'], encoding='utf-8')
    df = df.sample(frac=1).reset_index(drop=True)
    text = df.text.tolist()
    labels = df.label.tolist()
    
    assert len(text) == len(labels), 'Columns do not match!'
    
    len_data = len(text)
    split_border = int(split_ratio * len_data)
    
    train_split_texts = text[:split_border]
    train_split_labels = labels[:split_border]

    test_split_texts = text[split_border:]
    test_split_labels = labels[split_border:]
    
    train_split_texts, val_split_texts, train_split_labels, val_split_labels = train_test_split(train_split_texts, train_split_labels, test_size=.2)
    
    return train_split_texts, train_split_labels, test_split_texts, test_split_labels, val_split_texts, val_split_labels 

def prep_and_split_data(path_to_csv: str) -> list:
    
    train_split_texts, train_split_labels, test_split_texts, test_split_labels, val_split_texts, val_split_labels = prep_splits(path_to_csv, 0.8)
    
    df_train = pd.DataFrame(list(zip(train_split_texts, train_split_labels)), columns = ['text', 'label'])
    df_val = pd.DataFrame(list(zip(val_split_texts, val_split_labels)), columns = ['text', 'label'])
    df_test = pd.DataFrame(list(zip(test_split_texts, test_split_labels)), columns = ['text', 'label'])
    
    train_ds = Dataset.from_pandas(df_train, split="train")
    val_ds = Dataset.from_pandas(df_val, split="val")
    test_ds = Dataset.from_pandas(df_test, split="test")
    
    return train_ds, val_ds, test_ds
