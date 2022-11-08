Transform a CONLL type file into a hugging face dataset

To use the script in the notebook copy and paste the following code:



```python

!wget https://raw.githubusercontent.com/eistakovskii/NLP_projects/main/NER/CONLL_TO_HF/from_conll_to_hf.py

from from_conll_to_hf import *

main_path_in = YOUR PATH # insert here the path to your txt files split in train, valid and test splits as a string
tg_in = ("B-LOC", "I-LOC", "B-ORG", "I-ORG", "O") # add here your NE tags in the BIO format as a tuple
dataset = HF_NER_dataset(mp = main_path_in, tg = tg_in).dataset

print(dataset['train'])
print(dataset['test'])
print(dataset['validation'])

print("List of tags: ", dataset['train'].features['ner_tags'].feature.names)


print("First sample: ", dataset['train'][0])

```


