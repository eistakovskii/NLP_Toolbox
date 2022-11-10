import argparse
import random

import spacy
from spacy import displacy

from transformers import AutoModelForTokenClassification
from transformers import pipeline
from transformers import AutoTokenizer

import webbrowser

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_path", type=str, help="path to your preferred model checkpoint", required=True
    )
    parser.add_argument(
        "--tags", type=str, help="your NE tags", required=True
    )
    parser.add_argument(
        "--input_text",
        type=str,
        help="The text that you want to tag",
        required=True
    )

    args = parser.parse_args()

    NER_TAGS_TO_KEEP = args.tags.split(',')

    tag_list_temp = list('O')
        
    for tag in NER_TAGS_TO_KEEP:
        bio_tags_out = [prefix + tag for prefix in ['B-', 'I-']]
        tag_list_temp.extend(bio_tags_out)

    label_names = sorted(tag_list_temp)

    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {v: k for k, v in id2label.items()}

    model_checkpoint = args.checkpoint_path

    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        id2label=id2label,
        label2id=label2id,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, model_max_length=512, truncation=True)

    token_classifier = pipeline(
        "ner", model=model, tokenizer = tokenizer,  aggregation_strategy="max"
    )

    pred = token_classifier(args.input_text)

    def get_rand_hex_color(): 
        r = lambda: random.randint(0,255)
        rand_col = '#%02X%02X%02X' % (r(),r(),r())
        return rand_col

    doc_text = args.input_text

    look_for = ['start', 'end', 'entity_group']

    start_end_labels = [[None for i in range(len(look_for))] for i in range(len(pred))]

    for i in range(len(pred)):
        for ind,val in enumerate(look_for):
            start_end_labels[i][ind] = pred[i][val]

    entities = [pred[i]['entity_group'] for i in range(len(pred))]

    colors = {pred[i]['entity_group']:get_rand_hex_color() for i in range(len(pred))}

    options = {"ents": entities, "colors": colors}

    ex = [{"text": doc_text, "ents": [{"start": x[0], "end": x[1], "label": x[2]} for x in start_end_labels]}]
    
    tagged_sent = displacy.render(ex, style="ent", manual=True, options=options, jupyter=False)
    
    with open('tagged_temp.html','w') as f:
        f.write(tagged_sent)
    
    filename = 'tagged_temp.html'
    
    webbrowser.open_new_tab(filename)
