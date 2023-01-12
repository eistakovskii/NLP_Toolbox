import pymorphy2
import razdel

lemmatizer = pymorphy2.MorphAnalyzer()

def lemmatizations(text: str) -> list:
    '''
    Use this function to lemmatize the input string. The output is the list with lemmas.

    Input:
        text: a string to be lemmatized
    Output:
        _ : a list of with lemmas
    '''
    tokens_normalized, _ = [], [] # tokens_normalized, lemma_list = [], []
    token_list = [_.text for _ in razdel.tokenize(text.lower())]
    for token in token_list:
        lemma = lemmatizer.parse(token)[0].normal_form
        tokens_normalized.append(lemma)
    
    return " ".join(tokens_normalized).split(' ')
