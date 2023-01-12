import pymorphy2
import razdel

morph = pymorphy2.MorphAnalyzer()
morph = pymorphy2.MorphAnalyzer(lang='ru')

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

def get_num(inp_str: str) -> tuple:
    '''
    Use this function to separate a numeral and noun from the input string in Russian
    
    Input:
        inp_str: a string to extract the numeral from

    Output:
        _ : tuple containing on the first place the numeral and on the second - 
        the rest of the string, i.e. the noun/the rest of the noun phrase
    '''
    toks = inp_str.split()
    for ind, val in enumerate(toks):
        token_morphs = morph.parse(val)
        grms = list(token_morphs[0].tag.grammemes)
        if any(['NUMB' in grms, 'intg' in grms, 'NUMR' in grms, 'Anum' in grms, 'несколько' == val]):
            if ((ind + 1 ) / len(toks)) >= 0.5 and len(toks) != 2:
                numb_curr = val
                toks_curr = ' '.join([i for i in toks[:ind] if i not in set([val, 'единиц', 'единицами', 'единицы', 'ед', 'ед.', 
                'ед.:','ед.;', 'ЕД', 'ЕД.', 'ЕД.:', 'ЕД.;', '-'])])
                return numb_curr, toks_curr
            else:
                numb_curr = ' '.join(toks[:ind+1])
                toks_curr = ' '.join([i for i in toks[ind+1:] if i not in set([val, 'единиц', 'единицами', 'единицы', 'ед', 'ед.', 
                'ед.:', 'ед.;', 'ЕД', 'ЕД.', 'ЕД.:', 'ЕД.;', '-'])])
                return numb_curr, toks_curr
    else:
        return '1', ' '.join(toks)
