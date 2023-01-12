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

def get_num(inp_str: str, verbose = False) -> tuple:
    '''
    Use this function to separate a numeral and noun from the input string in Russian
    
    Input:
        inp_str: a string to extract the numeral from
        verbose: True means enabling the comments for debugging

    Output:
        _ : tuple containing on the first place the numeral and on the second - 
        the rest of the string, i.e. the noun/the rest of the noun phrase
    '''
    print('\n')
    if verbose: print(inp_str)
    excl_syms = set(list("●.\uF02D,€¶№%£♫♪░▒▓♫$│♪╖╣║╕╗╝┤╡╢▸╛�®‰™⏰⌚⌛⏳©¬³°¯²§±‡†¾½¼µ☒☐•!\#&\\()\u0093\u0094*+,/:;<=>?[\\\\]^_`{|}~’‘«»…"))

    toks = inp_str.split()
    toks = [''.join([i for i in j if i not in excl_syms]) for j in [list(i) for i in toks]]

    grms_of_str = list()
    for ind, val in enumerate(toks):
        token_morphs = morph.parse(val)
        grms = list(token_morphs[0].tag.grammemes)
        grms_of_str.append(grms)
        if verbose: print(grms)
        if any(['NUMB' in grms, 'intg' in grms, 'NUMR' in grms, 'Anum' in grms, 'несколько' == val]):
            toks_curr = [i for i in toks if i not in set(['единиц', 'eдиниц', 'единицами', 'единицы', 'ед', 'ЕД'])]
            toks_curr = [i.replace('ед.','') for i in toks_curr] # when two words merged
            toks_curr = [i.replace('единиц','') for i in toks_curr] # when two words merged
            if toks_curr[-1] == val:
                numb_curr = val
                toks_curr = ' '.join(toks_curr[:-1])
                return numb_curr, toks_curr
            else:
                numb_curr = ' '.join(toks_curr[:ind+1])
                toks_curr = ' '.join(toks_curr[ind+1:])
                return numb_curr, toks_curr
    else: 
        if any([True for i in grms_of_str for j in i if j == 'plur']):
            return 'больше двух', ' '.join(toks) 
        else:
            return '1', ' '.join(toks)
