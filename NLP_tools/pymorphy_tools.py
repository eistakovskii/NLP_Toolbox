import pymorphy2
import razdel

from collections import Counter
import re

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
    inp_str = inp_str.split()

    for ind, val in enumerate(inp_str):
        if re.search(r'(?<=\d),(?=\d)', val) and inp_str[ind+1] in set(['тысяч', 'тыс', 'тысячи', 'тыc.']):
            inp_str[ind] = re.sub(r'(?<=\d),(?=\d)', '', inp_str[ind])
            inp_str[ind + 1] = '00'

    inp_str = ' '.join(['000' if i in set(['тысяч', 'тыс', 'тысячи', 'тыc.']) else i for i in inp_str])
    inp_str = re.sub(r'(\d)\s+(\d)', r'\1\2', inp_str) # Delete spaces between two numbers

    inp_str = re.sub(r'(?<=\d),(?=\d)', '', inp_str)

    excl_syms = set(list("●.\uF02D,€¶№%£♫♪░▒▓♫$│♪╖╣║╕╗╝┤╡╢▸╛�®‰™⏰⌚⌛⏳©¬³°¯²§±‡†¾½¼µ☒☐•!\#&\\()\u0093\u0094*+,/:;<=>?[\\\\]^_`{|}~’‘«»…"))
    excl_toks = set(['шт', 'ед', 'eд', 'ЕД', 'EД']) 

    toks = inp_str.split()
    toks = [i.replace('ед.','') if i.startswith('ед.') and (len(list('ед.')) + 1) < len(list(i)) else i for i in toks] # when two words merged
    toks = [i.replace('eд.','') if i.startswith('eд.') and (len(list('eд.')) + 1) < len(list(i)) else i for i in toks] # when two words merged
    toks = [i.replace('единиц','') if i.startswith('единиц') and (len(list('единиц')) + 3) < len(list(i)) else i for i in toks] # when two words merged
    toks = [''.join([i for i in j if i not in excl_syms]) for j in [list(i) for i in toks]]

    grms_of_str = list()
    for ind, val in enumerate(toks):
        token_morphs = morph.parse(val)
        grms = list(token_morphs[0].tag.grammemes)
        grms_of_str.append(grms)

        if any(['NUMB' in grms, 'intg' in grms, 'NUMR' in grms, 'Anum' in grms, 'несколько' == val]):                          
            if toks[-1] in excl_toks:
                toks_curr = [i for i in toks if i not in excl_toks]
                numb_curr = val
                toks_curr = ' '.join(toks_curr[:-1])
                toks_curr = toks_curr[:-1].strip() if toks_curr.endswith('-') else toks_curr
                return numb_curr.lower(), toks_curr
            elif not any(['intg' in morph.parse(toks[-1])[0].tag.grammemes, 'intg' in morph.parse(toks[-2])[0].tag.grammemes]) or len(toks) <= 4: 
                numb_curr = ' '.join(toks[:ind+1])
                toks_curr = [i for i in toks if i not in set(['единиц', 'eдиниц', 'единицами', 'единицы', 'ед', 'ЕД'])]
                toks_curr = ' '.join(toks_curr[ind+1:])
                toks_curr = toks_curr[:-1].strip() if toks_curr.endswith('-') else toks_curr
                return numb_curr.lower(), toks_curr
    else: 
        try:
            temp_gr_count = Counter([j for i in grms_of_str for j in i])          
            if temp_gr_count.get('plur') >= 2 or (temp_gr_count.get('gent') == None and temp_gr_count.get('plur') == 1) or (temp_gr_count.get('gent') == 1 and temp_gr_count.get('plur') == 1):
                return 'больше двух', ' '.join(toks)
            else:
                return '1', ' '.join(toks)
        except:    
                return '1', ' '.join(toks) 
