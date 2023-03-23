def get_pp(key_w: str, sent: str, , verbose: bool=False):
  """
  DESCRIPTION:
  This function takes in a sentence in russian and a target noun and returns a prepositional group with this targer noun.
  E.g. sent - Мне нужной найти картину с портретом поэта, которуя я упоминал вчера.
       key_w - картин
       output - картину с портретом поэта
  INPUT:
  key_w: head word of your prepositional phrase
  sent: your input sentence
  
  Output:
  The function returns a string with the target prepositional group phrase
  """
    check_by = key_w
    sentence = sent
    doc = nlp(sentence)
    out_l = list()
    for token in doc:
        if check_by in token.text:
            childr_head = list(token.children)
            out_l.append(token)
            for i in childr_head:
                if i.dep_ == 'nmod':
                    temp_list = list(i.children)
                    out_l.append(i)
                    out_l.extend(i.children)
                    print(temp_list)
                    for j in temp_list:
                        out_l.append(j)
                        out_l.extend(j.children)
                        for k in j.children:
                            out_l.append(k)
                            out_l.extend(k.children)

    out_dict = dict()
    for item in out_l:
        temp_str = item.text
        temp_ind = item.i
        out_dict[temp_ind] = temp_str
    srt_l = sorted(out_dict.items(), key=lambda kv: kv[0])
    out_phrase = list()
    for i in srt_l:
        out_phrase.append(i[1])
    out_phrase = " ".join(out_phrase)
    
    if verbose:
      print(f'\nSENTENCE: {sent}')
      print(f'\nPHRASE: {out_phrase}')
    
    return out_phrase
