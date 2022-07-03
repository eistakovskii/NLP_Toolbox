from bs4 import BeautifulSoup
import requests
import pandas as pd
import Verb_unp as vu
from requests.utils import quote


########ANIMATION IMPORTS#######
import itertools
import threading
import time
import sys
################################


verbs_list = vu.wiki_list()


# ##################TO CREATE NEW URLS########################################
w_url = 'https://de.wiktionary.org/wiki/'

n_url_l = list()
for v in verbs_list:
    vn = quote(v)
    n_url = w_url + vn
    n_url_l.append(n_url)

####################### CODE TO FINALLY CHECK THE VERBS ON WIKI M'KAY #############################

v_404 = list()

d_verbs = dict(zip(verbs_list, n_url_l))

d_verbs_v = d_verbs.keys()

final_dict_UNP = list()
final_dict_P = list()
final_dict_inf = list()

########################LOADING ANIMATION###############################

done = False
#here is the animation
def animate():
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if done:
            break
        sys.stdout.write('\rProcessing ' + c)
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write('\rDone!     ')

t = threading.Thread(target=animate)
t.start()

#######################################################

for k in d_verbs_v:
    vt = k
    l = d_verbs[k]
    url = l
    try:
        dataframe_list = pd.read_html(url, flavor='bs4')
    except:
        v_404.append(vt)               
    else:
        v_t = pd.read_html(url, match="Präsens", flavor='bs4')[0]
        test_value = v_t['Wortform'][0]
        #print('Ich: ', test_value)
        test_value2 = v_t['Wortform'][2]
        #print('Sie,es,sie: ', test_value2)
        if len(test_value2) <= 3:
            final_dict_inf.append(k)
            # print('The infinitive verb: ', k)
            # print(len(test_value2))
        else:    
            if len(test_value) <= 3:
                final_dict_UNP.append(k)
                # print('The impersonal verb: ', k)
                # print(len(test_value))
            else:
                final_dict_P.append(k)
                # print('The normal verb: ', k)
                # print(len(test_value))
    #input()


# print('NOT FOUND\n', v_404)
# print('UNPERSONLICH\n', final_dict_UNP)
# print('PERSONLICH\n', final_dict_P)


#########################SAVE THE VERBS LISTS############################################################

with open('verbs_not_found.txt', 'w') as myfile1:
    for i in v_404:
        myfile1.write(i + '\n')

with open('impersonal_verbs.txt', 'w') as myfile2:
    for i in final_dict_UNP:
        myfile2.write(i + '\n')

with open('personal_verbs.txt', 'w') as myfile3:
    for i in final_dict_P:
        myfile3.write(i + '\n')

with open('only_infinitives_verbs.txt', 'w') as myfile4:
    for i in final_dict_inf:
        myfile4.write(i + '\n')

#long process here
time.sleep(10)
done = True


####################################################################################################

################### Code to creat the verbs URL list
# with open('verbs_urls.txt', 'w') as myfile:
#     for i in n_url_l:
#         myfile.write(i + '\n')
###################



########################FETCH CODE PART############################################

# url = "https://de.wiktionary.org/wiki/ähnlichsehen"

# dataframe_list = pd.read_html(url, flavor='bs4')

# #print(len(dataframe_list))


# #print(dataframe_list[0])

# v_t = pd.read_html(url, match="Präsens", flavor='bs4')[0]

# print(v_t.columns)

# print(v_t['Wortform'][0])

#######################################################################