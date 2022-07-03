

all_verbs_list = list()

with open(r"YOUR PATH\verbs_.txt",'r') as myfile:
    lines_t = myfile.readlines()
    for l in lines_t:
        lst_t = l.split('>')
        lst_t2 = lst_t[-2]
        vt = lst_t2.split('<')
        vt = vt[0]
        all_verbs_list.append(vt)

#print(all_verbs_list)


with open('all_german_verbs.txt', 'w') as myfile1:
    for i in all_verbs_list:
        myfile1.write(i + '\n')