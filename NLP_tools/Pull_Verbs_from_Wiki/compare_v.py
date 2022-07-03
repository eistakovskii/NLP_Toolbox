imp_verbs_ = list()
all_verbs_ = list()
inf_verbs_ = list()

with open('impersonal_verbs.txt', 'r') as mf1:
    lines_imp = mf1.readlines()
    for i in lines_imp:
        v = i.rstrip()
        imp_verbs_.append(v)

with open('only_infinitives_verbs.txt', 'r') as mf3:
    lines_imp = mf3.readlines()
    for i in lines_imp:
        v = i.rstrip()
        inf_verbs_.append(v)


with open('all_german_verbs.txt', 'r') as mf2:
    lines_a = mf2.readlines()
    for i in lines_a:
        v = i.rstrip()
        all_verbs_.append(v)


#print(all_verbs_)

set_a = set(imp_verbs_)
set_p = set(all_verbs_)
set_i = set(inf_verbs_)

#print(set_a & set_p)

intrc = set_a.intersection(set_p)   

print(len(intrc))

intrc2 = set_a.intersection(set_i)

print(intrc2)

with open('reverso_impersonal_verbs_to_check_final.txt', 'w') as myfile:
    for i in intrc:
        myfile.write(i + '\n')