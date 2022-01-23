import numpy as np
import ipdb

def read_entity(filename):
    ent=[]
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            ent.append(line)
    return ent
    
def read_train_triplets(file_name):
    sub = dict()

    file = open(file_name)
    for i, line in enumerate(file):
        if i==0:
            continue
        head, tail, relation = line.strip().split()
        if head not in sub.keys():
            sub[head]=1
        else:
            sub[head]=1 + sub[head]
        if tail not in sub.keys():
            sub[tail]=1
        else:
            sub[tail]=1 + sub[tail]
    file.close()

    return sub


def read_test_triplets(file_name1, file_name2, sub, ent):
    file = open(file_name1)
    #ipdb.set_trace()
    with open(file_name2, 'wt') as w:
        for i, line in enumerate(file):
            if i==0:
                continue
            head, tail, relation = line.strip().split()
            if tail in sub.keys():
                if sub[tail]<= 15 :
                    w.write('{} {} {}\n'.format(head, tail, relation))
    return 1

ent= read_entity('entity_ind.txt')
sub = read_train_triplets('auxiliary_ind.txt')
#ipdb.set_trace()
read_test_triplets('test2id.txt', 'test2id2.txt', sub, ent)
