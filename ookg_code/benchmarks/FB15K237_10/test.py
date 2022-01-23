import numpy as np
import ipdb
#随机种子

def read_test_triplets(file_name,file_name2):
    data = []
    ent=set()
    ent_random=set()

    file = open(file_name)
    for i, line in enumerate(file):
        head, tail, relation = line.strip().split()
        data.append((head, relation, tail))
    num= len(data)
    n2=int(num*0.1)
    entlist= np.random.choice(num, size=n2, replace=False)
    ent=list(ent)
    with open(file_name2, 'wt') as w:
        for i in range(num):
            if i in entlist:
                w.write('{} {} {}\n'.format(data[i][0], data[i][2], data[i][1]))
        
    
    #ipdb.set_trace()
    return 1

read_test_triplets('auxiliary_ind.txt', 'auxiliary_ind_0.1.txt')
