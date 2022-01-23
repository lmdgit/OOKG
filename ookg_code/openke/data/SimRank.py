import numpy as np
import ipdb
def parse_line(line):
    line = line.strip().split()
    sub = int(line[0])
    obj = int(line[1])
    rel = int(line[2])
    return sub, obj, rel

def load_neigh_from_txt(filename, filename2, ent, ent_num, rel_num, nei_num, mode='test', parse_line=parse_line):

    ent2ent=np.zeros([ent_num, nei_num])
    ent2weight=np.zeros([ent_num, nei_num])
    sim_num = ent_num
    
    sim_weight=np.log10(rel_num)/np.log10(rel_num*ent_num)
    sim_weight=round(sim_weight,2)
    
    ent2re_nei = dict()
    relList = dict()
    ent2e_nei = dict()

    for i in range(ent_num):
        ent2re_nei[i]=set()
        relList[i]=set()
        ent2e_nei[i]=set()

    with open(filename) as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if i==0:
            continue
        sub, obj, rel = parse_line(line)
        ent2re_nei[sub].add((rel,obj))
        relList[sub].add(rel)
        ent2e_nei[sub].add(obj)
        ent2re_nei[obj].add((-rel-1,sub))
        relList[obj].add(-rel-1)
        ent2e_nei[obj].add(sub)
        
    if mode=='test':
        sim_num=len(ent)
        with open(filename2) as f2:
            lines2 = f2.readlines()
    
        for i, line in enumerate(lines2):
            sub, obj, rel = parse_line(line)
            if sub in ent:
                relList[sub].add(rel)
                ent2re_nei[sub].add((rel,obj))
            if obj in ent:
                relList[obj].add(-rel-1)
                ent2re_nei[obj].add((-rel-1,sub))





    for i in range(sim_num):
        if i%100==0:
            print('Computing the similarity information: {}|{}'.format(i, sim_num))
        if mode=='test':
            i=ent[i]

        for j in range(ent_num):
            tmp1= len(relList[i].intersection(relList[j]))
            if i!=j and tmp1>0:
                tmp2= len(relList[i].union(relList[j]))
                score_r=float(tmp1) /tmp2
                    
                tmp3= len(ent2re_nei[i].intersection(ent2re_nei[j]))
                tmp4= len(ent2re_nei[i].union(ent2re_nei[j]))
                score_er=float(tmp3) /tmp4
                    
                score = (score_er + sim_weight * score_r)/(sim_weight + 1)
                
                Min=ent2weight[i][0]
                index=0
                for k in range(nei_num):
                    if (Min>ent2weight[i][k]):
                        Min=ent2weight[i][k]
                        index=k
                if (score> Min):
                    ent2weight[i][index]=score
                    ent2ent[i][index]=j
                    
    ent2weight=ent2weight.astype('float32')
    if mode=='test':
        np.save('./testfb/e_nei_sim_test', ent2ent.astype('int64'))
        np.save('./testfb/e_weight_sim_test', ent2weight)
    else:
        np.save('./testfb/e_nei_sim_train', ent2ent.astype('int64'))
        np.save('./testfb/e_weight_sim_train', ent2weight)        
    
    return ent2ent.astype('int64'), ent2weight
    



                    
                    
    
