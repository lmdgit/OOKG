import numpy as np
np.random.seed(16)
def parse_line(line):
    line = line.strip().split()
    sub = int(line[0])
    obj = int(line[1])
    rel = int(line[2])
    return sub, obj, rel

def read_entity(filename):
    ent=[]
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            ent.append(int(line))
    return ent
    
def load_supports_from_txt(filename, filename2, ent, ent_num, rel_num, nei_num, padding=True, parse_line=parse_line):


    relList_s = dict()
    relList_o = dict()
    for i in range(ent_num):
        relList_s[i]=[]
        relList_o[i]=[]

    with open(filename) as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if i==0:
            continue
        sub, obj, rel = parse_line(line)
        relList_s[sub].append((rel,obj))
        relList_o[obj].append((rel,sub))
    
    with open(filename2) as f2:
        lines2 = f2.readlines()
    for i, line in enumerate(lines2):
        sub, obj, rel = parse_line(line)
        if sub in ent:
            relList_s[sub].append((rel,obj))               
        if obj in ent:
            relList_o[obj].append((rel,sub))

    for i in range(ent_num):
        if len(relList_s[i])==0:
            relList_s[i].append((rel_num, i))
        if len(relList_o[i])==0:
            relList_o[i].append((rel_num, i))
    for i in range(ent_num):
        if len(relList_s[i])==0 or len(relList_o[i])==0:
            print(i)

    as_mask=np.ones([ent_num, nei_num])
    ao_mask=np.ones([ent_num, nei_num])
    
    for i in range(ent_num):
        a = []
        b = []
        a2 = []
        b2 = []
        if padding:
            if len(relList_s[i]) >= nei_num:
                sampled_neighbors = np.random.choice(len(relList_s[i]), size=nei_num,
                                                     replace=False)
            else:
                sampled_neighbors = [j for j in range(len(relList_s[i]))]
                for j in range(nei_num-len(relList_s[i])):
                    as_mask[i][j+len(relList_s[i])]=0
            if len(relList_o[i]) >= nei_num:
                sampled_neighbors2 = np.random.choice(len(relList_o[i]), size=nei_num,
                                                     replace=False)
            else:
                sampled_neighbors2 = [j for j in range(len(relList_o[i]))]
                for j in range(nei_num-len(relList_o[i])):
                    ao_mask[i][j+len(relList_o[i])]=0
            if len(relList_s[i]) == 1 and relList_s[i][0][0] == rel_num:
                as_mask[i][0]=0
            if len(relList_o[i]) == 1 and relList_o[i][0][0] == rel_num:
                ao_mask[i][0]=0
        else:
            sampled_neighbors = np.random.choice(len(relList_s[i]), size=nei_num,
                                                 replace=len(relList_s[i]) < nei_num)
            sampled_neighbors2 = np.random.choice(len(relList_o[i]), size=nei_num,
                                                 replace=len(relList_o[i]) < nei_num)
        
        for j in range(nei_num):
            if j < len(relList_s[i]):
                tmp=sampled_neighbors[j]
                a.append(relList_s[i][tmp][1])
                b.append(relList_s[i][tmp][0])
            else:
                a.append(i)
                b.append(rel_num)
            if j < len(relList_o[i]):
                tmp=sampled_neighbors2[j]
                a2.append(relList_o[i][tmp][1])
                b2.append(relList_o[i][tmp][0])
            else:
                a2.append(i)
                b2.append(rel_num)
        if i==0:
            nei_ent_s=np.array(a)
            nei_rel_s=np.array(b)
            nei_ent_o=np.array(a2)
            nei_rel_o=np.array(b2)
        else:
            nei_ent_s=np.vstack((nei_ent_s,np.array(a)))
            nei_rel_s = np.vstack((nei_rel_s, np.array(b)))
            nei_ent_o=np.vstack((nei_ent_o,np.array(a2)))
            nei_rel_o = np.vstack((nei_rel_o, np.array(b2)))
    as_mask=as_mask.astype('float32')
    ao_mask=ao_mask.astype('float32')

    np.save('./testfb/nei_ent_s', nei_ent_s)
    np.save('./testfb/nei_rel_s', nei_rel_s)
    np.save('./testfb/as_mask', as_mask)
    np.save('./testfb/nei_ent_o', nei_ent_o)
    np.save('./testfb/nei_rel_o', nei_rel_o)
    np.save('./testfb/ao_mask', ao_mask)
    return nei_ent_s, nei_rel_s, as_mask, nei_ent_o, nei_rel_o, ao_mask

