import psycopg2
import numpy as np
import re

def getConnection():
    # return psycopg2.connect("dbname='dbname' user='user' host='host' password='password' port='port'")
    raise NotImplementedError

def parseUnitsMap(filename='config/unitsmap.unit'):
    with open(filename, 'r') as f:
        lines = f.readlines()
        length = len(lines)
        t = 0
        unitsmap = {}
        while t < length:
            line = lines[t].rstrip('\r\n')
            if line.startswith('tablename'):
                tablename = line.split(':')[1]
                unitsmap[tablename] = {}
                t += 1
                while t < length:
                    line = lines[t].rstrip('\r\n')
                    if line == '':
                        break
                    res = line.split(',')
                    itemid, mainunit, umapstrs = int(res[0]), res[1], res[2:]
                    umap = {}
                    for umapstr in umapstrs:
                        umapstr = umapstr.split(':')
                        # print(umapstr)
                        umap[umapstr[0]] = float(umapstr[1])
                    unitsmap[tablename][itemid] = {}
                    unitsmap[tablename][itemid]['mainunit'] = mainunit
                    unitsmap[tablename][itemid]['umap'] = umap
                    t += 1
                t += 1
        return unitsmap

UNITSMAP = parseUnitsMap('config/unitsmap.unit')

PAT = re.compile(r'({0}-{0}|{0})'.format(r'(\d+\.\d*|\d*\.\d+|\d+)'))

def parseNum(s):
    try:
        num = float(s);
        return num
    except:
        try:
            res = re.search(r'({0}\:{0})'.format(r'((\d+\.\d*)|(\d*\.\d+)|(\d+))'), s).group()
            res = res.split(':')
            return (float(res[0]) / float(res[1]))
        except:
            pass
        res = re.findall(r"\d*\.\d*|\d+\-\d*\.\d*|\d+", s)
        try:
            if(len(res) > 2 ):
                return None;
            else:
                return (float(res[0]) + float(res[1]))/2.
        except:
            res = re.findall(r"[-+]?\d*\.\d+|\d+", s)
            nums = [];
            for r in res:
                try:
                    nums.append(float(r))
                except:
                    pass;

            if(len(nums) == 0):
                return None;

            nums = np.array(nums).mean()
            return nums

def sparsify(mat):
    coocode = []
    for i in range(len(mat)):
        for j in range(len(mat[i])):
            if(mat[i][j]==None):
                continue;
            else:
                coocode.append((i,j,mat[i][j]))
    return {'timestep':len(mat),'features':len(mat[0]),'codes':coocode};

def coodecode(coocode,f,t):
    mat = []
    for i in range(t):
        mat.append([None]*f);

    for c in coocode:
        mat[c[0]][c[1]] = c[2]

    return mat;

if __name__ == '__main__':
    pass
