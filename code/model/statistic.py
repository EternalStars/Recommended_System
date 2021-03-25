import numpy as np
import codecs
import files

# ------数据格式：[<user>, <count>, {<id1>: <score1>, <id2>: <score2>, ...}]------
def stat_loadtrain(filename):
    data = []
    with codecs.open(filename, encoding="utf-8") as f:
        _data = f.readlines()
        for line in _data:
            line = line.rstrip('\r\n')
            if '|' in line:
                data.append([int(item) for item in line.split('|')])
                data[-1].append({})
            else:   # 使用时要把数据集中的空行去掉，不然报错
                data[-1][2][line.split()[0]] = int(line.split()[1])
    return data

def stat_loadtest(filename):
    data = []
    with codecs.open(filename, encoding="utf-8") as f:
        _data = f.readlines()
        for line in _data:
            line = line.rstrip('\r\n')
            if '|' in line:
                data.append([int(item) for item in line.split('|')])
                data[-1].append({})
            else:   # 使用时要把数据集中的空行去掉，不然报错
                data[-1][2][line] = -1
    return data

if __name__ == "__main__":
    data1 = stat_loadtrain('data/train.txt')
    files.writepkl('pro_data/train_stat.pkl', data1)
    
    data2 = stat_loadtest('data/test.txt')
    files.writepkl('pro_data/test_stat.pkl', data2)

    '''
    #data = files.readpkl('data/test_stat.pkl')
    #data = files.readpkl('data/train_stat.pkl')
    print(len(data), data[-1][0])

    items, scores = [], []
    for e in data:
        items += list(e[2].keys())
        scores += list(e[2].values())
    
    items = list(set(items))
    items.sort()
    scores = list(set(scores))
    scores.sort()

    items = np.array(items, dtype=int)
    print(max(items), len(items))
    print(scores, len(scores))
    '''