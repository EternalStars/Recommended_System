import codecs
import pickle

def readtxt(filename):
    with codecs.open(filename, encoding="utf-8") as f:
        _data = f.readlines()
        data = [item.rstrip('\r\n') for item in _data]
    return data

def writetxt(filename, data):
    with codecs.open(filename, 'w', encoding='utf-8') as g:
        for line in data:
            g.write(str(line) + '\n')
    print("finish writing in %s !" % filename)

def readpkl(filename):
    f = open(filename, 'rb')
    data = pickle.load(f)
    f.close()
    return data

def writepkl(filename, data):
    g = open(filename, 'wb')
    pickle.dump(data, g)
    g.close()
    print("finish writing in %s !" % filename)