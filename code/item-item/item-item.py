import  files as f
import numpy as np
from scipy.sparse import coo_matrix
maxid = 624958
user_max = 19835

def get_item_record(item_list):
    item_record = {}
    for user in item_list:
        userid = user[0]
        for record in user[2]:
            if user[2][record]!=0:
                if userid not in item_record:
                    item_record[userid] = {}
                item_record[userid][record] = user[2][record]
    return item_record

def get_item_item_matrix(item_record):
    cnt = 0
    for ids in range(1,user_max,10):
        filename = "data/item_matrix2/"+str(int(ids/10))+".pkl" 
        #读取当前块存储的同现矩阵
        try:
            item_matrix = f.readpkl(filename)
            for user in range(ids,ids+10):
                for item1 in item_record[user]:#遍历用户评分的商品，组成商品对(item1,item2)
                    for item2 in item_record[user]:
                        item_pair = (item1,item2)
                        if item_pair not in item_matrix:
                            item_matrix[item_pair] = 0	#初始化字典条目
                        item_matrix[item_pair] = item_matrix[item_pair]+1	#同现矩阵计数+1
            f.writepkl(filename,item_matrix) #保存当前块
            cnt = cnt+1
            print("finish",cnt)
        except EOFError:
            continue
        except FileNotFoundError:
            continue
def get_item_cnt(item_record):
    item_cnt = {}
    for user in range(user_max):
        for item in item_record[user]:
            if item not in item_cnt:
                item_cnt[item] = 0
            item_cnt[item]  = item_cnt[item]+1
    return item_cnt

def ini_sim():
    for ids in range(int(maxid/20)+1):
        filename = "new/sim_matrix/"+str(ids)+".pkl"
        sim_matrix = {}
        f.writepkl(filename,sim_matrix)
def get_similarity(item_cnt):
    cnt = 0
    for ids in range(int(maxid/20)+1):
        filename = "new/sim_matrix/"+str(ids)+".pkl"
        matrix_file = "new/item_matrix/"+str(ids)+".pkl"
        try:
            matrix = f.readpkl(matrix_file)
            sim_matrix = f.readpkl(filename)
            for pair in matrix:
                item1 = pair[0]
                item2 = pair[1]
                if item1 not in sim_matrix:
                    sim_matrix[item1] = {}
                sim_matrix[item1][item2] = matrix[pair]/(item_cnt[item1]*item_cnt[item2])
            f.writepkl(filename,sim_matrix)
            cnt = cnt+1
            print ("finish",cnt)
        except FileNotFoundError:
            continue
def predict(item_record,user_item_record):
    cnt = 0
    for user in item_record:
        for item_predict in item_record[user]:
            predict_score = 0
            filename = "data/sim_matrix/"+str(int(int(item_predict)/10))+".pkl" #读取物品相似度矩阵
            sim_matrix = f.readpkl(filename)
            try:
                for item in user_item_record[user]:
                    if item in sim_matrix[item_predict]:# 计算评分
                        predict_score = predict_score+user_item_record[user][item]
                        				*sim_matrix[item_predict][item]
                item_record[user][item_predict] = predict_score #存储对应的预测评分
            except KeyError:
                continue
        cnt = cnt+1
        print ("finish",cnt)
    f.writepkl("new/result.pkl",item_record)
def ini_item_item():
    for ids in range(1,user_max,10):
        filename = "data/item_matrix2/" + str(int(ids / 10)) + ".pkl"
        item_matrix = {}
        f.writepkl(filename,item_matrix)
    print("ini ok")

if __name__ == '__main__':
    train_record = f.readpkl("data/item_record.pkl")
    test_record = f.readpkl("data/test_record.pkl")
    item_cnt = f.readpkl("data/item_cnt.pkl")
    item_dict = {}
    predict2(train_record=train_record,test_record=test_record,item_cnt=item_cnt,item_dict= item_dict)
    print ("ok")
    
    test_record = f.readpkl("test_item.pkl")
    item_cnt = get_item_cnt(item_record=train_record)
    item_dict = {}
    predict2(train_record=train_record,test_record=test_record,item_cnt=item_cnt,item_dict= item_dict)
    print("ok")'''
    
