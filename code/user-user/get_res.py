import files

filename = "data/k=3/answer.txt"
res = files.readpkl('data/k=3/test_res.pkl')
with open(filename, 'w')as file_obj:
    for i in range(0, len(res)):
        file_obj.write(str(res[i][0]) + '|' + str(res[i][1])+'\n')
        for key in res[i][2].keys():
            file_obj.write(key + ' ' + str(int(res[i][2][key])) + '\n')
