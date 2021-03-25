import files

# filename = 'data/train_res.txt'
# with open(filename, 'a') as file_obj:

raw_data = files.readpkl('data/train_stat.pkl')
all_person = 19835
all_item = 624961
len_list = []
sim_list = []


# sim_list = [[0 for i in range(0, all_person)] for i in range(0, all_person)]


def initial(raw_data):
    init_list = []
    for i in raw_data:
        init_list.append(i[2])
    return init_list


def pre_process(person_list):
    for i in person_list:
        sum = 0
        for value in i.values():
            sum += value
        average = sum / all_item
        sum = 0
        for key in i.keys():
            i[key] = i[key] - average
            sum += i[key] ** 2
        sum = sum ** 0.5
        # if sum <= 0:
        #     print(sum),
        #     print(i),
        len_list.append(sum)
    return person_list


def process(person_list):
    file_count = 0
    for tmp_i in range(0, all_person):
        for tmp_j in range(0, all_person):
            flag = len(person_list[tmp_i]) <= len(person_list[tmp_j])
            cos_sum = 0
            sim = 0
            if len_list[tmp_i] != 0 and len_list[tmp_j] != 0:
                if flag:
                    for key in person_list[tmp_i].keys():
                        if key in person_list[tmp_j].keys():
                            cos_sum += person_list[tmp_i][key] * person_list[tmp_j][key]
                else:
                    for key in person_list[tmp_j].keys():
                        if key in person_list[tmp_i].keys():
                            cos_sum += person_list[tmp_i][key] * person_list[tmp_j][key]
                sim = cos_sum / (len_list[tmp_i] * len_list[tmp_j])
            sim_list.append(sim)
        if tmp_i % 100 == 99:
            files.writepkl('data/train_res%d.pkl' % file_count, sim_list)
            file_count += 1
            sim_list.clear()
            print((tmp_i, tmp_j, sim)),
    files.writepkl('data/train_res%d.pkl' % file_count, sim_list)
    print("all files finishi")


if __name__ == "__main__":
    init_list = initial(raw_data)
    # print(init_list[0]),
    res_list = pre_process(init_list)
    # print(res_list[0]),
    # print(len_list),
    process(res_list)
    print("Finish process!")
