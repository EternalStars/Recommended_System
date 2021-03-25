import files
import heapq

num_of_people = 19835
relative_people_num = 3
item_per_person = 6

train_stat = files.readpkl('data/train_stat.pkl')
test_stat = files.readpkl('data/test_stat.pkl')


def initial(raw_data):
    init_list = []
    for i in raw_data:
        init_list.append(i[2])
    return init_list


def find_relative_people(user_id, item_id):
    people_list = []
    sim_list = []
    relative_people = []
    relative_sim = []
    file_num = int(user_id / 100)
    tmp_list = files.readpkl('data/train_res%d.pkl' % file_num)
    index = user_id % 100
    start = index * num_of_people
    for i in range(0, num_of_people):
        if item_id in train_stat[i][2].keys() and user_id != i:
            relative_people.append(i)
            relative_sim.append(tmp_list[start + i])
    if len(relative_people) ==0:
        return people_list,sim_list
    ordered = heapq.nlargest(relative_people_num, relative_sim)
    minimum = ordered[-1]
    for i in range(0, len(relative_people)):
        if relative_sim[i] >= minimum and i != user_id:
            people_list.append(relative_people[i])
            sim_list.append(relative_sim[i])
    return people_list, sim_list


if __name__ == "__main__":
    person_list = initial(train_stat)
    counter = 0
    for i in range(0, len(test_stat)):
        for key in test_stat[i][2].keys():
            scored_people, scored_sim = find_relative_people(i, key)
            if len(scored_people)==0:
                test_stat[i][2][key] = 0
                continue
            sim_sum=0
            for j in range(0,len(scored_sim)):
                sim_sum +=scored_sim[j]
            # print(coefficience)
            # print('sim sum:%f' % sim_sum)
            sum = 0
            for j in range(0,len(scored_people)):
                sum+=train_stat[scored_people[j]][2][key]*scored_sim[j]
            # print('item sum:%f' % sum)
            if sim_sum != 0:
                sum = sum / sim_sum
            # print('modified sum:%f'%sum)
            test_stat[i][2][key] = round(sum,2)
        # break
        # print(test_stat[i])
        if i % 100 == 99:
            print("Finish %d*100 users" % counter)
            counter+=1
    files.writepkl('data/test_res.pkl', test_stat)
