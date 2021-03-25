import files
import heapq

num_of_people = 19835
relative_people_num = 3
item_per_person = 6

train_stat = files.readpkl('data/train_stat.pkl')
test_stat = files.readpkl('data/test_stat.pkl')
train_res = files.readpkl('data/train_res0.pkl')

if __name__ == "__main__":
    print(test_stat[12])
    people = train_res[12*num_of_people:12*num_of_people+num_of_people]
    print('people',people)

    for key in test_stat[12][2].keys():
        print('==============')
        print(key)
        scored=[]
        scored_sim=[]
        for i in range(0, len(train_stat)):
            if (key in train_stat[i][2].keys() and i!=12):
                scored.append(i)
                scored_sim.append(people[i])
        print('scored',scored)
        print('scored_sim',scored_sim)
        ordered = heapq.nlargest(relative_people_num, scored_sim)
        print('ordered',ordered)
        if(len(ordered)==0):
            print("warning")
        minimum = ordered[-1]
        print('minmum',minimum)
        relative_people = []
        for i in range(0, len(scored)):
            if (scored_sim[i] >= minimum and i != 0):
                relative_people.append(scored[i])
                print(scored[i], scored_sim[i])
        print('-------------------')
        for i in relative_people:
            print(train_stat[i])
            print(train_stat[i][2][key])
    print('==============')
