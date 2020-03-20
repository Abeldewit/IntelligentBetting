yeah_counter = 0
neutral_counter = 0
meh_counter = 0
total_rated_counter = 0
total_counter = 0

# Holds lists of 5, yeah meh, neutral, total rated and total counter with each movie
counter_list = []

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np

def dummie():
    counter_list.append([1,2,5,4,6])
    counter_list.append([5,2,4,3,1])
    counter_list.append([1,2,5,4,6])
    counter_list.append([1,2,5,4,6])
    counter_list.append([1,2,5,4,6])
    counter_list.append([1,2,5,4,6])
    counter_list.append([1,2,5,4,6])
    counter_list.append([1,2,5,4,6])
    counter_list.append([1,2,5,4,6])
    counter_list.append([1,2,5,4,6])
    counter_list.append([1,2,5,4,6])
    counter_list.append([1,2,5,4,6])
    counter_list.append([1,2,5,4,6])
    counter_list.append([1,2,5,4,6])
    counter_list.append([1,2,5,4,6])
    counter_list.append([1,2,5,4,6])
    counter_list.append([1,2,5,4,6])
    counter_list.append([1, 2, 5, 4, 6])
    counter_list.append([5, 2, 4, 3, 1])
    counter_list.append([1, 2, 5, 4, 6])
    counter_list.append([1, 2, 5, 4, 6])
    counter_list.append([1, 2, 5, 4, 6])
    counter_list.append([1, 2, 5, 4, 6])
    counter_list.append([1, 2, 5, 4, 6])
    counter_list.append([1, 2, 5, 4, 6])
    counter_list.append([1, 2, 5, 4, 6])
    counter_list.append([1, 2, 5, 4, 6])
    counter_list.append([1, 2, 5, 4, 6])
    counter_list.append([1, 2, 5, 4, 6])
    counter_list.append([1, 2, 5, 4, 6])
    counter_list.append([1, 2, 5, 4, 6])
    counter_list.append([1, 2, 5, 4, 6])
    counter_list.append([1, 2, 5, 4, 6])
    counter_list.append([1, 2, 5, 4, 6])
    plot_values()


def set_counters(t):
    global yeah_counter
    global neutral_counter
    global meh_counter
    global total_rated_counter
    global total_counter
    global counter_list

    if t == 1:
        yeah_counter += 1
        total_rated_counter += 1

    if t == 0:
        neutral_counter += 1

    if t == -1:
        meh_counter += 1
        total_rated_counter += 1

    total_counter += 1
    # print('experiment score noted : ', t)
    tup = [yeah_counter, neutral_counter, meh_counter, total_rated_counter, total_counter]
    print('tup', tup)
    counter_list.append(tup)

    if total_counter%10==0:
        plot_values()

def plot_values():

    global counter_list

    yeahlist = np.array(counter_list)[0:,0]
    neutrallist = np.array(counter_list)[1:,0]
    mehlist = np.array(counter_list)[2:,0]
    totalratedlist = np.array(counter_list)[3:, 0]
    totallist = np.array(counter_list)[4:,0]
    accuracylist = []
    for i in range(0, len(totalratedlist)):
        acc = yeahlist[i]/totalratedlist[i]
        accuracylist.append(acc)


    print('TOTAL COUNTER', counter_list)

    print(yeahlist)
    print(totalratedlist)
    fig = plt.figure()
    ax = plt.axes()

    plt.plot(accuracylist)

    plt.show()


# dummie()
