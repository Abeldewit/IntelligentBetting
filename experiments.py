import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np


yeah_counter = 0
neutral_counter = 0
meh_counter = 0
total_rated_counter = 0
total_counter = 0
movie_counter = 0


# Amount of movies it does not score (training/exploring time)
treshold = 20
# Amount of movies graded until graph is plotted
test_size = 30



# Holds lists of 5, yeah, meh, neutral, total rated and total counter with each movie
yeahlist = []
neutrallist = []
mehlist = []
totalratedlist = []
totallist = []




def set_counters(t):
    global yeah_counter
    global neutral_counter
    global meh_counter
    global total_rated_counter
    global total_counter
    global yeahlist
    global neutrallist
    global mehlist
    global totalratedlist
    global totallist
    global movie_counter
    global treshold
    global test_size

    if t != 0:
        movie_counter += 1
    if movie_counter > treshold:
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
        yeahlist.append(yeah_counter)
        neutrallist.append(neutral_counter)
        mehlist.append(meh_counter)
        totalratedlist.append(total_rated_counter)
        totallist.append(total_counter)

        if total_counter%test_size==0:
            plot_values()
        # plot_values()

def plot_values():

    global yeahlist
    global neutrallist
    global mehlist
    global totalratedlist
    global totallist
    accuracylist = []

    for i in range(0, len(totalratedlist)):
        if totalratedlist[i] == 0:
            acc = 0
        else:
            acc = yeahlist[i]/totalratedlist[i]
        accuracylist.append(acc)

    print(yeahlist)
    print(totalratedlist)
    fig = plt.figure()
    ax = plt.axes()
    ax.set_ylabel('accuracy')
    ax.set_xlabel('rated movies')


    plt.plot(accuracylist)

    plt.show()


# dummie()
