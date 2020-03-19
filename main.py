import pandas as pd
import numpy as np
from ast import literal_eval
from gui import UserInterface
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from modAL.models import ActiveLearner
from sklearn.ensemble import RandomForestClassifier
import random
import csv
import os
from sklearn.model_selection import train_test_split


class AccuracyMeasure:
    def __init__(self):
        self.number_bad = 0
        self.number_good = 0
        self.number_total = 0

    def update(self, user_score):
        if user_score != 0:
            self.number_total += 1
        if user_score == -1:
            self.number_bad += 1
        elif user_score == 1:
            self.number_good += 1

    def print_score(self):
        if self.number_total != 0:
            print("Accuracy: {}%".format((self.number_good / self.number_total) * 100))
        else:
            return 0

# constant which determines the amount of movies in a genre's top
topX = 40
df = pd.read_csv('data/movieData_Dummie.csv')
df['user_score'] = -2
score_writer = csv.writer(open('data/user/scored.csv', 'a'))
UI = UserInterface()
scoredArr = []  # array where all the imdb ids and scores are handled.

learner = ActiveLearner(
    estimator=RandomForestClassifier(),
)
AM = AccuracyMeasure()


def main():
    begin()
    UI.run()

    return 0


def begin():
    # get the top genres directly from the top 100
    tmp_df = pd.read_csv('data/topMovies/top100.csv')

    for index in range(5):
        movieIndex = random.randint(0, 100)
        tmp_movie = tmp_df.iloc[movieIndex]
        UI.add_movie(tmp_movie.imdb_id)

    return 0


def choose_new():
    top_genre_list = os.listdir('data/topMovies/')
    # Pick a random genre
    random_genre = random.randint(0, len(top_genre_list) - 1)
    genre = top_genre_list[random_genre]
    tmp_df = pd.read_csv('data/topMovies/' + genre)

    # Pick a random movie
    tmp_movie = df.loc[df['imdb_id'] == tmp_df.loc[random.randint(0, tmp_df.shape[0])]['imdb_id']]

    imdb_id_movietoadd = -1

    if int(tmp_movie['user_score']) != -2:
        print('movie already rated')
        choose_new()
    elif len(df[df['user_score'] != -2]) > 3:
        print('using predictor to pick movies')
        imdb_id_movietoadd = predictor()
    elif int(tmp_movie['user_score']) == -2:
        print('picking random movie')
        imdb_id_movietoadd = tmp_movie.imdb_id.values[0]

    print(f'adding movie {imdb_id_movietoadd}')
    UI.add_movie(imdb_id_movietoadd)
    # from UI pass_user_score() is called which calls choose_new() again after UI.add_movie()


# print(UI.get_movieList())

# topX = the amount of movies we want in the genre specific database
def createTable(genre, indexOfBest, df, topX):
    # TODO Goeie documentatie wat deze functie doet
    topMoviesIndex = []
    count = 0

    for index in indexOfBest:

        if genre in df.iloc[index]['genres'] and count < topX:
            count += 1
            topMoviesIndex.append(index)

    dfTopMovies = pd.DataFrame(df.iloc[topMoviesIndex[0]])

    # wrong dimensions
    for index in topMoviesIndex[1:]:
        tempDf = pd.DataFrame(df.iloc[index])
        dfTopMovies = pd.concat([dfTopMovies, tempDf], axis=1)

    # flip it to get the correct dimensions
    dfTopMovies = dfTopMovies.transpose()

    # create csv file for the top movies
    dfName = 'data/topMovies/' + genre + '.csv'
    dfTopMovies.to_csv(dfName, sep=',', encoding='utf-8', index=False)

    return 0


def topXTables(topX, df):
    # TODO Goeie documentatie wat deze functie doet
    genres = dict()

    for index, subset in df.iterrows():
        genresArr = literal_eval(subset.genres)
        for genre in genresArr:
            if genre in genres:
                genres[genre] += 1
            else:
                genres[genre] = 1

    indexOfBest = df.weightedRating.sort_values(ascending=False).index
    for key in genres:
        genreTop = createTable(key, indexOfBest, df, topX)
        print("done ", key)

    return 0


def createTop100(df):
    # TODO Goeie documentatie wat deze functie doet
    dfTopInd = []

    for index in df.weightedRating.sort_values(ascending=False).head(100).index:
        dfTopInd.append(index)

    dfTop = pd.DataFrame(df.iloc[dfTopInd[0]])

    # wrong dimensions
    for index in dfTopInd[1:]:
        tempDf = pd.DataFrame(df.iloc[index])
        dfTop = pd.concat([dfTop, tempDf], axis=1)

    # flip it to get the correct dimensions
    dfTop = dfTop.transpose()

    dfTop.to_csv('data/topMovies/top100.csv', sep=',', encoding='utf-8', index=False)
    return 0


# This is where we get the title of the movie and the users score
def pass_user_score(score, imdb):
    if score != 0:
        df.loc[df['imdb_id'] == imdb, 'user_score'] = score

    # And here we write the scored movie to a csv file
    row = df.loc[df['imdb_id'] == imdb].iloc[0]
    score_writer.writerow([row['imdb_id'], row['user_score']])
    scoredArr.append((row['imdb_id'], row['user_score']))

    curr_inst = np.array(df[df['imdb_id'] == imdb].select_dtypes(exclude=['object']).iloc[:, :-1])
    learner.teach(curr_inst.reshape(1,-1), np.array(1).reshape(1,-1))
    choose_new()
    # #
    # # if len(scoredArr) < 20:
    # #
    # #
    # # else:
    # #     print("classification here")
    # #     predictor()
    # #
    # # return 0


# TODO construct a predictor for the new suggestions based on a decision tree
def predictor():
    prediction = -1

    non_rated = df[df['user_score'] == -2]
    rated = df[df['user_score'] != -2]
    rated = rated[rated['user_score'] != 0]
    
    # non_rated = non_rated.select_dtypes(exclude=['object'])
    rated = rated.select_dtypes(exclude=['object'])

    X = np.array(rated.iloc[:, :-1])
    y = np.array(rated.iloc[:, -1])

    X_non_rated = np.array(df[df['user_score'] == -2].iloc[:, :-1])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    explore_thresh = 1
    if random.random() < explore_thresh:
        # initializing the learner
        # query for labels
        query_idx, query_sample = learner.query(X_non_rated)
        # print(f'query_idx {query_idx}')
        print(f'query_sample {query_sample}')
        # TODO actually link query_idx to prediction id
        prediction = query_idx
    else:
        DTC = RandomForestClassifier(n_estimators=100)
        DTC.fit(X,y)


        #TODO make batches of random movies that have -2 as userscore, (batches of 1000)
        #we chose the one with the highest mean weightedrating
        non_rated_shuffle = shuffle(non_rated)
        splitArrays = np.array_split(non_rated_shuffle, 43)
        maxBatch = -1

        count = 0
        for dataFrame in splitArrays:

            meanRating = dataFrame['weightedRating'].mean()

            if meanRating > maxBatch:
                maxBatch = meanRating
                index = count
                dfBatch = dataFrame

            count += 1


        print("maxBatch = ", meanRating, "index = " ,index)

        results = DTC.predict(np.array(dfBatch.select_dtypes(exclude=['object']).iloc[:, :-1].fillna(0)))
        dfBatch.reset_index()

        index_score_good = []
        index_score_bad = []
        for i in range(len(results)):
            if results[i] != 1:
                index_score_bad.append(dfBatch.iloc[i]['imdb_id'])
            if results[i] == 1:
                index_score_good.append(dfBatch.iloc[i]['imdb_id'])

        print(len(index_score_good), " - ", len(index_score_bad))

        if len(index_score_good) > 1:
            next_choice = index_score_good[random.randint(0, len(index_score_good)-1)]
            # print(next_choice)
            UI.add_movie(next_choice)
        else:
            choose_new()

    return prediction

if __name__ == '__main__':
    main()
