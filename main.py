import pandas as pd
import numpy as np
from ast import literal_eval
from modAL.utils import multi_argmax
from sklearn.exceptions import NotFittedError
from gui import UserInterface
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling, entropy_sampling, margin_sampling
from sklearn.ensemble import RandomForestClassifier
import random
import csv
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

is_calculatecossim = True
n_iteration = 0


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


dfTitles = pd.read_csv('data/MovieData.csv')


class CosineSimilarity:
    def __init__(self):
        global dfTitles
        links_small = pd.read_csv('data/links_small.csv')

        links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')
        dfTitles['id'] = dfTitles['id'].astype('int')
        dfTitles = dfTitles[dfTitles['id'].isin(links_small)]
        dfTitles['titleOverview'] = dfTitles['Title'] + dfTitles['overview']
        tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
        tfidf_matrix = tf.fit_transform(dfTitles['titleOverview'])

        self.cos = linear_kernel(tfidf_matrix, tfidf_matrix)

    def getCos(self):
        return self.cos

# constant which determines the amount of movies in a genre's top
topX = 40
df = pd.read_csv('data/movieData_Dummie.csv')

df['user_score'] = -2
score_writer = csv.writer(open('data/user/scored.csv', 'a'))
UI = UserInterface()
scoredArr = []  # array where all the imdb ids and scores are handled.

def custom_sampling(classifier, X_pool):
    popularity_colidx = 3
    popularity_median = np.median(X_pool[:, popularity_colidx])
    bool_arr = np.apply_along_axis(lambda row: row[popularity_colidx] > popularity_median, 1, X_pool)
    X_pool = X_pool[bool_arr]
    try:
        classwise_uncertainty = classifier.predict_proba(X_pool)
    except NotFittedError:
        return np.ones(shape=(X_pool.shape[0],))
    uncertainty = 1 - np.max(classwise_uncertainty, axis=1)
    query_idx = multi_argmax(uncertainty, n_instances=1)
    return query_idx, X_pool[query_idx]

learner = ActiveLearner(
    estimator=RandomForestClassifier(),
    query_strategy=custom_sampling
)
AM = AccuracyMeasure()
CS = CosineSimilarity()


def main():
    begin()
    UI.run()
    return 0


def begin():
    # get the top genres directly from the top 100
    tmp_df = pd.read_csv('data/topMovies/top100.csv')
    print(df.select_dtypes(exclude='object').head())
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
    tmp_movie = df.loc[df['imdb_id'] == tmp_df.loc[random.randint(0, len(tmp_df) - 1)]['imdb_id']]

    imdb_id_movietoadd = -1

    if int(tmp_movie['user_score']) != -2:
        print('movie already rated')
        choose_new()
    elif len(df[df['user_score'] != -2]) >= 0:  # change to number larger than 0 if you want a 'queue' of movies
        print('using predictor to pick movies')
        imdb_id_movietoadd = predictor()
    elif int(tmp_movie['user_score']) == -2:
        print('picking random movie')
        imdb_id_movietoadd = tmp_movie.imdb_id.values[0]

    print(f'adding movie {imdb_id_movietoadd}')
    UI.add_movie(imdb_id_movietoadd)
    global n_iteration
    n_iteration += 1
    # from UI pass_user_score() is called which calls choose_new() again after UI.add_movie()

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
    df.loc[df['imdb_id'] == imdb, 'user_score'] = score

    # And here we write the scored movie to a csv file
    row = df.loc[df['imdb_id'] == imdb].iloc[0]
    score_writer.writerow([row['imdb_id'], row['user_score']])
    scoredArr.append((row['imdb_id'], row['user_score']))

    curr_inst = np.array(df[df['imdb_id'] == imdb].select_dtypes(exclude=['object']).iloc[:, :-1].fillna(0))
    learner.teach(curr_inst.reshape(1, -1), np.array(score).reshape(1, -1)[0])
    choose_new()
    AM.update(score)
    AM.print_score()

# TODO construct a predictor for the new suggestions based on a decision tree
def predictor():
    cosine_sim = CS.getCos()
    prediction = -1

    non_rated = df[df['user_score'] == -2]
    rated = df[df['user_score'] != -2]
    rated = rated[rated['user_score'] != 0]

    # non_rated = non_rated.select_dtypes(exclude=['object'])
    # rated = rated.select_dtypes(exclude=['object'])

    X = np.array(rated.iloc[:, :-1].select_dtypes(exclude=['object']))
    y = np.array(rated.iloc[:, -1])

    X_non_rated = non_rated.iloc[:, :-1].fillna(0).select_dtypes(exclude=['object'])
    from gui import sliderValue
    explore_thresh = sliderValue

    if random.random() > explore_thresh or n_iteration == 0:
        print('exploring')
        query_idx, query_sample = learner.query(np.array(X_non_rated))

        tmp_id = non_rated['imdb_id'].iloc[query_idx].values[0]
        # TODO fix hacky way to deal with delay when rating
        df.loc[df['imdb_id'] == str(tmp_id), 'user_score'] = 0
        prediction = tmp_id
    else:
        print('exploiting')
        DTC = RandomForestClassifier(n_estimators=100)
        DTC.fit(X, y)
        feature_imp = DTC.feature_importances_
        column_index_most = np.where(feature_imp == np.max(feature_imp))
        # print("Most important features:", list(rated.columns[column_index_most]))

        # TODO make batches of random movies that have -2 as userscore, (batches of 1000)
        # we chose the one with the highest mean weightedrating
        links_small = pd.read_csv('data/links_small.csv')
        links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')

        movies = []


        ratedShuf = shuffle(rated)

        for index, row in ratedShuf.iterrows():
            if row['id'] in links_small and len(movies) < 1000:
                print(row['Title'])
                cosMov  = get_recommendations(row['Title'], cosine_sim)

                for imdbID in cosMov.values:
                    if imdbID in non_rated['imdb_id'].values:
                        movies.append(imdbID)


        print(movies)

        if len(movies) < 1000:
            non_rated_shuffle = shuffle(non_rated)

            # make sure to fill up the movies up to batches where 50% of the movies in the batch are chosen by the
            # cosine similarity function
            if len(movies) > 0:
                threshold = len(movies)/0.5
                splitVal = len(non_rated_shuffle)/(threshold - len(movies))
            else:
                splitVal = len(non_rated_shuffle)/(100 - len(movies))

            print(splitVal)
            splitArrays = np.array_split(non_rated_shuffle, splitVal)
            maxBatch = -1

            count = 0
            for dataFrame in splitArrays:

                meanRating = dataFrame['weightedRating'].mean()

                if meanRating > maxBatch:
                    maxBatch = meanRating
                    index = count
                    dfBatch = dataFrame

                count += 1

        print('dfBatch before conc = ', len(dfBatch))

        index_movies = []
        if len(movies) > 0:
            for i in movies:
                index_movies.append(non_rated.index[non_rated['imdb_id'] == i].values[0])
            dfBatch = pd.concat([dfBatch, non_rated.iloc[index_movies]], axis=0)

        print('dfBatch after conc = ', len(dfBatch))

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
            next_choice = index_score_good[random.randint(0, len(index_score_good) - 1)]
            # print(next_choice)
            prediction = next_choice
        else:
            tmp_movie = non_rated.loc[
                non_rated['imdb_id'] == non_rated.loc[random.randint(0, non_rated.shape[0])]['imdb_id']]
            prediction = tmp_movie.imdb_id.values[0]
    return prediction


# from kaggle project on this database
# https://www.kaggle.com/rounakbanik/movie-recommender-systems



def get_recommendations(title, cosine_sim):
    global dfTitles

    print(dfTitles.shape)
    dfTitles = dfTitles.reset_index(drop=True)
    indices = pd.Series(dfTitles.index, index=dfTitles['Title'])

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]

    return dfTitles["imdb_id"].iloc[movie_indices]


if __name__ == '__main__':
    main()
