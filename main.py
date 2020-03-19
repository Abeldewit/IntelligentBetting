import pandas as pd
import numpy as np
from ast import literal_eval
from gui import UserInterface
from sklearn.tree import DecisionTreeClassifier
import random
import csv
import os
from sklearn.model_selection import train_test_split

# constant which determines the amount of movies in a genre's top
topX = 40
df = pd.read_csv('data/movieData.csv')
df['user_score'] = -2
score_writer = csv.writer(open('data/user/scored.csv', 'a'))
UI = UserInterface()
scoredArr = [] #array where all the imdb ids and scores are handled.



def main():
	begin()
	UI.run()
	
	return 0


def begin():
	# get the top genres directly from the top 100
	tmp_df = pd.read_csv('data/topMovies/top100.csv')

	for index in range(5):
		movieIndex = random.randint(0,100)
		tmp_movie = tmp_df.iloc[movieIndex]
		UI.add_movie(tmp_movie.imdb_id)

	return 0


def choose_new():
	top_genre_list = os.listdir('data/topMovies/')
	# Pick a random genre
	random_genre = random.randint(0, len(top_genre_list)-1)
	genre = top_genre_list[random_genre]
	tmp_df = pd.read_csv('data/topMovies/' + genre)

	# Pick a random movie
	tmp_movie = df.loc[df['imdb_id'] == tmp_df.loc[random.randint(0, tmp_df.shape[0])]['imdb_id']]
	# print(tmp_movie['user_score'])

	if int(tmp_movie['user_score']) == -2:
		UI.add_movie(tmp_movie.imdb_id.values[0])

	else:
		choose_new()


	print(UI.get_movieList())

# topX = the amount of movies we want in the genre specific database
def createTable(genre, indexOfBest, df, topX):
	#TODO Goeie documentatie wat deze functie doet
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


	if len(scoredArr) < 20:
		choose_new()

	else:
		print("classification here")
		predictor()



	return 0

#TODO mconstruct a predictor for the new suggestions based on a decision tree
def predictor():
	prediction = []

	trainingData = df.loc[(df['user_score'] != -2)]
	trainingData = trainingData.loc[(trainingData['user_score'] != 0)]
	dfPred = df.loc[(df['user_score'] == -2)]

	# create dummies
	dfX = trainingData[trainingData.columns.difference(['user_score'])]
	nonNumer = dfX.select_dtypes(exclude=np.number).columns
	numericCols = list(dfX.select_dtypes(include=np.number).columns)  # find the columns that are numeric
	numericColsPred = list(dfPred.select_dtypes(include=np.number).columns)

	dummy_vars_dfX = pd.get_dummies(dfX[list(dfX.select_dtypes(exclude=np.number).columns)])
	dfX_Dummy = dfX[numericCols].join(dummy_vars_dfX)

	dummy_vars_dfPred = pd.get_dummies(dfPred[list(dfPred.select_dtypes(exclude=np.number).columns)])
	dfPred_Dummy = dfPred[numericColsPred].join(dummy_vars_dfPred)

	# maybe drop the weighted rating and the voterating while this may have a lot of influence?
	x = dfX_Dummy
	y = trainingData['user_score']


	X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

	DTC = DecisionTreeClassifier()
	DTC.fit(X_train, y_train)
	score = DTC.score(X_test, y_test)

	results = DTC.predict(dfPred_Dummy[:50])
	print(results)



	print("score = ", score)



	# make the classifier (random Forrest?, on what data do we predict.

	## add the movie predicted based on the imdb_id
	# UI.add_movie()

	return prediction


if __name__ == '__main__':
	main()
