import pandas as pd
import numpy as np
from ast import literal_eval
from gui import add_movie
from gui import UserInterface
import random

# constant which determines the amount of movies in a genre's top
topX = 40


def main():
	df = pd.read_csv('data/movieData.csv')
	action_df = pd.read_csv('data/topMovies/Action.csv')


	# create all the topX tables for every genre
	# topXTables(topX, df)
	# createTop100(df)


	# for index, row in action_df.iterrows():
	# 	add_movie(row['imdb_id'])

	begin()
	user_interface = UserInterface()
	# exit()

	return 0

def begin():
	count = 0
	df = pd.read_csv('data/topMovies/top100.csv')

	while count < 10:
		add_movie(df.iloc[random.randint(0, len(df))]['imdb_id'])
		count += 1

	return 0

# This is where we get the title of the movie and the users score
def pass_user_score(imdb, score):
	print("Imdb id {} got scored {}".format(imdb, score))

# topX = the amount of movies we want in the genre specific database
def createTable(genre, indexOfBest, df, topX):
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
	dfTopMovies.to_csv(dfName, sep=',', encoding='utf-8', index = False)

	return 0

def topXTables(topX, df):
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

if __name__ == '__main__':
	main()
