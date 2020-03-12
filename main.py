import pandas as pd
import numpy as np
from ast import literal_eval
from gui import add_movie
from gui import UserInterface
import random
import csv
import os

# constant which determines the amount of movies in a genre's top
topX = 40
df = pd.read_csv('data/movieData.csv')
df['user_score'] = 0
score_writer = csv.writer(open('data/user/scored.csv', 'w'))


def main():
	begin()
	UserInterface()


def begin():
	top_genre_list = os.listdir('data/topMovies/')
	# Pick a random genre
	random_genre = random.randint(0, len(top_genre_list))
	genre = top_genre_list[random_genre]
	tmp_df = pd.read_csv('data/topMovies/'+genre)
	for index in range(5):
		tmp_movie = tmp_df.iloc[index]
		add_movie(tmp_movie.imdb_id)

	# count = 0
	# while count < 10:
	# 	add_movie(df.iloc[random.randint(0, len(df))]['imdb_id'])
	# 	count += 1
	return 0


def choose_new():
	top_genre_list = os.listdir('data/topMovies/')
	# Pick a random genre
	random_genre = random.randint(0, len(top_genre_list)-1)
	genre = top_genre_list[random_genre]
	tmp_df = pd.read_csv('data/topMovies/' + genre)

	# Pick a random movie
	tmp_movie = df.loc[df['imdb_id'] == tmp_df.loc[random.randint(0, tmp_df.shape[0])]['imdb_id']]

	if int(tmp_movie['user_score']) == 0:
		add_movie(tmp_movie.imdb_id)
	else:
		choose_new()


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
	# We set the score in our dataframe
	df.loc[df['imdb_id'] == imdb, 'user_score'] = score

	# And here we write the scored movie to a csv file
	row = df.loc[df['imdb_id'] == imdb].iloc[0]
	score_writer.writerow([row['imdb_id'], row['user_score']])

	choose_new()
	return 0


if __name__ == '__main__':
	main()
