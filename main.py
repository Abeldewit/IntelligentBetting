import pandas as pd
import numpy as np
from ast import literal_eval


topX = 40

def main():
	df = pd.read_csv('dataClean/movieDataClean.csv')
	topXTables(topX, df)



	return 0

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



	return topMoviesIndex

def topXTables(topX, df):
	genres = dict()


	for index, subset in df.iterrows():
		genresArr = literal_eval(subset.genres)
		for genre in genresArr:
			if genre in genres:
				genres[genre] += 1
			else:
				genres[genre] = 1

	indexOfBest = df.vote_average.sort_values(ascending=False).index

	for key in genres:
		genreTop = createTable(key, indexOfBest, df, topX)
		print(genreTop)



	return 0

if __name__ == '__main__':
    main()
