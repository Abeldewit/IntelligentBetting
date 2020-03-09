from flexx import flx, ui
import urllib3, json
import random
import pandas as pd

REQUEST = 'http://www.omdbapi.com/?apikey=2b5ae7ec&'
DATA = pd.read_csv('dataClean/movieDataClean.csv')


def get_movie_info():
    index = random.randint(0, DATA.shape[0])
    movie = DATA.iloc[index]['imdb_id']
    movie = movie.replace(' ', '+')
    http = urllib3.PoolManager()
    r = http.request('GET', REQUEST + 'i=' + movie)
    movie_info = json.loads(r.data.decode('utf8'))
    r.release_conn()
    return movie_info


class Example(flx.PyWidget):

    def init(self):
        with flx.VSplit():
            with flx.VSplit(flex=2):
                self.image = flx.Widget(flex=2, style='background: black;')
                self.overview = ui.Label(flex=1, wrap=True)
            with flx.HBox(flex=1):
                self.h_button = flx.Button(text='Horrible', flex=1)
                self.m_button = flx.Button(text='Meh', flex=1)
                self.g_button = flx.Button(text='Good', flex=1)
                self.a_button = flx.Button(text='Amazing', flex=1)

    @flx.reaction('h_button.pointer_click')
    def set_movie(self, *events):
        information = get_movie_info()
        style = 'background: url(https://{}); ' \
                'background-repeat: no-repeat; ' \
                'background-position: center;' \
                'background-color: black;' \
                'background-size: contain' \
                .format(information['Poster'][8:])
        self.image.apply_style(style)
        self.overview.set_text(information['Plot'])


if __name__ == "__main__":
    app = flx.App(Example)
    app.launch('browser')
    flx.run()

