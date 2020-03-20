import json
import pandas as pd
import urllib3
import random
from flexx import flx, ui, app

sliderValue = 1

# from main import pass_user_score


REQUEST = 'http://www.omdbapi.com/?apikey=b5eee38&'
MOVIES_INDEX = []


# This is some black magic fuckery you don't want to touch #
class Website(flx.PyWidget):
    imdb_index = flx.AnyProp(0, settable=True)

    def init(self):
        self.main_container = flx.VSplit(style='font-family: "Helvetica Neue";')
        with self.main_container:
            with flx.VSplit(flex=2):
                self.image = flx.Widget(flex=4, style='background: black;')
                self.title_w = flx.Widget(flex=1)
                with self.title_w:
                    self.title = ui.Label(wrap=True)
                self.overview_w = flx.Widget(flex=1, style='height: 10vh;')
                with self.overview_w:
                    self.overview = ui.Label(wrap=True)
            with flx.HBox(flex=1):
                # self.h_button = flx.Button(text='Horrible', flex=1)
                self.m_button = flx.Button(text='Meh', flex=1)
                self.n_button = flx.Button(text='Neutral', flex=1)
                self.g_button = flx.Button(text='Yeah!', flex=1)
                # self.a_button = flx.Button(text='Amazing', flex=1)
            with flx.HBox(flex=0):
                self.slider = ui.Slider(text='Exploration/Exploitation', flex=3, min=0, max=1, step=0.01, value=sliderValue)
                # self.label = ui.Label(flex=1)
                # self.label = flx.Label(flex=1)

            # class JS:
            #     @react.connect('slider.value')
            #     def _change_label(self, value):
            #         self.label.text('x'.repeat(value))
        self.set_movie()

    # Here we get the new information about the movie and display it on the webpage #
    @flx.action
    def set_movie(self, *events):

        information, index = UserInterface.get_movie_info(self)
        if information is not None:
            self._mutate_imdb_index(index)
            style = 'background: url(https://{}); ' \
                    'background-repeat: no-repeat; ' \
                    'background-position: center;' \
                    'background-color: black;' \
                    'background-size: contain' \
                .format(information['Poster'][8:])
            self.image.apply_style(style)
            self.title.set_text(information['Title'])
            self.title_w.apply_style('font-size: calc(1em + 2vw); font-weight: \'bold\'; text-align: center;')
            self.overview_w.apply_style('font-size: calc(1em + 0.85vw); text-align: center;')
            self.overview.set_text(information['Plot'])
        else:
            print("No info passed")

    # These functions will process the user's reaction and display the next movie in line
    # @flx.reaction('h_button.pointer_down')
    # def click_horrible(self, *events):
    #     score_movie(-2, self.imdb_index)
    #     self.set_movie()

    @flx.reaction('m_button.pointer_down')
    def click_meh(self, *events):
        UserInterface.score_movie(self, -1, self.imdb_index)
        self.set_movie()

    @flx.reaction('n_button.pointer_down')
    def click_not_seen(self, *events):
        UserInterface.score_movie(self, 0, self.imdb_index)
        self.set_movie()

    @flx.reaction('g_button.pointer_down')
    def click_good(self, *events):
        UserInterface.score_movie(self, 1, self.imdb_index)
        self.set_movie()

    @flx.reaction('slider.user_value')
    def slide(self, *events):
        global sliderValue
        # self.label.set_text('{:.2f}'.format(self.slider1.value))
        sliderValue = self.slider.value

    # @flx.reaction('a_button.pointer_down')
    # def click_amazing(self, *events):
    #     score_movie(2, self.imdb_index)
    #     self.set_movie()


class UserInterface:
    def __init__(self):
        print("Created new UI")
        # app = flx.App(Website)
        # app.launch('browser')
        # flx.run()

    def get_movie_info(self):
        if len(MOVIES_INDEX) > 1:
            imdb_i = MOVIES_INDEX.pop(0)
            http = urllib3.PoolManager()
            try:
                r = http.request('GET', REQUEST + 'i=' + str(imdb_i))
                movie_info = json.loads(r.data.decode('utf8'))
                r.release_conn()
            except:
                print("Movie not found in API")
                self.score_movie(0, imdb_i)
            return movie_info, imdb_i
        else:
            print("Not enough movies in queue")

    def add_movie(self, index):
        MOVIES_INDEX.append(index)

    def run(self):
        app = flx.App(Website)
        app.launch('browser')
        flx.run()

    def get_movieList(self):
        return MOVIES_INDEX

    def score_movie(self, score, index):
        from main import pass_user_score
        pass_user_score(score, index)
        return 0  # pass_user_score(score, index)


if __name__ == "__main__":
    app = flx.App(Website)
    app.launch('browser')
    flx.run()
