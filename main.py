from sportmonks.soccer import SoccerApiV2

TOKEN = 'H3KQigPBXehIQBrVn691GOgEPPiJUSnDvFEPh7lfIoha7ud5OxQRX0PAYBJB'

# type in de python console: "pip install sportmonks" en daarna "pip install tzlocal"

def main():
    soccer = SoccerApiV2(api_token=TOKEN)
    fixtures = soccer.fixtures_today(includes=('localTeam', 'visitorTeam'))

    for f in fixtures:
        print(f['localTeam']['name'], 'plays at home against', f['visitorTeam']['name'])



if __name__ == '__main__':
    main()