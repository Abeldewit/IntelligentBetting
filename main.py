from sportmonks.soccer import SoccerApiV2

TOKEN = 'H3KQigPBXehIQBrVn691GOgEPPiJUSnDvFEPh7lfIoha7ud5OxQRX0PAYBJB'

# type in de python console: "pip install sportmonks" en daarna "pip install tzlocal"

def main():
    soccer = SoccerApiV2(api_token=TOKEN)
    

if __name__ == '__main__':
    main()