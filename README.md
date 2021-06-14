<p align="center">
<img height=150 src="https://i.imgur.com/VCPjCtU.png">
</p>

# About

This is an artificial intelligence project, with the objective of translating pieces of visual art into music, enabling any person to feel art regardless of their visual impairment.

For this purpose, we have developed a convolutional neural network that extracts the sentiment from a painting [in terms of valence-arousal](https://en.wikipedia.org/wiki/Emotion_classification#Circumplex_model). Then, a MIDI audio file is picked from a dataset labelled by emotion, which combines classical music and songs from videogames. A MusicTransformer produces an entirely new song inspired on the selected track.

This package features a Streamlit application that brings all these pieces together.

## Usage

Clone the repository and navigate into the folder:

```bash
git clone https://github.com/GANs-N-Roses/deploy.git art2music
cd art2music
```

This repository does not include a soundfount for filesize reasons. Download a soundfont and place it (a default.sf2 file) in the ```./visualization/soundfonts/``` directory. This soundfount will be used to render the output song.

Build and run the Docker containers:

```bash
docker-compose up
```

Visit the Streamlit web application (at http://localhost:8000), input an image and wait a few minutes to get a completely original, emotion-conditioned song generated

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
