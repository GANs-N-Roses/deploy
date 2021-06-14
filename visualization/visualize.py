import os

import yaml
import pandas as pd
import streamlit as st
from midi2audio import FluidSynth
import matplotlib.pyplot as plt
import numpy as np
import pickle
import shortuuid


from pipelines import image_pipe, intermediate_pipe, music_pipe

PIPE_DIR = 'pipelines'
PICTURE_TYPES = ['jpg', 'jpeg', 'bmp', 'png']
OUTPUT_WAV = 'output.wav'
DEFAULT_SECONDS = 5
DEFAULT_ACCOMPANIAMENT = True
DEBUG = True

##################
# CACHED OBJECTS #
##################
@st.cache
def get_configuration(verbose: bool=False) -> dict:
    config = yaml.safe_load(open(os.path.join(PIPE_DIR, 'config.yaml'), 'r'))

    if verbose:
        print('CONFIG:')
        print(config)

    real_input_dir = os.path.join(PIPE_DIR, config['image_dir_path'])
    config['image_dir_path'] = real_input_dir
    config['input_dir'] = os.path.join(PIPE_DIR, config['input_dir'])
    config['output_dir'] = os.path.join(PIPE_DIR, config['output_dir'])
    config['initial_midi_dir'] = os.path.join(PIPE_DIR, config['initial_midi_dir'])

    return config

@st.cache
def get_image_model(config: dict):
    path_model = os.path.join(config['input_dir'], config['input_image_model'])
    model = image_pipe.load_h5model(path_model)
    return model

@st.cache(allow_output_mutation=True)
def get_knn_model(config: dict):
    with open(os.path.join(config['input_dir'], config['knn_model']), 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache
def get_knn_df(config: dict) -> pd.DataFrame:
    df_file = os.path.join(config['image_dir_path'], config['pieces_file'])
    df = pd.read_csv(df_file)
    df = df.set_index('Unnamed: 0')
    return df

#########################
# REST OF THE FUNCTIONS #
#########################
def get_image_path(config: dict, uploaded_image):
    bytes_data = uploaded_image.getvalue()
    filename = 'image.{}'.format(uploaded_image.type.split('/')[1])
    image_file = os.path.join(config['image_dir_path'], filename)
    try:
        f = open(image_file, 'wb')
        f.write(bytes_data)
        f.close()
    except:
        return None
    
    return filename

def get_circle_pic(painting_coordinates, midi_coordinates):
    plt.figure(figsize=(12, 12))
    ax = plt.gca()

    # Draw circle
    ax.set_aspect(1)
    circle = plt.Circle((0, 0), 1, fill=False)
    ax.add_artist(circle)

    # Fill 4 parts of circle
    x = np.arange(0.0, 1.01, 0.01)
    circle = np.sin(np.arccos(x))
    ax.fill_between(x, 0, circle, color='lightgreen')
    circle = -np.sin(np.arccos(x))
    ax.fill_between(x, 0, circle, color='lightskyblue')
    circle = -np.sin(np.arccos(x))
    ax.fill_between(-x, 0, circle, color='lightsteelblue')
    circle = np.sin(np.arccos(x))
    ax.fill_between(-x, 0, circle, color='lightcoral')

    # Add text
    ax.text(1.1, 0.05, 'possitive')
    ax.text(-1.3, -0.1, 'negative')
    ax.text(0.05, 1.05, 'active')
    ax.text(-0.2, -1.05, 'passive')

    inner_font = {'weight': 'heavy', 'size': 'xx-large'}
    ax.text(0.45, 0.45, 'Joy', color='w', fontdict=inner_font)
    ax.text(-0.55, 0.45, 'Anger', color='w', fontdict=inner_font)
    ax.text(-0.59, -0.45, 'Sadness', color='w', fontdict=inner_font)
    ax.text(0.38, -0.45, 'Calm', color='w', fontdict=inner_font)

    # Add arrows
    origin = (0, 0)
    q1 = ax.quiver(origin, origin, *painting_coordinates, scale=1,
                   scale_units='xy', angles='xy', color='tab:blue')
    q2 = ax.quiver(origin, origin, *midi_coordinates, scale=1,
                   scale_units='xy', angles='xy', color='tab:green')
    ax.quiverkey(q1, 1.2, 1.2, 0.5, 'painting', coordinates='data')
    ax.quiverkey(q2, 1.2, 1, 0.5, 'midi', coordinates='data')

    # Axis settings
    ax.set_xlim((-1.5, 1.5))
    ax.set_ylim((-1.5, 1.5))
    # Move left y-axis and bottim x-axis to centre, passing through (0,0)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['left'].set_linestyle((0, (8, 5)))
    ax.spines['bottom'].set_linestyle((0, (8, 5)))
    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    # Turn off tick labels
    ax.set_yticks([])
    ax.set_xticks([])

    plt.savefig(os.path.join(config['output_dir'], 'circle.png'), transparent=True)

#############
# MAIN FLOW #
#############

st.subheader('Options')
secs = st.number_input('Seconds to take from the initial sequence', min_value = 1, max_value = 20, value = DEFAULT_SECONDS)
acc = st.checkbox('Generate accompaniament for the melody', value = DEFAULT_ACCOMPANIAMENT)

st.subheader('Paint to sound!')
uploaded_image = st.file_uploader('Upload a picture', type=PICTURE_TYPES)

if uploaded_image:
    config = get_configuration(verbose=DEBUG)
    image_path = get_image_path(config, uploaded_image)
    if not image_path:
        st.error("We couldn't save the picture, please try again")
    else:
        # show the image
        st.image(os.path.join(config['image_dir_path'], image_path))

        # get the emotion attached to it
        emo_pred = image_pipe.image_pipe(config, image_path, model=get_image_model(config), verbose=DEBUG)

        # convert the emotion to an initial midi song
        knn_model = get_knn_model(config)
        knn_df = get_knn_df(config)
        initial_midi, midi_coordinates = intermediate_pipe.intermediate_pipe(config, emo_pred, model=knn_model, knn_df=knn_df, verbose=DEBUG)

        print(emo_pred)

        get_circle_pic(emo_pred[0].tolist(), midi_coordinates)
        st.image(os.path.join(config['output_dir'], 'circle.png'))

        # create the final midi song
        final_song_path = music_pipe.music_pipe(config, initial_midi, verbose=DEBUG, seconds = secs, acc = acc)

        OUTPUT_WAV = shortuuid.uuid() + '.wav'

        # save the midi song as a wav song
        final_wav_path = os.path.join(config['output_dir'], OUTPUT_WAV)
        FluidSynth().midi_to_audio(final_song_path, final_wav_path)

        # read the audio file to play it in streamlit
        audio_file = open(final_wav_path, 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes)
