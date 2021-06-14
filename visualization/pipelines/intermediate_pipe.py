import os
import math
import argparse
import pickle
from shutil import copyfile
from typing import Tuple

import pandas as pd
import numpy as np
import yaml

def load_model(model_filename, labels_filename):
    with open(model_filename, 'rb') as f:
        model = pickle.load(f)

    with open(labels_filename, 'rb') as f:
        d_labels = pickle.load(f)
    
    return model, d_labels

def select_piece(model, valence_arousal):
    piece_name = model.predict(valence_arousal)[0]
    return piece_name

def get_initial_midi_file(piece_name, config: dict, knn_df=None, file=None, verbose=False):
    if knn_df is not None:
        if file is None:
            file = os.path.join(config['image_dir_path'], config['pieces_file'])
        
        knn_df = pd.read_csv(file)
        knn_df = knn_df.set_index('Unnamed: 0')
    
    if verbose:
        print(">>> Selected piece")
        print(knn_df.loc[piece_name])

    return knn_df.loc[piece_name]

def normalize_vector(unique_vector: Tuple[float, float]) -> Tuple[float, float]:
    norm_x = unique_vector[0] / math.sqrt(math.pow(unique_vector[0], 2) + math.pow(unique_vector[1], 2))
    norm_y = unique_vector[1] / math.sqrt(math.pow(unique_vector[0], 2) + math.pow(unique_vector[1], 2))
    return (norm_x, norm_y)

def get_unique_vector(vector_per_cuadrant: dict) -> Tuple[float, float]:
    x = 0
    y = 0
    for vector in vector_per_cuadrant.values():
        x += vector[0]
        y += vector[1]

    return (x, y)

def get_valence_arousal(emotions: list, predictions: dict) -> dict:
    valence_arousal = {}
    counter = 1
    for i in range(1, len(emotions)):
        emt = emotions[i]
        x_i = predictions[emt] * math.cos((counter) * math.pi / 4)
        y_i = predictions[emt] * math.sin((counter) * math.pi / 4)
        va_i = (x_i, y_i)
        valence_arousal[emt] = va_i
        counter += 2

    return valence_arousal

def to_valence_arousal(config: dict, predictions: dict, verbose: bool=False) -> Tuple[float, float]:
    valence_arousal = get_valence_arousal(config['emotions'], predictions)
    if verbose:
        print('---- Valence-arousal')
        print(valence_arousal)
    
    vector_per_cuadrant = get_unique_vector(valence_arousal)
    if verbose:
        print('---- Vector per cuadrant')
        print(vector_per_cuadrant)

    norm_vector = normalize_vector(vector_per_cuadrant)
    if verbose:
        print('---- Vector normalized')
        print(norm_vector)
    
    return np.array(norm_vector).reshape(1,-1)

def create_midi(config: dict, va_vector: Tuple[float, float], model=None, knn_df=None, verbose: bool=False):
    """
    Given a valence-arousal vector, convert it into a midi file and save it
    """
    if model is None:
        with open(os.path.join(config['input_dir'], config['knn_model']),
                  'rb') as f:
            model = pickle.load(f)
    piece_name = select_piece(model, va_vector)
    initial_midi = get_initial_midi_file(piece_name, config, knn_df=knn_df, verbose=True)
    initial_midi_file = initial_midi['midi']
    if verbose:
        print('---- Selected initial midi: {}'.format(initial_midi_file))
    return initial_midi_file, (initial_midi['valence'], initial_midi['arousal'])

def intermediate_pipe(config: dict, va_vector: np.array, model=None, knn_df=None, verbose: bool=False):
    midi_file, (valence, arousal) = create_midi(config, va_vector, model=model, knn_df=knn_df, verbose=verbose)
    return midi_file, (valence, arousal)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='yaml config file')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--midi_path', type=str, default='', help='path of the output midi file')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, 'r'))
    
    if args.midi_path:
        config['initial_song'] = args.midi_path

    intermediate_pipe(config, config['default_preds'], verbose=args.verbose)