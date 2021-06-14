import os
import argparse

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as image_utils
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import yaml


def make_predictions(model, image_path_or_image, img_size, channels):
    image = image_utils.load_img(image_path_or_image, target_size=(
    img_size, img_size)) if isinstance(image_path_or_image,
                                       str) else image_path_or_image
    image = image_utils.img_to_array(image)
    image = image.reshape(1, img_size, img_size, channels)
    image = preprocess_input(image)
    return model.predict(image)[0]


def load_h5model(path_model: str):
    return load_model(path_model)


def image_pipe(config: dict, image_path: str, model=None,
               verbose=False) -> list:
    path_model = os.path.join(config['input_dir'], config['input_image_model'])
    if verbose:
        print('Path to model: {}'.format(path_model))
    model = load_h5model(path_model) if not model else model
    path_image = os.path.join(config['image_dir_path'], image_path)
    preds = make_predictions(model, path_image, config['img_size'],
                             config['channels'])
    if verbose:
        print(f'Valence: {preds[0]}, Arousal: {preds[1]}')
    return preds.reshape(1, -1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='yaml config file')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--image_model', type=str, default='',
                        help='path of model for image classification (if it is not the one in the config file)')
    parser.add_argument('--image_path', type=str, default='',
                        help='path of the image to classify')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, 'r'))

    if args.image_model:
        config['input_image_model'] = args.image_model
    if args.image_path:
        config['image_path'] = args.image_path

    image_pipe(config, verbose=args.verbose)
