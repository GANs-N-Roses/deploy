import os
import argparse
import requests
import yaml
import shortuuid

def continue_primer(config: dict, primer_file: str, seconds: int, acc: bool, verbose: bool = False):
    """
    Send a HTTP request to MusicTransformer to generate a new song based on primer_file
    """
    url = f'http://{config["transformer_host"]}:{config["transformer_port"]}/predict/{primer_file}'

    if isinstance(seconds, int) or isinstance(acc, bool):
        url += '?'

    if isinstance(seconds, int):
        url += f's={seconds}'
        if isinstance(acc, bool):
            url += '&'

    if isinstance(acc, bool):
        url += f'acc={1 if acc else 0}'

    if verbose:
        print(f'----- Getting song from MusicTransformer at {url}')

    r = requests.get(url)

    output_file = shortuuid.uuid() + '.mid'
    output_path = os.path.join(config['output_dir'], output_file)

    with open(output_path, 'wb') as f:
        f.write(r.content)

    return output_path

def music_pipe(config: dict, initial_midi: str, verbose: bool=False, seconds: int=5, acc: bool=True) -> str:
    final_midi = continue_primer(config, initial_midi, seconds, acc, verbose = verbose)
    if verbose:
        print('---- Final midi song path: {}'.format(final_midi))
    return final_midi

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='yaml config file')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--music_model_path', type=str, default='', help='path to the music model')
    parser.add_argument('--initial_midi', type=str, default='', help='name of the initial midi file')
    parser.add_argument('--final_midi_path', type=str, default='', help='path of the output midi file')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, 'r'))
    
    if args.music_model_path:
        config['input_music_model'] = args.music_model_path
    if args.initial_midi_path:
        config['initial_song'] = args.initial_midi
    if args.final_midi_path:
        config['output_song'] = args.final_midi_path

    music_pipe(config, config['initial_song'], verbose=args.verbose)