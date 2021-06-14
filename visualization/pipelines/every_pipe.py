import image_pipe
import music_pipe
import intermediate_pipe

def every_pipe(config: dict, verbose: bool=False) -> str:
    emo_pred = image_pipe.image_pipe(config, verbose=verbose)
    initial_midi = intermediate_pipe.intermediate_pipe(config, emo_pred, verbose=verbose)
    final_song_path = music_pipe.music_pipe(config, initial_midi, verbose=verbose)
    return final_song_path