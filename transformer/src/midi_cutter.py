from mido import MidiFile
from mido import tick2second
from mido import merge_tracks
import os
from mido.midifiles.tracks import MidiTrack

def load_MidiFile(input_file):
    out = MidiFile(filename=input_file)
    return out

def cut_starting_silence(mid):
    """Filters out starting silence on any music composition
    Returns a MidiFile"""
    out = MidiFile()
    new_track = MidiTrack()
    melody_started = False

    track = merge_tracks(mid.tracks)

    for msg in track:
        # print(msg)
        if not melody_started and msg.type == 'control_change':
            pass
        elif not melody_started and msg.type == 'note_on':
            melody_started = True
            print('Melody started')

        if melody_started or msg.type != 'control_change':
            new_track.append(msg)
            print(f'>>> Added {msg}')
        else:
            print(f'>>> NOT ADDED {msg}')
            # pass
    
    out.tracks.append(new_track)
    out.ticks_per_beat = mid.ticks_per_beat
    out.save('test_cut_silence.mid')
    return out

def check_tempo(msg, current_tempo=None):
    if msg.type == 'set_tempo':
        return msg.tempo
    else:
        return current_tempo

def cut_midi(mid, start=5, end=999, output_file='output.mid', cut_start=False, _verbose=False):
    out_mid = MidiFile()
    new_track = MidiTrack()

    tempo = 0
    count_on = 0

    if cut_start:
        mid = cut_starting_silence(mid)

    track = merge_tracks(mid.tracks)
    elapsed_time = 0  
    for msg in track:
        if _verbose:
            print(msg, elapsed_time)
        
        # Messages can be notes, configurations, etc
        tempo = check_tempo(msg, current_tempo=tempo)
        
        # print(f'>>> Time.. {msg.time} {mid.ticks_per_beat} {tempo}, {tick2second(msg.time, mid.ticks_per_beat, tempo)}')
        elapsed_time  += 2 * tick2second(msg.time, mid.ticks_per_beat, tempo)  # 2x because wtf
        
        if msg.type == 'note_on':
            count_on += 1
        elif msg.type == 'note_off':
            count_on -= 1
        # print(f'>>> Tone on count {count_on} at {elapsed_time}')

        # add only the selected messages
        if (start < elapsed_time < end) or elapsed_time == 0: 
            new_track.append(msg)
            if _verbose:
                print(f'>>> Adding {msg} {elapsed_time}')

            if msg.type == 'end_of_track':
                break

    out_mid.tracks.append(new_track)
    out_mid.ticks_per_beat = mid.ticks_per_beat

    out_mid.save(output_file)    

if __name__ == '__main__':
    # Example of use
    mid = load_MidiFile('input.mid')
    cut_midi(mid, start=5, output_file='output.mid')
