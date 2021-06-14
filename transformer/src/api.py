import midi_cutter
from transformer import MusicTransformer
from conditional_transformer import MelodyConditionedTransformer

from flask import Flask, request, send_file
import gevent.pywsgi
import shortuuid

CONFIG = {
    'default_seconds': 5,
    'default_acc': True,
    'in_path': '/music/initial_midis/',
    'out_path': '/music/final_midis/'
}

app = Flask(__name__)
transformer = MusicTransformer()
melody_transformer = MelodyConditionedTransformer()

@app.route('/predict/<filename>', methods = ['GET'])
def predict(filename):
    """
    Handles requests for MusicTransformer to continue a MIDI file

    Parameters:
        filename (str): Filename of the MIDI to continue
        s (int): Optional, initial seconds to take from the primer. Defaults to 5.
        acc (int): Optional. Should accompaniament be generated for the melody? Defaults to 1 (true).
    """

    s = request.args.get('s')

    if s is not None:
        try:
            s = int(s)
        except Exception as e:
            raise e
            return 's must be an integer', 400
    else:
        s = CONFIG['default_seconds']

    acc = request.args.get('acc')

    if acc is not None:
        try:
            acc = bool(acc)
        except Exception:
            return 's must be 0 or 1', 400
    else:
        acc = CONFIG['default_acc']
    
    uuid = shortuuid.uuid()
    output_file = uuid + '.mid'
    output_path = CONFIG['out_path'] + output_file

    print(f'Invoking MusicTransformer with primer {filename} ({s}s), output: {output_file}')

    # Continue the primer
    try:
        transformer.predict(
            input = CONFIG['in_path'] + filename,
            output = output_path,
            seconds = s,
        )
    except Exception as e:
        return e, 500

    # Cut the primer out of the result
    try:
        mid = midi_cutter.load_MidiFile(output_path)
        midi_cutter.cut_midi(mid, start = s, output_file = output_path)
    except Exception as e:
        return e, 500

    if acc:
        try:
            melody_transformer.predict(
                    input = output_path,
                    output = output_path
            )
        except Exception as e:
            print('WARNING: Melody transformer failed, skipping accompaniament')

    # Return the generated midi file
    return send_file(
        output_path,
        mimetype = 'audio/midi'
    )

app_server = gevent.pywsgi.WSGIServer(('0.0.0.0', 8001), app)
print('Serving MusicTransformer API on port 8001')
app_server.serve_forever()