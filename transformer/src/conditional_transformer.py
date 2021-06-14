import numpy as np
import tensorflow.compat.v1 as tf

from tensor2tensor import models
from tensor2tensor import problems
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import decoding
from tensor2tensor.utils import trainer_lib

from magenta.models.score2perf import score2perf
import note_seq

tf.disable_v2_behavior()

class MelodyToPianoPerformanceProblem(score2perf.AbsoluteMelody2PerfProblem):
  @property
  def add_eos_symbol(self):
    return True

class MelodyConditionedTransformer:
  def __init__(self,model_name = 'transformer', hparams_set = 'transformer_tpu', ckpt_path = '/music/models/melody_conditioned_model_16.ckpt'):
    problem = MelodyToPianoPerformanceProblem()
    self.melody_conditioned_encoders = problem.get_feature_encoders()

    # Set up HParams.
    hparams = trainer_lib.create_hparams(hparams_set=hparams_set)
    trainer_lib.add_problem_hparams(hparams, problem)
    hparams.num_hidden_layers = 16
    hparams.sampling_method = 'random'

    # Set up decoding HParams.
    decode_hparams = decoding.decode_hparams()
    decode_hparams.alpha = 0.0
    decode_hparams.beam_size = 1

    # Create Estimator.
    run_config = trainer_lib.create_run_config(hparams)
    estimator = trainer_lib.create_estimator(
        model_name, hparams, run_config,
        decode_hparams=decode_hparams)

    self.inputs = []
    self.decode_length = 0

    # Create input generator.
    def input_generator():
      global inputs
      while True:
        yield {
            'inputs': np.array([[self.inputs]], dtype=np.int32),
            'targets': np.zeros([1, 0], dtype=np.int32),
            'decode_length': np.array(self.decode_length, dtype=np.int32)
        }

    # Start the Estimator, loading from the specified checkpoint.
    input_fn = decoding.make_input_fn_from_generator(input_generator())
    self.melody_conditioned_samples = estimator.predict(
        input_fn, checkpoint_path=ckpt_path)

    # "Burn" one.
    _ = next(self.melody_conditioned_samples)
  
  # Decode a list of IDs.
  def __decode(self, ids, encoder):
    ids = list(ids)
    if text_encoder.EOS_ID in ids:
      ids = ids[:ids.index(text_encoder.EOS_ID)]
    return encoder.decode(ids)
  
  def predict(self, input, output = None):
    melody_ns = note_seq.midi_file_to_note_sequence(input)
    melody_instrument = note_seq.infer_melody_for_sequence(melody_ns)
    notes = [note for note in melody_ns.notes
            if note.instrument == melody_instrument]
    del melody_ns.notes[:]
    melody_ns.notes.extend(
        sorted(notes, key=lambda note: note.start_time))
    for i in range(len(melody_ns.notes) - 1):
      melody_ns.notes[i].end_time = melody_ns.notes[i + 1].start_time
    self.inputs = self.melody_conditioned_encoders['inputs'].encode_note_sequence(
        melody_ns)

    self.decode_length = 4096
    sample_ids = next(self.melody_conditioned_samples)['outputs']

    # Decode to NoteSequence.
    midi_filename = self.__decode(
        sample_ids,
        encoder=self.melody_conditioned_encoders['targets'])
    accompaniment_ns = note_seq.midi_file_to_note_sequence(midi_filename)

    output_file = output

    if not output:
      output_file = '/music/output/' + input.split('/')[-1]

    note_seq.sequence_proto_to_midi_file(
        accompaniment_ns, output_file)