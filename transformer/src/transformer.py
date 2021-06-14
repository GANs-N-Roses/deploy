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

class PianoPerformanceLanguageModelProblem(score2perf.Score2PerfProblem):
  @property
  def add_eos_symbol(self):
    return True

class MusicTransformer:
  def __init__(self,model_name = 'transformer', hparams_set = 'transformer_tpu', ckpt_path = '/music/models/unconditional_model_16.ckpt'):
    
    problem = PianoPerformanceLanguageModelProblem()
    self.unconditional_encoders = problem.get_feature_encoders()

    # Set up HParams.
    hparams = trainer_lib.create_hparams(hparams_set = hparams_set)
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

    # These values will be changed by subsequent cells.
    self.targets = []
    self.decode_length = 0

    # Start the Estimator, loading from the specified checkpoint.
    input_fn = decoding.make_input_fn_from_generator(self.__input_generator())
    self.unconditional_samples = estimator.predict(
        input_fn, checkpoint_path = ckpt_path)

    # "Burn" one.
    _ = next(self.unconditional_samples)

    self.targets = []
    self.decode_length = 1024

  # Decode a list of IDs.
  def __decode(self, ids, encoder):
    ids = list(ids)
    if text_encoder.EOS_ID in ids:
      ids = ids[:ids.index(text_encoder.EOS_ID)]
    return encoder.decode(ids)

  # Create input generator (so we can adjust priming and
  # decode length on the fly).
  def __input_generator(self):
    while True:
      yield {
          'targets': np.array([self.targets], dtype=np.int32),
          'decode_length': np.array(self.decode_length, dtype=np.int32)
      }
  
  def predict(self, input, output = None, seconds = 5):
    primer_ns = note_seq.midi_file_to_note_sequence(input)

    # Handle sustain pedal in the primer.
    primer_ns = note_seq.apply_sustain_control_changes(primer_ns)

    # Trim to desired number of seconds.
    max_primer_seconds = seconds
    if primer_ns.total_time > max_primer_seconds:
      print('Primer is longer than %d seconds, truncating.' % max_primer_seconds)
      primer_ns = note_seq.extract_subsequence(
          primer_ns, 0, max_primer_seconds)

    # Remove drums from primer if present.
    if any(note.is_drum for note in primer_ns.notes):
      print('Primer contains drums; they will be removed.')
      notes = [note for note in primer_ns.notes if not note.is_drum]
      del primer_ns.notes[:]
      primer_ns.notes.extend(notes)

    # Set primer instrument and program.
    for note in primer_ns.notes:
      note.instrument = 1
      note.program = 0

    self.targets = self.unconditional_encoders['targets'].encode_note_sequence(
        primer_ns)

    # Remove the end token from the encoded primer.
    self.targets = self.targets[:-1]

    self.decode_length = max(0, 4096 - len(self.targets))
    if len(self.targets) >= 4096:
      print('Primer has more events than maximum sequence length; nothing will be generated.')

    # Generate sample events.
    sample_ids = next(self.unconditional_samples)['outputs']

    # Decode to NoteSequence.
    midi_filename = self.__decode(
        sample_ids,
        encoder=self.unconditional_encoders['targets'])
    ns = note_seq.midi_file_to_note_sequence(midi_filename)

    # Append continuation to primer.
    continuation_ns = note_seq.concatenate_sequences([primer_ns, ns])

    output_file = output

    if not output:
      output_file = '/music/output/' + input.split('/')[-1]

    note_seq.sequence_proto_to_midi_file(
        continuation_ns, output_file)