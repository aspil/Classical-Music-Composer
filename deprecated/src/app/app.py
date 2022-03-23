import glob
import os

from src.midi_preprocessing.midipreprocessor import MidiPreprocessor
from src.midi_preprocessing.seq2midi import create_midi
from src.network.BiLSTMAttentionLSTM import BiLSTMAttentionLSTM
from deprecated.src.utils.music21_utils import get_unique_notes


class App:
    def __init__(self, composers_list, dataset_path=None):
        self.composers = composers_list
        self.dataset_path = dataset_path

    def run(self):
        print("Starting")

        paths_to_composer_midis = [os.path.join(os.path.join(
            self.dataset_path, composer), '*.mid') for composer in self.composers]
        print(paths_to_composer_midis)

        midi_files = [
            file for path in paths_to_composer_midis for file in glob.glob(path)]

        midi_preprocessor = MidiPreprocessor(self.composers)

        notes_list = midi_preprocessor.get_cached_notes()
        if len(notes_list) == 0:
            print("Parsing midi files for composers ", self.composers)
            notes_list = midi_preprocessor.parse_midi_files(midi_files)

        x_train, y_train = midi_preprocessor.notes_to_sequences(notes_list)

        network = BiLSTMAttentionLSTM(
            composers=self.composers,
            input_shape=(x_train.shape[1], x_train.shape[2]),
            output_shape=y_train.shape[1]
        )

        network.build_layers()
        network.train(x_train, y_train, 10, 128)

        prediction = network.generate_sequence(
            250, x_train, get_unique_notes(notes_list))
        path = os.path.join(os.path.abspath('generated'),
                            '_'.join(self.composers))
        create_midi(path_to_save=path, prediction_output=prediction)

        print("Done")
