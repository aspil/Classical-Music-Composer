import os
import numpy as np
import pickle

from music21 import converter

from keras.utils import np_utils

from src.utils.music21_utils import *


class MidiPreprocessor:

    def __init__(self, composers):
        self.midi_files = []
        self.composers = composers
        self.notes_path = os.path.abspath(
            'dataset/notes/notes_' + '_'.join(self.composers))

    def parse_midi_files(self, midi_files):
        notes = []

        for midi in midi_files:
            print("Fetched", midi)
            notes += get_notes_from_file(midi)

        with open(self.notes_path, 'wb') as filepath:
            pickle.dump(notes, filepath)

        return notes

    def notes_to_sequences(self, notes_list, sequence_length=50):
        # Get all pitch names
        pitchnames = get_unique_notes(notes_list)

        n_vocab = len(pitchnames)

        # Create a dictionary to map pitches to integers
        note_to_int = note_to_int_dict(pitchnames)

        num_training = len(notes_list) - sequence_length

        network_input = []
        network_output = []

        for i in range(num_training):
            # The i'th sequence
            sequence_in = notes_list[i:i + sequence_length]
            # The note right after the i'th sequence
            sequence_out = notes_list[i + sequence_length]
            network_input.append([note_to_int[char] for char in sequence_in])
            network_output.append(note_to_int[sequence_out])

        n_patterns = len(network_input)

        # Reshape the input into a format compatible with LSTM layers
        network_input = np.reshape(
            network_input, (n_patterns, sequence_length, 1))

        # Normalize input
        network_input = network_input / float(n_vocab)

        # Perform One-Hot Encoding
        network_output = np_utils.to_categorical(network_output)

        return network_input, network_output

    def get_cached_notes(self):
        notes = []
        try:
            with open(self.notes_path, 'rb') as fp:
                notes = pickle.load(fp)
                return notes
        except IOError:
            return []


def get_notes_from_file(midi_file):
    """
    Converts a midi file to a corresponding list of pitches.
    """
    midi = converter.parse(midi_file)
    cc = midi.flat
    notes_to_parse = cc.notes
    return parse_notes(notes_to_parse)


def parse_notes(notes_to_parse):
    notes = []
    for element in notes_to_parse:
        duration = get_duration(element)
        if is_note(element):
            name = get_pitch(element)

        elif is_chord(element):
            name = get_chord(element)

        elif is_rest(element):
            name = element.name

        if is_note(element) or is_chord(element) or is_rest(element):
            notes.append(str(name) + '$' + str(duration))

    return notes
