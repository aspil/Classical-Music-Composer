import os.path

from music21 import instrument, note, chord, stream
from src.utils.utils import convert_to_float


def create_midi(prediction_output, path_to_save):
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        pattern, dur = pattern.split('$')
        # Chord pattern
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # Rest pattern
        elif 'rest' in pattern:
            new_rest = note.Rest(pattern)
            new_rest.offset = offset
            new_rest.storedInstrument = instrument.Piano()
            output_notes.append(new_rest)
        # Note pattern
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        offset += convert_to_float(dur)

    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', path_to_save)
