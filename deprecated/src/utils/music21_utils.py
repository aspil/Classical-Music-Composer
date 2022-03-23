from music21 import note, chord


def get_duration(element):
    return element.duration.quarterLength


def is_note(element):
    return isinstance(element, note.Note)


def get_pitch(element):
    return element.pitch


def is_chord(element):
    return isinstance(element, chord.Chord)


def get_chord(element):
    """
    Returns a chord.Chord element as its custom-made numerican representation.
    ex. 
    """
    return '.'.join(str(n) for n in element.normalOrder)


def is_rest(element):
    return isinstance(element, note.Rest)


def get_unique_notes(notes):
    return sorted(set(item for item in notes))


def note_to_int_dict(pitchnames: list) -> dict:
    return {note: i for i, note in enumerate(pitchnames)}


def int_to_note_dict(pitchnames: list) -> dict:
    return {i: note for i, note in enumerate(pitchnames)}
