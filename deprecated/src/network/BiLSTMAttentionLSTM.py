import os

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Bidirectional, LSTM, Dropout, Flatten, Dense, Activation
from keras.models import Sequential, load_model
from keras_self_attention import SeqSelfAttention

from deprecated.src.utils.music21_utils import int_to_note_dict


class BiLSTMAttentionLSTM:
    def __init__(self, composers, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.model_save_path = os.path.abspath(
            'models/BiLSTM_Att_LSTM/' + '_'.join(composers) + "model.hdf5")
        self.checkpoint_model_path = os.path.abspath(
            "models/BiLSTM_Att_LSTM/checkpoints/" + '_'.join(composers) + "_weights-{epoch:03d}-{loss:.4f}.hdf5")

        self.model = Sequential()
        self.callbacks_list = [
            ModelCheckpoint(
                self.checkpoint_model_path,
                save_freq='epoch',
                period=10,
                monitor='loss',
                verbose=1,
                save_best_only=False,
                mode='min'
            )
        ]

    def build_layers(self):
        self.model.add(Bidirectional(
            LSTM(512, return_sequences=True),
            input_shape=(self.input_shape[0], self.input_shape[1]))
        )
        self.model.add(SeqSelfAttention(attention_activation='sigmoid',
                                        attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL))
        self.model.add(Dropout(0.3))

        self.model.add(LSTM(512, return_sequences=True))
        self.model.add(Dropout(0.3))

        self.model.add(Flatten())
        self.model.add(Dense(self.output_shape))
        self.model.add(Activation('softmax'))

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='rmsprop', metrics=['accuracy'])

    def summary(self):
        return self.model.summary()

    def save_model(self):
        self.model.save(os.path.abspath(self.model_save_path))

    def load_model(self, path):
        self.model = load_model(
            path, custom_objects=SeqSelfAttention.get_custom_objects())

    def train(self, x, y, epochs=200, batch_size=64):
        return self.model.fit(x, y, epochs=epochs, batch_size=batch_size, callbacks=self.callbacks_list)

    def generate_sequence(self, n_notes, x_train, pitchnames):
        n_vocab = len(pitchnames)

        # Create a dictionary to map pitches to integers
        int_to_note = int_to_note_dict(pitchnames)
        start = np.random.randint(0, len(x_train) - 1)

        pattern = x_train[start]
        prediction_output = []

        # Generate a piece of <n_notes> notes
        for _ in range(n_notes):
            prediction_input = np.reshape(pattern, (1, len(pattern), 1))
            prediction_input = prediction_input / float(n_vocab)
            prediction = self.model.predict(prediction_input, verbose=0)

            index = np.argmax(prediction)
            result = int_to_note[index]
            prediction_output.append(result)
            pattern = np.append(pattern, index)
            pattern = pattern[1:len(pattern)]

        return prediction_output
