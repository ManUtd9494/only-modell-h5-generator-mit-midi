# -*- coding: utf-8 -*-

import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox, QFormLayout, QLineEdit, QMainWindow
import pretty_midi


class MidiLstmTrainer(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('MIDI LSTM Trainer')
        self.setGeometry(100, 100, 400, 200)

        layout = QVBoxLayout()

        form_layout = QFormLayout()
        self.sequence_length_edit = QLineEdit()
        self.sequence_length_edit.setPlaceholderText("100")
        form_layout.addRow(QLabel("Sequence Length:"), self.sequence_length_edit)

        self.epochs_edit = QLineEdit()
        self.epochs_edit.setPlaceholderText("100")
        form_layout.addRow(QLabel("Epochs:"), self.epochs_edit)

        self.batch_size_edit = QLineEdit()
        self.batch_size_edit.setPlaceholderText("64")
        form_layout.addRow(QLabel("Batch Size:"), self.batch_size_edit)

        layout.addLayout(form_layout)

        self.load_data_button = QPushButton('Load Training Data', self)
        self.load_data_button.clicked.connect(self.load_training_data)
        layout.addWidget(self.load_data_button)

        self.train_button = QPushButton('Train Model', self)
        self.train_button.clicked.connect(self.train_model)
        self.train_button.setEnabled(False)
        layout.addWidget(self.train_button)

        self.setLayout(layout)

    def load_training_data(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Training Data", "", "NPY Files (*.npy);;All Files (*)", options=options)
        if file_path:
            self.input_data = np.load(file_path, allow_pickle=True)
            print("Loaded input data:", self.input_data)
            self.train_button.setEnabled(True)
        else:
            print("No file selected.")


    def prepare_data(self, input_data):
        print("Input data:", input_data)
        sequence_length = int(self.sequence_length_edit.text() or "100")
        num_epochs = int(self.epochs_edit.text() or self.epochs_edit.placeholderText() or "100")
        batch_size = int(self.batch_size_edit.text() or self.batch_size_edit.placeholderText() or "64")

        flat_input_data = []
        network_output = []

        for song in input_data:
            for track in song:
                sequence_out = None  # Initialize sequence_out to None
                for note in track:
                    if isinstance(note, dict) and 'velocity' in note:
                        flat_input_data.append('name' + str(note['velocity']))
                    elif isinstance(note, dict):
                        flat_input_data.append(str(note))
                    elif isinstance(note, str):
                        flat_input_data.append(note)

                    if sequence_out is None:  # Check if sequence_out has been assigned a value
                        sequence_out = note

                if sequence_out is None:  # If sequence_out was not assigned a value, assign a default value
                    sequence_out = track[0]

                if isinstance(sequence_out, dict) and 'velocity' in sequence_out:
                    network_output.append('name' + str(sequence_out['velocity']))
                elif isinstance(sequence_out, dict):
                    network_output.append(str(sequence_out))
                else:
                    network_output.append(sequence_out)

        unique_values = len(set(flat_input_data))
        note_to_int = {note: num for num, note in enumerate(set(flat_input_data))}
        int_to_note = {num: note for num, note in enumerate(set(flat_input_data))}

        network_input = []
        network_output_int = []

        for i in range(0, len(flat_input_data) - sequence_length, 1):
            sequence_in = flat_input_data[i:i + sequence_length]
            network_input.append([note_to_int[char] for char in sequence_in])
            network_output_int.append(note_to_int[flat_input_data[i + sequence_length]])

        n_patterns = len(network_input)
        network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
        network_input = network_input / float(unique_values)
        network_output = tf.keras.utils.to_categorical(network_output_int, num_classes=unique_values)

        print("Network input:", network_input)

        return network_input, network_output, unique_values, note_to_int, int_to_note






    def train_model(self):
        network_input, network_output, unique_values, note_to_int, int_to_note = self.prepare_data(self.input_data)

        sequence_length = int(self.sequence_length_edit.text() or "100")
        num_epochs = int(self.epochs_edit.text() or self.epochs_edit.placeholderText() or "100")
        batch_size = int(self.batch_size_edit.text() or self.batch_size_edit.placeholderText() or "64")

        model = Sequential()
        model.add(LSTM(512, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(512, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(512))
        model.add(Dense(256))
        model.add(Dropout(0.3))
        model.add(Dense(unique_values))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')

        filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]

        model.fit(network_input, network_output, epochs=num_epochs, batch_size=batch_size, callbacks=callbacks_list)

        self.model_save_path = "trained_model.h5"
        model.save(self.model_save_path)
        self.generate_button.setEnabled(True)
        QMessageBox.information(self, "Success", "Model trained and saved successfully.")


    def parse_midi(self, midi_path):
        midi_file = pretty_midi.PrettyMIDI(midi_path)

        # Tempo information
        tempo_changes = midi_file.get_tempo_changes()
        tempos = [x[0] for x in tempo_changes]
        tempo_times = [x[1] for x in tempo_changes]

        # Time signature information
        time_signature_changes = midi_file.time_signature_changes
        time_signatures = [(x.numerator, x.denominator) for x in time_signature_changes]
        time_signature_times = [x.time for x in time_signature_changes]

        notes = []
        for instrument in midi_file.instruments:
            for note in instrument.notes:
                notes.append({
                    "pitch": note.pitch,
                    "velocity": note.velocity,  # Hinzugefügt
                    "time": note.time,
                    "duration": note.end - note.start,
                    "instrument": instrument.program
                })

        notes = sorted(notes, key=lambda x: (x["time"], x["pitch"], x["duration"]))

        return {
            "notes": notes,
            "tempos": tempos,
            "tempo_times": tempo_times,
            "time_signatures": time_signatures,
            "time_signature_times": time_signature_times
        }


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MidiLstmTrainer()
    ex.show()
    sys.exit(app.exec_())

