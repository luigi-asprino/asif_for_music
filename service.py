import sys

from mido import Message, MidiFile, MidiTrack
from harte.harte import Harte
from datasets import load_dataset
from flask import Flask, request, send_file
from text2chords import get_best_chord_sequence_candidate, get_best_lyrics_candidate



def create_midi_from_chords(chords, output_file, tempo=500):
    """
    Genera un file MIDI a partire da una sequenza di accordi.

    :param chords: Una lista di accordi, ciascuno rappresentato da una lista di note MIDI.
    :param output_file: Nome del file MIDI di output.
    :param tempo: Durata di ogni accordo in millisecondi (default: 500 ms).
    """
    # Crea un nuovo file MIDI
    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)

    # Imposta il tempo in base al valore fornito
    track.append(Message('program_change', program=0, time=0))

    for harte_chord in chords:

        chord = [p.midi for p in Harte(harte_chord).pitches]
        # Aggiunge le note dell'accordo
        for note in chord:
            track.append(Message('note_on', note=note, velocity=64, time=0))

        #  Mantiene l'accordo per la durata specificata
        #track.append(Message('note_off', note=chord[0], velocity=64, time=tempo))

        # Rilascia tutte le note dell'accordo
        for note in chord:
            track.append(Message('note_off', note=note, velocity=64, time=tempo))

    # Salva il file MIDI
    midi.save(output_file)


# Flask app
app = Flask(__name__)

@app.route('/generate_midi_from_chords', methods=['POST'])
def generate_midi_from_chords():
    data = request.json

    if not data or 'chords' not in data:
        return {"error": "Missing 'chords' in request body."}, 400

    #chords = get_best_chord_sequence_candidate([data['lyrics']])

    midi_file_path = "output_chords.mid"

    chords_sequence = data["chords"].split()

    create_midi_from_chords(chords_sequence, "output_chords.mid")

    return send_file(midi_file_path, as_attachment=True, download_name="output.mid", mimetype="audio/midi")


@app.route('/get_chord_sequences', methods=['POST'])
def get_chord_sequences():
    data = request.json

    if not data or 'lyrics' not in data:
        return {"error": "Missing 'lyrics' in request body."}, 400

    return get_best_chord_sequence_candidate([data['lyrics']])


@app.route('/get_lyrics', methods=['POST'])
def get_lyrics():
    data = request.json

    if not data or 'chords' not in data:
        return {"error": "Missing 'chords' in request body."}, 400

    return get_best_lyrics_candidate([data['chords']])

app.run(port=int(sys.argv[1]))