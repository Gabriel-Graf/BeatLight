import csv
import os

from matplotlib.pyplot import title
from mutagen.mp3 import MP3
from mutagen.id3 import ID3

from pydub import AudioSegment

# Ordnerpfade
AudioSegment.ffmpeg = r"C:\Program Files\ffmpeg\bin"


def process_audio(file_path, output_folder, num_segments=4, segment_size=30, skip_intro=30, max_length_skip_intro=180):
    try:
        # Audio-Datei laden
        audio = AudioSegment.from_file(file_path, format="mp3")
        song_length_ms = len(audio)
        song_length_sec = song_length_ms / 1000

        # Initialen Trim bestimmen
        initial_trim_sec = skip_intro if song_length_sec > max_length_skip_intro else 0
        remaining_duration_sec = song_length_sec - initial_trim_sec

        # Anzahl möglicher Segmente berechnen
        n_segments = min(num_segments, int(remaining_duration_sec // segment_size))
        if n_segments < 1:
            print(f"{file_path} ist zu kurz für ein Segment")
            return

        # Vorhandene Dateien zählen für Benennung
        existing_files = len([f for f in os.listdir(output_folder) if os.path.isfile(os.path.join(output_folder, f))])
        folder_name = os.path.basename(output_folder).replace(" ", "")

        title = os.path.basename(file_path).replace(".mp3", "")

        for i in range(n_segments):
            # Startzeit berechnen
            if n_segments == 1:
                start_sec = initial_trim_sec
            else:
                start_sec = initial_trim_sec + (i * (remaining_duration_sec - 30)) / (n_segments - 1)

            start_ms = int(start_sec * 1000)
            end_ms = start_ms + 30 * 1000

            # Sicherstellen, dass Ende nicht überschritten wird
            end_ms = min(end_ms, song_length_ms)

            # Audio schneiden
            segment = audio[start_ms:end_ms]

            # Dateiname generieren
            output_file = os.path.join(
                output_folder,
                f"{folder_name}_{title}_{existing_files + i + 1:05d}.wav"
            )

            # Exportieren
            segment.export(output_file, format="wav")
            print(f"Segment {i + 1}/{n_segments} erstellt: {output_file}")

    except Exception as e:
        print(f"Fehler bei {file_path}: {e.with_traceback()}")


def process_audio_dir(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # Alle MP3-Dateien im Input-Ordner durchlaufen
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".mp3"):
            file_path = os.path.join(input_folder, file_name)
            process_audio(file_path, output_folder)


def create_csv(input_folder, output_csv):
    # CSV schreiben
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # Header schreiben
        writer.writerow(['filename', 'label'])

        # Durch alle Unterordner und Dateien gehen
        for root, _, files in os.walk(input_folder):
            for file in files:
                if file.endswith('.wav'):
                    # Dateiname und Subordner-Name holen
                    filename = file
                    label = os.path.basename(root).replace(" ", "")
                    # In die CSV schreiben
                    writer.writerow([filename, label])

    print(f"CSV-Datei wurde erfolgreich erstellt: {output_csv}")


if __name__ == '__main__':
    in_dir = r"Dataset_GTZAN/samples/pop"
    out_dir = r"../Dataset/samples/Grindcore"
    # process_audio_dir(in_dir, out_dir)
    create_csv(r"Dataset_GTZAN/samples", r"Dataset_GTZAN/metadata.csv")
