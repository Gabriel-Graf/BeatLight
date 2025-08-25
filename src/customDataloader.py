import os

import torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB


class SoundDSMel(Dataset):
    def __init__(self, df, data_path):
        self.df = df
        self.data_path = str(data_path)
        self.label_dict = {label: idx for idx, label in enumerate(sorted(df['label'].unique()))}
        self.num_classes = 10  # len(self.label_dict)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            # Audiopfad erstellen
            audio_path = self.df.loc[idx, "relative_path"]
            audio_path = os.path.join(self.data_path, audio_path).replace("\\", "/")

            # Audio laden
            audio, sr = self.load_wave(audio_path)

            # Leeres Audio erkennen
            # if audio is None or len(audio.shape) == [1, 128, 282]:
            #     raise ValueError(f"Leeres Audio oder fehlerhafter Dateiinhalt bei Index {idx}: {audio_path}")

            # In Mono konvertieren
            audio, sr = self.wave_to_mono(audio, sr)

            # Mel-Spektrogramm erzeugen
            mel_spec = self.wave_to_melspec(audio, sr, n_fft=4096, hop_length=128, n_mels=256)
            pad_mel = mel_spec[..., :10336]  # 1280 ultraHigh=10336, high=2584 normal=2584 low=1292 ultraLow=1292
            normalized_mel_spec = self.normalize_and_center_melspec(pad_mel)
            mel_tensor = normalized_mel_spec.float()

            # Label holen und umwandeln
            label = self.df.loc[idx, "label"]
            label_num = self.label_dict[label]

            # print(f"audio len: {len(audio.shape)}, mel shape: {mel_spec.shape}")

            return mel_tensor, torch.tensor(label_num)

        except Exception as e:
            label = self.df.loc[idx, "label"]
            print(f"Fehler beim Verarbeiten von Index {idx}_{label}: {e}")

            # Optional: Überspringen, indem ein Dummy-Tensor zurückgegeben wird
            return torch.zeros((128, 128)), torch.tensor(-1)

    @staticmethod
    # 1. Lade die Wave-Datei
    def load_wave(file_path):
        waveform, sample_rate = torchaudio.load(file_path)  # Lädt die Audiodaten und die Sampling-Rate
        return waveform, sample_rate

    @staticmethod
    def wave_to_mono(waveform, sample_rate):
        # Convert to mono if not already
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        return waveform, sample_rate

    @staticmethod
    # 2. Konvertiere die Wave zu einem Mel-Spektrogramm
    def wave_to_melspec(waveform, sample_rate, n_fft=2048, n_mels=128, hop_length=512):
        mel_spectrogram = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        return mel_spectrogram(waveform)

    @staticmethod
    # 3. Logarithmische Skalierung und Normalisierung
    def normalize_and_center_melspec(mel_spec):
        # Logarithmische Transformation (Amplitude in dB umwandeln)
        amp_to_db = AmplitudeToDB(stype="power", top_db=80)  # `top_db` definiert die maximale Dynamik
        mel_spec_db = amp_to_db(mel_spec)

        # Werte zwischen 0 und 1 normalisieren
        mel_spec_normalized = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())

        # Zentrierung um 0 (Skalierung auf -1 bis 1)
        mel_spec_centered = (mel_spec_normalized - 0.5) * 2
        return mel_spec_centered

    @staticmethod
    def pad_melspec(mel_spec):
        pad_length = (mel_spec.size(-1) % 16)  # Erforderliches Padding
        print(pad_length)
        mel_spec = torch.nn.functional.pad(mel_spec, (0, pad_length))  # Rechts auffüllen

        return mel_spec
