import os
import numpy as np
import librosa
import soundfile

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".aif", ".aiff", ".ogg", ".m4a"}
TARGET_SAMPLERATE = 44100
TARGET_SUBTYPE = "PCM_16"

def find_audio_files(directory):
    for root, _, files in os.walk(directory):
        for fname in files:
            if os.path.splitext(fname)[1].lower() in AUDIO_EXTENSIONS:
                yield os.path.join(root, fname)

def standardize(src_path, output_dir, src_root):
    rel_path = os.path.relpath(src_path, src_root)
    out_path = os.path.join(output_dir, os.path.splitext(rel_path)[0] + ".wav")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    audio, _ = librosa.load(src_path, sr=TARGET_SAMPLERATE, mono=True)
    audio_int16 = np.clip(audio, -1.0, 1.0)
    soundfile.write(out_path, audio_int16, TARGET_SAMPLERATE, subtype=TARGET_SUBTYPE)
    return out_path

def main():
    src_dir = os.path.join("data", "drums")
    out_dir = os.path.join("data", "processed")

    audio_files = list(find_audio_files(src_dir))
    print(f"Found {len(audio_files)} audio files in '{src_dir}'")

    for i, src_path in enumerate(audio_files, 1):
        try:
            out_path = standardize(src_path, out_dir, src_dir)
            print(f"[{i}/{len(audio_files)}] {src_path} -> {out_path}")
        except Exception as e:
            print(f"[{i}/{len(audio_files)}] SKIP {src_path}: {e}")

    print("Done.")

if __name__ == "__main__":
    main()