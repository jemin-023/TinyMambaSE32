import os
from scipy.signal import resample_poly
import soundfile as sf

dataset_dir = "speech-noise-dataset"

clean_dir = os.path.join(dataset_dir, "clean_speech")
noisy_dir = os.path.join(dataset_dir, "noisy_speech")

clean_out = os.path.join(dataset_dir, "clean_speech_16k")
noisy_out = os.path.join(dataset_dir, "noisy_speech_16k")

os.makedirs(clean_out, exist_ok=True)
os.makedirs(noisy_out, exist_ok=True)


for fname in sorted(os.listdir(clean_dir)):

    if not fname.endswith(".wav"):
        continue

    clean_path = os.path.join(clean_dir, fname)
    noisy_path = os.path.join(noisy_dir, fname)

    # ---------- CLEAN ----------
    clean_audio, clean_sr = sf.read(clean_path)
    if clean_sr != 16000:
        clean_audio = resample_poly(
            clean_audio,
            16000,
            clean_sr
        )

    sf.write(
        os.path.join(clean_out, fname),
        clean_audio,
        16000
    )
    # ---------- NOISY ----------
    noisy_audio, noisy_sr = sf.read(noisy_path)
    if noisy_sr != 16000:
        noisy_audio = resample_poly(
            noisy_audio,
            16000,
            noisy_sr
        )
    sf.write(
        os.path.join(noisy_out, fname),
        noisy_audio,
        16000
    )
    print(f"Processed: {fname}")