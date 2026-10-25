import struct
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import serial
import soundfile as sf
from matplotlib.animation import FuncAnimation

N = 512  # samples per frame, must match the ESP32

port, path = sys.argv[1], sys.argv[2]  # e.g. /dev/ttyACM0 clip.wav

audio, sr = sf.read(path)
if audio.ndim > 1:
    audio = audio[:, 0]  # keep one channel
frames = len(audio) // N
audio = audio[: frames * N]

# send one frame at a time, ESP32 replies with its FFT
ser = serial.Serial(port, 115200, timeout=5)
time.sleep(2)  # opening the port reboots the ESP32, wait for it
ser.reset_input_buffer()
spectra, times = [], []
for i in range(frames):
    frame = (audio[i * N : (i + 1) * N] * 32767).astype(np.int16)
    ser.write(frame.tobytes())
    data = struct.unpack("<256fI", ser.read(256 * 4 + 4))
    spectra.append(data[:256])
    times.append(data[256])
ser.close()
spectra = np.array(spectra)

# check the ESP32 against numpy
expected = [np.abs(np.fft.fft(audio[i * N : (i + 1) * N]))[: N // 2] for i in range(frames)]
print("frames:", frames)
print("average FFT time on ESP32: %.0f us" % np.mean(times))
print("max difference from numpy:", np.max(np.abs(spectra - expected)))

# top: waveform, bottom: frequency content over time
# both are drawn left to right over SECONDS, however fast the ESP32 was
SECONDS, FPS = 5, 30
duration = frames * N / sr
t = np.arange(len(audio)) / sr
db = 20 * np.log10(spectra.T + 1e-6)

plt.style.use("dark_background")
fig, (top, bottom) = plt.subplots(2, 1, sharex=True)
(line,) = top.plot([], [])
top.set_xlim(0, duration)
top.set_ylim(audio.min(), audio.max())
top.set_title("Time domain")
top.set_ylabel("Amplitude")
image = bottom.imshow(
    np.full_like(db, np.nan),
    origin="lower",
    aspect="auto",
    extent=[0, duration, 0, sr / 2],
    vmin=db.min(),
    vmax=db.max(),
)
bottom.set_title("Frequency domain (FFT done on ESP32, %.0f us per frame)" % np.mean(times))
bottom.set_ylabel("Frequency (Hz)")
bottom.set_xlabel("Time (s)")
plt.tight_layout()


def draw(frame):
    done = (frame + 1) / (SECONDS * FPS)  # 0..1
    n = int(done * len(t))
    line.set_data(t[:n], audio[:n])
    shown = np.full_like(db, np.nan)
    cols = int(done * frames)
    shown[:, :cols] = db[:, :cols]
    image.set_data(shown)


animation = FuncAnimation(fig, draw, frames=SECONDS * FPS, interval=1000 / FPS, repeat=False)
plt.show()
