import subprocess
import numpy as np

#calculate volume of the voice
def RMS(process, width = 30):

    data = process.stdout.read(4096)
    samples = np.frombuffer(data, dtype=np.int16)
    samples = samples.astype(np.float32) / 32768.0
    rms = np.sqrt(np.mean(samples ** 2))

    bars = int(rms * 1000)
    bars = min(bars, width)

    print(
        f"\r{'#' * bars}{'.' * (width - bars)} \t\t\t Volume: {rms * 1000:.4f}",
        end="",
        flush=True
    )

    

#get voice from mic whitout saving
process = subprocess.Popen(
    [
        "ffmpeg",
        "-f", "pulse",
        "-i", "alsa_input.usb-Jin-Audio_GM306_AP5980_20220308-00.analog-stereo",

        "-f", "s16le",
        "-ac", "1",
        "-ar", "44100",
        "-"
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL
)


try:
    while True:
        RMS(process)
        

except KeyboardInterrupt:
    process.terminate()