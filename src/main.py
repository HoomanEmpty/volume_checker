import subprocess
import numpy as np

#calculate volume of the voice
def RMS(process, width = 30):

    data = process.stdout.read(4096)
    samples = np.frombuffer(data, dtype=np.int16)
    samples = samples.astype(np.float32) / 32768.0
    rms = np.sqrt(np.mean(samples ** 2))

    bars = int(rms * 100)
    bars = min(bars, width)

    print(
        f"\r{'#' * bars}{'.' * (width - bars)} \t\t\t Volume: {bars:.4f}",
        end="",
        flush=True
    )
 

def get_mic_list():
    mic_list = subprocess.Popen(
        [
            "pactl",
            "list",
            "short",
            "sources"
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )

    mic_list = mic_list.stdout.read()
    lines = mic_list.splitlines()
    parts = []
    part = []
    counter = 1
    for i in lines:
        parts.append(i.split())


    for i in range(len(parts)):
        if parts[i][1].startswith("alsa_input"):
            part.append([counter, parts[i][1], parts[i][-1]])
            counter += 1

    return part


#get voice from mic whitout saving
def get_voice(mic):
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-f", "pulse",
            "-i", mic,

            "-f", "s16le",
            "-ac", "1",
            "-ar", "44100",
            "-"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )

    return process


try:
    show_mics = get_mic_list()

    for i in show_mics:
        print(f"{i[0]}- {i[1]:<80}[{i[2]}]")

    choose_mic = int(input("choose your mic: "))
    choose_mic = show_mics[choose_mic - 1][1]
    process_mic = get_voice(choose_mic)

    print("\n\n Ctrl + C for quit \n\n")

    while True:
        RMS(process_mic)

except KeyboardInterrupt:
    process_mic.terminate()

finally:
    process_mic.terminate()