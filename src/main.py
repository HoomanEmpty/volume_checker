import subprocess
import numpy as np
import time

rms_list = np.array([])
start_time = None
silence_threshold = 0.001

#calculate volume of the voice
def show_information(volume, frequency, width = 25, sensitivity = 1000):
    global start_time
    global silence_threshold

    rms, rms_list, rms_mean = rms_calculate(volume)

    std = std_calaulate(rms_list, rms_mean)
    
    bars = int(rms * sensitivity)
    bars = min(bars, width)

    if rms <= silence_threshold:
        start_time = time.monotonic()
        is_talking()
    


    print(
        f"\r{'#' * bars}{'.' * (width - bars)} \t\t\t Volume: {bars:.4f} Frequency: {frequency:.4f}Hz Std: {std:.4f} Action: {is_talking}",
        end="",
        flush=True
    )

def rms_calculate(volume):
    global rms_list

    rms = np.sqrt(np.mean(volume ** 2))
    rms_list = np.append(rms_list, [rms])

    if len(rms_list) > 100:
        rms_list = np.delete(rms_list, 0)
    rms_mean = np.mean(rms_list)

    return rms, rms_list, rms_mean

def std_calaulate(rms_list, rms_mean):
    std = np.sqrt(np.mean((rms_list - rms_mean) ** 2))

    return std

def is_talking():
    ...

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

    #show your activate microphone
    for i in show_mics:
        print(f"{i[0]}- {i[1]:<80}[{i[2]}]")

    #choose your microphone
    choose_mic = int(input("choose your mic: "))
    choose_mic = show_mics[choose_mic - 1][1]
 
    #get info from your voice
    process = get_voice(choose_mic)

    print("\n\n Ctrl + C for quit \n\n")

    while True:
        
        data = process.stdout.read(4096)

        samples = np.frombuffer(data, dtype=np.int16)
        samples = samples.astype(np.float32) / 32768.0

        magnitude = np.abs(np.fft.rfft(samples))
        peak_index = np.argmax(magnitude)
        frequencies = np.fft.rfftfreq(len(samples), d = 1/44100)
        dominant_frequency = frequencies[peak_index]

        
        show_information(samples, dominant_frequency)

except KeyboardInterrupt:
    process.terminate()

finally:
    process.terminate()