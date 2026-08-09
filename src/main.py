import subprocess
import numpy as np
import time


# =========================
# Global variables
# =========================

rms_list = np.array([])

state = "CALIBRATING"
start_time = None

calibration_start = time.monotonic()
calibration_duration = 2.0

# Noise detection
noise_floor = 0.001
alpha = 0.05

silence_multiplier = 2.0
silence_threshold = 0.0

# STD detection
std_floor = 0.001
std_alpha = 0.05

std_multiplier = 2.0
std_threshold = 0.0

# Frequency detection
frequency_floor = 0.0
frequency_alpha = 0.05

frequency_std = 0.0
frequency_std_multiplier = 2.0

min_frequency = 0.0
max_frequency = 0.0

frequency_initialized = False

# Talking detection
talking_score_threshold = 1

# AGC (Automatic Gain Control)
agc_gain = 1.0
target_rms = 0.05
max_gain = 8
agc_alpha = 0.02

# =========================
# Show information
# =========================

def show_information(samples, display_samples=None, width=25, sensitivity=1000):
    global noise_floor, silence_threshold, state

    # display_samples (post-AGC) is only used for the volume bar.
    # All detection/baseline logic below uses raw `samples`, so the
    # scale never mixes with the AGC gain.
    if display_samples is None:
        display_samples = samples

    # RMS
    rms, rms_list, rms_mean = rms_calculate(samples)

    # STD
    std = np.sqrt(np.mean((rms_list - rms_mean) ** 2))

    # Frequency
    magnitude = np.abs(np.fft.rfft(samples))

    peak_index = np.argmax(magnitude)

    frequencies = np.fft.rfftfreq(len(samples), d=1 / 44100)

    dominant_frequency = frequencies[peak_index]

    # Display-only RMS (post-AGC), used solely for the volume bar
    display_rms = np.sqrt(np.mean(display_samples ** 2))

    # =========================
    # Calibration
    # =========================

    if state == "CALIBRATING":

        noise_detection(rms)

        std_calculate(std)

        #when the signal not strong
        if rms <= noise_floor * 3:
            frequency_calculate(dominant_frequency)

        elapsed = time.monotonic() - calibration_start

        if elapsed >= calibration_duration:

            silence_detection(noise_floor)

            state = "SILENCE"

        action = state

    # =========================
    # Normal operation
    # =========================

    else:

        action = is_talking(rms, silence_threshold, std, dominant_frequency)

    # =========================
    # Volume bar
    # =========================

    bars = int(display_rms * sensitivity)
    bars = min(bars, width)

    if action == "WAITING_FOR_SILENCE":
        action = "TALKING"

    print(
        f"\r"
        f"{'#' * bars}"
        f"{'.' * (width - bars)}"
        f"\t\t Volume: {display_rms:.4f}"
        f" Frequency: {dominant_frequency:8.2f}Hz"
        f" Std: {std:.4f}"
        f" Action: {action:<11}",
        end="",
        flush=True
    )


# =========================
# RMS calculation
# =========================

def rms_calculate(volume):
    global rms_list

    rms = np.sqrt(np.mean(volume ** 2))

    rms_list = np.append(rms_list, rms)

    if len(rms_list) > 100:
        rms_list = np.delete(rms_list, 0)

    rms_mean = np.mean(rms_list)

    return rms, rms_list, rms_mean


# =========================
# STD baseline
# =========================

def std_calculate(std):
    global std_floor, std_threshold

    std_floor = (std_alpha * std + (1 - std_alpha) * std_floor)

    std_threshold = (std_floor * std_multiplier)


# =========================
# Frequency baseline
# =========================

def frequency_calculate(dominant_frequency):
    global frequency_floor, frequency_std, min_frequency, max_frequency, frequency_initialized

    #first value
    if not frequency_initialized:

        frequency_floor = dominant_frequency
        frequency_std = 0.0

        frequency_initialized = True

        min_frequency = frequency_floor
        max_frequency = frequency_floor

        return

    #old baseline
    old_frequency_floor = frequency_floor

    #baseline for EMA
    frequency_floor = (frequency_alpha * dominant_frequency + (1 - frequency_alpha) * frequency_floor)

    # EMA variance
    variance = (
        frequency_alpha * (dominant_frequency - old_frequency_floor) ** 2 + (1 - frequency_alpha) * frequency_std ** 2)

    frequency_std = np.sqrt(variance)

    #frequency range
    min_frequency = (frequency_floor - frequency_std_multiplier * frequency_std)

    max_frequency = (frequency_floor + frequency_std_multiplier * frequency_std)

def auto_gain_control(samples, current_rms):
    global agc_gain, noise_floor, agc_alpha

    if current_rms > (noise_floor * 1.2):
        current_level = current_rms * agc_gain
        error = target_rms - current_level

        agc_gain += error * agc_alpha

        # Clamp gain to [1.0, max_gain]
        agc_gain = max(1.0, min(agc_gain, max_gain))

    else:
        agc_gain += (1.0 - agc_gain) * (agc_alpha / 2.0)

    amplified_samples = samples * agc_gain

    # Prevent clipping beyond the valid sample range
    amplified_samples = np.clip(amplified_samples, -1.0, 1.0)

    return amplified_samples

# =========================
# Update noise baseline
# =========================

def update_baseline(rms):

    noise_detection(rms)
    silence_detection(noise_floor)


# =========================
# Talking detection
# =========================

def is_talking(rms, silence_threshold, std, frequency):
    global state, start_time
    
    # =========================
    # Silence
    # =========================

    if rms <= silence_threshold:

        #Update only noise
        update_baseline(rms)

        # TALKING -> WAITING
        if state == "TALKING":

            state = "WAITING_FOR_SILENCE"

            start_time = time.monotonic()

        # WAITING -> SILENCE
        elif state == "WAITING_FOR_SILENCE":

            elapsed = (time.monotonic() - start_time)

            if elapsed >= 1.0:

                state = "SILENCE"

                start_time = None

        return state

    # =========================
    # Talking score
    # =========================

    score = 0

    # STD
    if std >= std_threshold:
        score += 1

    # Frequency
    if (frequency < min_frequency or frequency > max_frequency):
        score += 1

    # =========================
    # Talking
    # =========================

    if score >= talking_score_threshold:

        if state == "WAITING_FOR_SILENCE":

            start_time = None

        state = "TALKING"

    return state


# =========================
# Noise detection
# =========================

def noise_detection(rms):
    global noise_floor

    noise_floor = (alpha * rms + (1 - alpha) * noise_floor)

    return noise_floor


# =========================
# Silence threshold
# =========================

def silence_detection(noise_floor):
    global silence_threshold

    silence_threshold = (noise_floor * silence_multiplier)

    return silence_threshold


# =========================
# Microphone list
# =========================

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

        parts.append(
            i.split()
        )

    for i in range(len(parts)):

        if parts[i][1].startswith(
            "alsa_input"
        ):

            part.append(
                [
                    counter,
                    parts[i][1],
                    parts[i][-1]
                ]
            )

            counter += 1

    return part


# =========================
# Get voice from microphone
# =========================

def get_voice(mic):

    process = subprocess.Popen(
        [
            "ffmpeg",
            "-f",
            "pulse",
            "-i",
            mic,
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            "44100",
            "-"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )

    return process


# =========================
# Main
# =========================

try:

    show_mics = get_mic_list()

    # Show microphones

    for i in show_mics:

        print(f"{i[0]}- {i[1]:<80}[{i[2]}]")

    # Choose microphone

    choose_mic = int(input("choose your mic: "))

    choose_mic = (show_mics[choose_mic - 1][1])

    # Get voice

    process = get_voice(choose_mic)

    print("\n\nCtrl + C for quit\n\n")

    # Main loop

    while True:

        data = (process.stdout.read(4096))

        if not data:
            break

        samples = np.frombuffer(data, dtype=np.int16)

        samples = (samples.astype(np.float32) / 32768.0)

        temp_rms = np.sqrt(np.mean(samples ** 2))

        if state != "CALIBRATING":
            display_samples = auto_gain_control(samples, temp_rms)
        else:
            display_samples = samples

        show_information(samples, display_samples)


except KeyboardInterrupt:

    process.terminate()

finally:

    process.terminate()
    process.stdout.close()