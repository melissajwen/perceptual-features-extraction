from scipy import signal, fft
from scipy.io import wavfile
import numpy as np

GROUND_TRUTH_FILE = "music_speech.mf"
RESULT_FILE = "result.arff"

BUFFER_LEN = 1024
HOP_SIZE = 512

# PRECISION = "%.6f"

# HEADER = "@RELATION music_speech\n" \
#          "@ATTRIBUTE RMS_MEAN NUMERIC\n" \
#          "@ATTRIBUTE ZCR_MEAN NUMERIC\n" \
#          "@ATTRIBUTE SC_MEAN NUMERIC\n" \
#          "@ATTRIBUTE SRO_MEAN NUMERIC\n" \
#          "@ATTRIBUTE SFM_MEAN NUMERIC\n" \
#          "@ATTRIBUTE RMS_STD NUMERIC\n" \
#          "@ATTRIBUTE ZCR_STD NUMERIC\n" \
#          "@ATTRIBUTE SC_STD NUMERIC\n" \
#          "@ATTRIBUTE SRO_STD NUMERIC\n" \
#          "@ATTRIBUTE SFM_STD NUMERIC\n" \
#          "@ATTRIBUTE class {music,speech}\n" \
#          "@DATA\n"


def main():
    # Open the necessary files
    ground_truth = open(GROUND_TRUTH_FILE, "r")
    result = open(RESULT_FILE, "w")

    # Write header to output file
    # result.write(HEADER)

    # Main loop to perform wav calculations
    for line in ground_truth:
        line_arr = line.split("\t")
        wav_file_name = "music_speech/" + line_arr[0].strip()
        wav_file_type = line_arr[1].strip()

        # Split up wav file into buffers
        buffers = get_buffers_from_wav(wav_file_name)


# Function to calculate buffers
def get_buffers_from_wav(wav_file_name):
    freq, file_data = wavfile.read(wav_file_name)
    data = file_data / 32768.0  # convert to samples

    buffers = []
    start = 0
    end = BUFFER_LEN
    num_buffers = int(len(file_data) / HOP_SIZE - 1)

    for i in range(num_buffers):
        buffer_data = data[start:end]
        start += HOP_SIZE
        end += HOP_SIZE

        if len(buffer_data) == BUFFER_LEN:
            buffers.append(buffer_data)

    return buffers


# Convert buffers from time domain to frequency domain
def get_windows_from_buffers(buffers):
    windows = []

    for buf in buffers:
        win = buf * signal.hamming(len(buf))
        win = fft(win)

        # Only keep the first half of the array
        win = win[:int(len(win) / 2 + 1)]
        windows.append(win)

    return windows


main()