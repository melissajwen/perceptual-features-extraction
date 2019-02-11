from scipy import signal, fftpack
from scipy.io import wavfile
import numpy as np
import math
import matplotlib.pyplot as plot

GROUND_TRUTH_FILE = "music_speech.mf"
RESULT_FILE = "result.arff"

BUFFER_LEN = 1024
HOP_SIZE = 512
NUM_FILTERS = 26

PRECISION = "%.6f"


def main():
    # Open the necessary files
    ground_truth = open(GROUND_TRUTH_FILE, "r")

    # Write header to output file
    write_header()

    # Initialize the mel filters
    filters = np.zeros((26, 513))

    # Main loop to read in file
    for line in ground_truth:
        line_arr = line.split("\t")
        wav_file_name = "music_speech/" + line_arr[0].strip()
        wav_file_type = line_arr[1].strip()

        # Split up wav file into buffers
        freq, file_data = wavfile.read(wav_file_name)
        buffers = separate_into_buffers(file_data)

        # Calculate the MFCC
        # Apply pre-emphasis filtering and windowing
        buffers = pre_emphasis_filtering(buffers)
        buffers *= signal.hamming(BUFFER_LEN)

        # Perform FFT and discard negative frequencies
        data = fftpack.fft(buffers)
        buffers = np.abs(data[:, :int(BUFFER_LEN / 2 + 1)])  # only keep the first half

        # Convert the sample frequency to mel-scale
        mel = freq_to_mel(freq / 2) / (NUM_FILTERS + 1)

        # Calculate the mel frequency filters
        for i in range(NUM_FILTERS):
            # Calculate the x-axis points of the left, top, and right points of the triangle
            left_point = math.floor(mel_to_freq(mel * i) / (freq / BUFFER_LEN))
            top_point = round(mel_to_freq(mel * (i + 1)) / (freq / BUFFER_LEN))
            right_point = math.ceil(mel_to_freq(mel * (i + 2)) / (freq / BUFFER_LEN))

            # Linearly interpolate in between the x-axis points
            left_bin = np.linspace(0, 1, top_point - left_point + 1)
            right_bin = np.linspace(1, 0, right_point - top_point + 1)
            right_bin = np.delete(right_bin, 0)  # delete the duplicate 1

            # Combine all bins to produce the full x-axis range of the triangle
            filters[i] = np.concatenate((np.zeros(left_point), left_bin, right_bin, np.zeros(512 - right_point)))

        # Take the dot product and apply log
        transposed = filters.T
        result = np.dot(buffers, transposed)
        result = np.log10(result)

        # Perform DCT
        result = fftpack.dct(result)

        # Write results to ARFF file
        write_result(result.T, wav_file_type)

    plot_entire_range(filters, freq)
    plot_300(filters, freq)


# Function to write header of ARFF result file
def write_header():
    output = open(RESULT_FILE, 'w')
    output.write('@RELATION music_speech\n')

    for i in range(NUM_FILTERS):
        output.write('@ATTRIBUTE MEAN_%d NUMERIC\n' % (i + 1))

    for i in range(NUM_FILTERS):
        output.write('@ATTRIBUTE STD_%d NUMERIC\n' % (i + 1))

    output.write('@ATTRIBUTE class {music,speech}\n')
    output.write('@DATA\n')


# Function to write @DATA section of ARFF result file
def write_result(result, file_type):
    output = open(RESULT_FILE, 'a')

    mean = []
    std = []

    for mfcc_bin in result:
        mean.append(np.mean(mfcc_bin))
        std.append(np.std(mfcc_bin))

    mfcc_str = ""
    for i in range(len(mean)):
        mfcc_str += PRECISION % mean[i] + ","

    for i in range(len(std)):
        mfcc_str += PRECISION % std[i] + ","

    output.write(mfcc_str + file_type + "\n")


# Function that takes in wav file data and separates into buffers
# Using a length of 1024 for each buffer and a 50% hop size (len 512)
def separate_into_buffers(file_data):
    # Convert to samples
    data = file_data / 32768.0

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


# Apply pre-emphasis filtering on input buffer
def pre_emphasis_filtering(buffers):
    results = []

    for buf in buffers:
        res = [buf[0]]

        for i in range(len(buf) - 1):
            y = buf[i + 1] - 0.95 * buf[i]
            res.append(y)

        results.append(res)

    return results


# Convert frequency to mel-scale
def freq_to_mel(freq):
    mel = 1127 * np.log(1 + freq / 700)
    return mel


# Convert mel-scale to frequency
def mel_to_freq(mel):
    freq = 700 * (math.exp(mel / 1127) - 1)
    return freq


# Function to plot the filters over its entire range
def plot_entire_range(filters, freq):
    x_axis = np.linspace(0, freq / 2, 513)

    for i in range(NUM_FILTERS):
        plot.plot(x_axis, filters[i])

    plot.xlim(0, 12000)
    plot.ylim(0, 1)

    plot.title("26 Triangular MFCC filters, 22050Hz signal, window size 1024")
    plot.xlabel("Frequency (Hz)")
    plot.ylabel("Amplitude")
    plot.savefig("plot_1.png")


# Function to plot the filters from 0-300 on its x-axis
def plot_300(filters, freq):
    x_axis = np.linspace(0, freq / 2, 513)

    for i in range(NUM_FILTERS):
        plot.plot(x_axis, filters[i], 'o')

    plot.xlim(0, 300)
    plot.ylim(0, 1)

    plot.title("26 Triangular MFCC filters, 22050Hz signal, window size 1024")
    plot.xlabel("Frequency (Hz)")
    plot.ylabel("Amplitude")
    plot.savefig("plot_2.png")


main()
