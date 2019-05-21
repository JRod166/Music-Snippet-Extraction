import librosa
import librosa.display
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
import time


SAMPLING_RATE = 22050

#audio_files to be extracted
audio_file = ["digital_love.mp3", "pequenas_cosas.mp3"]

#audio loaded files


def load_file(file):
    try:
        logging.info("Loading file {}".format(os.path.basename(file)))
        audio, sr=librosa.load(file,sr=SAMPLING_RATE)
    except:
        logging.info("Couldn't load file {}".format(os.path.basename(file)))
        return

    try:
        logging.info("Generating spectograms for {}".format(os.path.basename(file)))
        D=librosa.stft(audio)
        D_harmonic, D_percussive = librosa.decompose.hpss(D)
        rp = np.max(np.abs(D))

        plt.figure(figsize=(12, 8))

        plt.subplot(3, 1, 1)
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(D), ref=rp), y_axis='log')
        plt.colorbar()
        plt.title('Full spectrogram')

        plt.subplot(3, 1, 2)
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(D_harmonic), ref=rp), y_axis='log')
        plt.colorbar()
        plt.title('Harmonic spectrogram')

        plt.subplot(3, 1, 3)
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(D_percussive), ref=rp), y_axis='log', x_axis='time')
        plt.colorbar()
        plt.title('Percussive spectrogram')
        plt.savefig("{}.png".format(file), box_inches='tight')
        logging.info("Generated spectograms for {}".format(os.path.basename(file)))
    except:
        logging.info("Failed to generate spectograms for {}".format(os.path.basename(file)))

#Setup logger
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',level=logging.INFO)
for item in audio_file:
    load_file(item)
    print("Sleep")
    time.sleep(100)
    print("Wake")
