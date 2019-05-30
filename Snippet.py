import librosa
import librosa.display
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
import time

#Globals
FFT_SIZE=2048 #2^11
HOP_SIZE=512 #2^6
SAMPLING_RATE = 22050
N_MELS= 128
N_MFCCS=12

#audio_files to be extracted
audio_file = ["digital_love.mp3", "pequenas_cosas.mp3","heartbeat.mp3","mindless.ogg","sines.ogg"]



def process_file(file):

    #Load audio
    try:
        logging.info("Loading file {}".format(os.path.basename(file)))
        audio, sr=librosa.load(file,sr=SAMPLING_RATE)
    except:
        logging.info("Couldn't load file {}".format(os.path.basename(file)))
        return

    #Harmonic and Precusive spectrograms
    try:
        logging.info("Generating spectrograms for {}".format(os.path.basename(file)))

        D=librosa.stft(audio)

        logging.info("Compute Harmonic-Percussive separation")

        D_harmonic, D_percussive = librosa.decompose.hpss(D)
        rp = np.max(np.abs(D))
        plt.figure(figsize=(12, 16))
        plt.subplot(4, 2, 1)
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(D), ref=rp), y_axis='log')
        plt.colorbar()
        plt.title('Full spectrogram')

        logging.info("Full spectrogram: done")

        plt.subplot(4, 2, 2)
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(D_harmonic), ref=rp), y_axis='log')
        plt.colorbar()
        plt.title('Harmonic spectrogram')

        logging.info("Harmonic spectrogram: done")

        plt.subplot(4, 2, 3)
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(D_percussive), ref=rp), y_axis='log', x_axis='time')
        plt.colorbar()
        plt.title('Percussive spectrogram')

        logging.info("Percussive spectrogram: done")

    except:
        logging.info("Failed to generate spectrograms for {}".format(os.path.basename(file)))
    try:
        logging.info("Computing MFCC spectrogram")

        features={}
        S= librosa.feature.melspectrogram(audio,sr=sr,n_fft=FFT_SIZE, hop_length=HOP_SIZE, n_mels=N_MELS)
        plt.subplot(4, 2, 4)
        log_S= librosa.amplitude_to_db(S,ref=np.max)
        librosa.display.specshow(log_S)
        plt.colorbar()
        plt.title('MFCC spectrogram')

        logging.info("Generated spectrograms for {}".format(os.path.basename(file)))

        features["sequence"] =librosa.feature.mfcc(S=log_S, n_mfcc=N_MFCCS).T
        plt.subplot(4,2,5)
        librosa.display.specshow(features["sequence"],x_axis="time")
        plt.colorbar()
        plt.title('MFCC spectrogram')

        chroma=librosa.feature.chroma_cens(y=audio,sr=sr)
        plt.subplot(4,2,6)
        librosa.display.specshow(chroma,y_axis='chroma',x_axis='time')
        plt.colorbar()
        plt.title('Chroma Spectrogram')

        plt.savefig("{}.png".format(file), box_inches='tight')
    except Exception as e:
        raise


#Setup logger
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',level=logging.INFO)
for item in audio_file:
    process_file(item)
