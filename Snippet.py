import librosa
import librosa.display
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse

#Globals
FFT_SIZE=2048 #2^11
HOP_SIZE=512 #2^6
SAMPLING_RATE = 22050
N_MELS= 128
N_MFCCS=12


def normalize(X):
    X += np.abs(X.min())
    X /= X.max()
    return X


def compute_beats(y, sr,oe):

    logging.info("Estimating Beats...")
    tempo, beats_idx = librosa.beat.beat_track(y=y, sr=sr,
                                               hop_length=HOP_SIZE, onset_envelope=oe)
    return beats_idx, librosa.frames_to_time(np.arange(len(oe)), sr=sr,
                                             hop_length=HOP_SIZE)



def process_features(file):

    plt.figure(figsize=(12, 16))
    #Load audio
    try:
        logging.info("Loading file {}".format(os.path.basename(file)))
        audio, sr=librosa.load("audios/{}".format(file),sr=SAMPLING_RATE)
    except exception as e:
        logging.info("Couldn't load file {}".format(os.path.basename(file)))
        logging.info("Error: {}".format(e))
        return

    #Harmonic and Precusive spectrograms
    try:
        logging.info("Generating spectrograms for {}".format(os.path.basename(file)))

        D=librosa.stft(audio)

        logging.info("Compute Harmonic-Percussive separation")

        D_harmonic, D_percussive = librosa.decompose.hpss(D)
        rp = np.max(np.abs(D))
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

        features={}
        S= librosa.feature.melspectrogram(audio,sr=sr,n_fft=FFT_SIZE, hop_length=HOP_SIZE, n_mels=N_MELS)
        plt.subplot(4, 2, 4)
        log_S= librosa.amplitude_to_db(S,ref=np.max)
        librosa.display.specshow(log_S)
        plt.colorbar()
        plt.title('MFCC spectrogram')

        logging.info("MFCC spectrogram: done")

        features["sequence"] =librosa.feature.mfcc(S=log_S, n_mfcc=N_MFCCS)
        plt.subplot(4,2,5)
        librosa.display.specshow(features["sequence"],x_axis="time")
        plt.colorbar()
        plt.title('MFCC spectrogram')

        logging.info("MFCC spectrogram: done")

        chroma=librosa.feature.chroma_cens(y=audio,sr=sr)
        plt.subplot(4,2,6)
        librosa.display.specshow(chroma,y_axis='chroma',x_axis='time')
        plt.colorbar()
        plt.title('Chroma Spectrogram')

        logging.info("Chroma spectrogram: done")

        #Normalize sequence
        features["sequence"]=normalize(features["sequence"])
        plt.subplot(4,2,7)
        librosa.display.specshow(features["sequence"],x_axis="time")
        plt.colorbar()
        plt.title('MFCC normalized')

        logging.info("MFCC normalized spectrogram: done")
        logging.info("Generated spectrograms for {}".format(os.path.basename(file)))

    except Exception as e:
        logging.info("Failed to generate spectrograms for {}".format(os.path.basename(file)))
        logging.info("Error: {}".format(e))

    try:
        onset_env = librosa.onset.onset_strength(audio, sr=sr,aggregate=np.median)
        features["beats_idx"], times = compute_beats(audio,sr=sr,oe=onset_env)
        features["beats"]=times;
        plt.subplot(4,2,8)
        plt.plot(times,librosa.util.normalize(onset_env),label='Onset Strength')
        plt.vlines(times[features["beats_idx"]],0,1,alpha=0.5,color='r',linestyle='--',label='Beats')
        plt.legend(frameon=True, framealpha=0.75)
        plt.gca().xaxis.set_major_formatter(librosa.display.TimeFormatter())
        plt.title('Beat track')
        logging.info("Generated beat-track")

    except Exception as e:
        logging.info("Failed Beat Estimation")
        logging.info("Error: {}".format(e))

    plt.savefig("graphs/{}.png".format(file), box_inches='tight')

if __name__ == '__main__':
    #Setup logger
    parser = argparse.ArgumentParser(description="Generates a music snippet form a given audio file",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("audio_file",
                        action="store",
                        help="Input audio file")
    parser.add_argument("-D",
                        action="store_const",
                        dest="Debug",
                        const=True,
                        default=False,
                        help="Debug mode")
    args=parser.parse_args()
    if(args.Debug):
        logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',level=logging.INFO)
    process_features(args.audio_file)
