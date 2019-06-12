import librosa
import librosa.display
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import chroma_exp
from collections import Counter
from scipy.io import wavfile
from scipy.spatial import distance
from sklearn.cluster import KMeans

#Globals
FFT_SIZE=2048 #2^11
HOP_SIZE=512 #2^6
SAMPLING_RATE = 22050
N_MELS= 128
N_MFCCS=16


def normalize(X):
    X += np.abs(X.min())
    X /= X.max()
    return X


def time_to_sample(time,sr):
    return int(time*sr)

def most_frequent(List):
    occurence_count = Counter(List)
    #print(occurence_count)
    return occurence_count.most_common(1)[0][0]

def get_indeces(List):
    indeces=[]
    number=most_frequent(List)
    for i in range(0,len(List)):
        if(List[i]==number):
            indeces.append(i)
    return indeces

def compute_beats(y, sr):

    logging.info("Estimating Beats...")
    tempo, beats_idx = librosa.beat.beat_track(y=y, sr=sr,
                                               hop_length=HOP_SIZE)
    return beats_idx, librosa.frames_to_time(beats_idx, sr=sr,
                                             hop_length=HOP_SIZE)


def process_features(file):

    plt.figure(figsize=(12, 18))
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
        #Lu y Zhang --> MFCC
        S= librosa.feature.melspectrogram(audio,sr=sr,n_fft=FFT_SIZE, hop_length=HOP_SIZE, n_mels=N_MELS)
        plt.subplot(4, 2, 4)
        log_S= librosa.amplitude_to_db(S,ref=np.max)
        librosa.display.specshow(log_S)
        plt.colorbar()
        plt.title('MFCC')

        logging.info("MFCC spectrogram: done")

        features["sequenceMFCC"] =librosa.feature.mfcc(S=log_S, n_mfcc=N_MFCCS)
        plt.subplot(4,2,5)
        librosa.display.specshow(features["sequenceMFCC"],x_axis="time")
        plt.colorbar()
        plt.title('MFCC spectrogram')
        features["sequenceMFCC"]=features["sequenceMFCC"].T

        logging.info("MFCC spectrogram: done")

        #Xu et al. --> Chroma
        features["sequenceChroma"]=librosa.feature.chroma_cqt(y=audio,sr=sr,hop_length=HOP_SIZE)
        plt.subplot(4,2,6)
        librosa.display.specshow(features["sequenceChroma"],y_axis='chroma',x_axis='time')
        plt.colorbar()
        plt.title('Chroma Spectrogram')
        features["sequenceChroma"]=features["sequenceChroma"].T

        logging.info("Chroma spectrogram: done")

        #Normalize sequence
        features["sequenceMFCC"]=normalize(features["sequenceMFCC"])

        #Normalize sequence
        features["sequenceChroma"]=normalize(features["sequenceChroma"])

        logging.info("Normalized features: done")

        logging.info("Generated spectrograms for {}".format(os.path.basename(file)))
    except Exception as e:
        logging.info("Failed to generate spectrograms for {}".format(os.path.basename(file)))
        logging.info("Error: {}".format(e))

    try:
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr,aggregate=np.median)
        features["beats_idx"],features["beats"] = compute_beats(audio,sr=sr)
        times=librosa.frames_to_time(np.arange(np.abs(D).shape[1]))
        plt.subplot(4,2,7)
        plt.plot(times,librosa.util.normalize(onset_env),label='Onset Strength')
        plt.vlines(times[features["beats_idx"]],0,1,alpha=0.5,color='r',linestyle='--',label='Beats')
        plt.legend(frameon=True, framealpha=0.75)
        plt.gca().xaxis.set_major_formatter(librosa.display.TimeFormatter())
        plt.title('Power Spectrogram')
        logging.info("Generated beat-track")

    except Exception as e:
        logging.info("Failed Beat Estimation")
        logging.info("Error: {}".format(e))

    plt.savefig("graphs/{}.png".format(file), box_inches='tight')
    return features, audio

def filter(List,positioning,M):
    third_part=round(M/3)
    Lista=[]
    if positioning=="beginning":
        start=0
        end=start+third_part
    elif positioning=="middle":
        start=third_part+1
        end=2*third_part
    elif positioning=="end":
        start=(third_part*2)+1
        end=M
    else:
        return List
    for i in List:
        if(i<end and i>start):
            Lista.append(i)
    if(len(Lista)>0):
        return Lista
    else:
        return List


def find_idxs(sequence, P=1, N=16, L=None,positioning="None"):
    """Parameters
    ----------
    sequence : np.array(M, n_features)
        Representation of the audio track on beats.
    N : int > 0
        Numnber of beats per subsequence.
    L : int > 0 < N
        Length of the shingles (If None, L = N / 4)

    Returns
    -------
    idxs : list (len == P)
        List of indeces
    """
    assert len(sequence) >= P * N

    M = len(sequence)
    if L is None:
        L = int(N / 4)
    x=64
    while(x>=len(sequence)):
        x=x/2
    kmeans=KMeans(n_clusters=int(x)).fit(sequence)
    idx=get_indeces(kmeans.labels_)
    if positioning!=None:
        idx=filter(idx,positioning,M)
    #print(idx)

    return idx

def find_idxs_chroma(sequence, P=1, N=16, L = None):
    assert len(sequence) >= P * N

    M = len(sequence)
    if L is None:
        L = int(N / 4)
    cont=0
    patterns=chroma_exp.get_indeces(sequence)
    pattern_features=[]
    for i in patterns:
        pattern_features.append(sequence[i])
    x=64
    while(x>=len(pattern_features)):
        x=x/2
    kmeans=KMeans(n_clusters=int(x)).fit(pattern_features)
    idx_pf=get_indeces(kmeans.labels_)
    idx=[]
    for i in idx_pf:
        idx.append(patterns[i])

    return idx

def subsequent_idxs(idxs):
    temp_start=idxs[0]
    end=0
    start=0
    temp_end=0
    cont=1
    max_cont=0
    for i in range (0,len(idxs)-1):
        if (idxs[i]+1==idxs[i+1]):
            cont+=1
        else:
            temp_end=idxs[i]
            max_cont=max(cont,max_cont)
            if(max_cont==cont):
                end=temp_end
                start=temp_start
            temp_start=idxs[i+1]
            cont=1
    return [start,end]

def synth_snippet(audio,beats,idxs,N):
    logging.info("Synthesizing snippet")
    sr=SAMPLING_RATE
    fade=2
    n=len(audio)
    snippet=np.empty(0)
    idxs=subsequent_idxs(idxs)
    #print(idxs)
    # Create Subsequence
    start_time=beats[int(idxs[0])]
    end_time=beats[int(idxs[1]+(N))]
    while(end_time-start_time<18):
        if start_time-N>=0:
            start_time-=N
        if end_time+N<=n:
            end_time+=N
    while(end_time-start_time>18):
        end_time-=1
    start_sample=time_to_sample(start_time+1,sr)
    end_sample=time_to_sample(end_time-1,sr)
    subseq_audio = audio[start_sample:end_sample]

    fade_seg_start=time_to_sample(np.max([0,start_time-1]),sr)
    fade_seg_end=time_to_sample(np.min([n-1,start_time+1]),sr)
    n_samples=fade_seg_end-fade_seg_start
    mask=np.arange(n_samples) / float(n_samples)
    fade_in=audio[fade_seg_start:fade_seg_end]*mask

    fade_seg_start=time_to_sample(np.max([0,end_time-1]),sr)
    fade_seg_end=time_to_sample(np.min([n-1,end_time+1]),sr)
    n_samples=fade_seg_end-fade_seg_start
    mask=1-(np.arange(n_samples) / float(n_samples))
    fade_out=audio[fade_seg_start:fade_seg_end]*mask
    subseq_audio=np.concatenate((fade_in,subseq_audio))
    subseq_audio=np.concatenate((subseq_audio,fade_out))
    return subseq_audio

def generate_snippet(audio_file,positioning,N=16):
    features,audio= process_features(audio_file)
    features["bs_sequenceMFCC"]=librosa.util.sync(features["sequenceMFCC"].T,
                                              features["beats_idx"],
                                              pad=False).T
    snippet_idxs=find_idxs(sequence=features["bs_sequenceMFCC"],positioning=positioning)
    #print(snippet_idxs)
    snippet=synth_snippet(audio,features["beats"],snippet_idxs,N)
    logging.info("Writting snippet for {} - MFCC".format(audio_file))
    wavfile.write("test/{}_Test_MFCC.wav".format(audio_file),SAMPLING_RATE,snippet)
    features["bs_sequenceChroma"]=librosa.util.sync(features["sequenceChroma"].T,
                                              features["beats_idx"],
                                              pad=False).T
    snippet_idxs=find_idxs_chroma(sequence=features["bs_sequenceChroma"])
    snippet=synth_snippet(audio,features["beats"],snippet_idxs,N)
    logging.info("Writting snippet for {} - Chroma".format(audio_file))
    wavfile.write("test/{}_Test_Chroma.wav".format(audio_file),SAMPLING_RATE,snippet)



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
    parser.add_argument("-P",
                        action="store",
                        dest="positioning",
                        help="Weight for positioning (beginning,middle,end)",
                        default="None")
    args=parser.parse_args()
    if(args.Debug):
        logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',level=logging.INFO)
    generate_snippet(args.audio_file,args.positioning)
