import numpy as np
import librosa
import os, json
import collections


def extract_mfcc(y, sr, size=1):
    """
    extract MFCC feature
    :param y: np.ndarray [shape=(n,)], real-valued the input signal (audio time series)
    :param sr: sample rate of 'y'
    :param size: the length (seconds) of random crop from original audio, default as 3 seconds
    :return: MFCC feature
    """
    # normalization
    y = y.astype(np.float32)
    normalization_factor = 1 / np.max(np.abs(y))
    y = y * normalization_factor

    # random crop
    # start = random.randint(0, len(y) - size * sr)
    # y = y[start: start + size * sr]

    # extract log mel spectrogram #####
    melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(melspectrogram), n_mfcc=20)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta_delta = librosa.feature.delta(mfcc_delta)
    mfcc_comb = np.concatenate([mfcc, mfcc_delta, mfcc_delta_delta], axis=0)

    return mfcc_comb



def extract_logmel(y, sr, size=1):
    """
    extract log mel spectrogram feature
    :param y: the input signal (audio time series)
    :param sr: sample rate of 'y'
    :param size: the length (seconds) of random crop from original audio, default as 3 seconds
    :return: log-mel spectrogram feature
    """
    # normalization
    y = y.astype(np.float32)
    normalization_factor = 1 / np.max(np.abs(y))
    y = y * normalization_factor

    # random crop
    # start = random.randint(0, len(y) - size * sr)
    # y = y[start: start + size * sr]

    # extract log mel spectrogram #####
    melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024, n_mels=60)
    logmelspec = librosa.power_to_db(melspectrogram)

    return logmelspec



def get_feature():
    wav_path = '../clips_rd_sox'
    save_path = '../clips_rd_mfcc'
    i = 0
    for root, dirs, files in os.walk(wav_path):
        if files:
            for file in files:
                f_path = os.path.join(root, file)
                try:
                    y, sr = librosa.load(f_path, sr=None)
                    mfcc_feat = extract_mfcc(y, sr, size=3)
                    mfcc_feat = mfcc_feat.transpose(1, 0)
                    np.save('../clips_rd_mfcc/{}.npy'.format(file.split('.')[0]), mfcc_feat)
                    print('the {} is done'.format(str(i)))
                    i += 1
                except:
                    print(file)
get_feature()

