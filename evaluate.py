#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 18:36:04 2024

@author: jrnmapanao

model evaluation
"""

import faiss
import os
import pickle
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import librosa
import numpy as np
import faiss
import os
import pickle
import soundfile as sf


def convert_to_log_mel_spectrogram(y, sr=22050, n_mels=256, stft_win=1024, stft_hop=256):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=stft_hop, win_length=stft_win)
    log_S = librosa.power_to_db(S, ref=np.max)
    return log_S

def extract_feature_vector(log_mel_spectrogram, model):
    log_mel_spectrogram = torch.tensor(log_mel_spectrogram).unsqueeze(0).unsqueeze(0).float().cuda()
    with torch.no_grad():
        feature_vector = model(log_mel_spectrogram).cpu().numpy().flatten()
    return feature_vector
    

def generate_overlapping_segments(audio, sr, segment_duration=1, hop_duration=0.5):
    segment_length = int(segment_duration * sr)
    hop_length = int(hop_duration * sr)
    segments = []

    for start in range(0, len(audio) - segment_length + 1, hop_length):
        end = start + segment_length
        segments.append(audio[start:end])

    return segments


def evaluate_with_overlapping_segments(model, test_dirs, index, file_names, top_k=[1, 5, 10], sr=22050):
    segment_accuracy = {duration: {k: 0 for k in top_k} for duration in test_dirs.keys()}
    total_segments = {duration: 0 for duration in test_dirs.keys()}

    for duration, test_dir in test_dirs.items():
        print(f"Evaluating segments of duration: {duration}s")
        test_files = [f for f in os.listdir(test_dir) if f.endswith('.wav')]

        for file in test_files:
            total_segments[duration] += 1
            file_path = os.path.join(test_dir, file)
            y, sr = librosa.load(file_path, sr=sr)

            # Generate overlapping 1-second segments
            overlapping_segments = generate_overlapping_segments(y, sr, segment_duration=1, hop_duration=0.5)
            feature_vectors = []


            # Extract feature vectors for each overlapping segment
            for segment in overlapping_segments:
                log_mel_spectrogram = convert_to_log_mel_spectrogram(segment, sr=sr, n_mels=256, stft_win=1024, stft_hop=256)
                feature_vector = extract_feature_vector(log_mel_spectrogram, model)
                feature_vectors.append(feature_vector)

            # Average the feature vectors
            avg_feature_vector = np.mean(feature_vectors, axis=0)

            # Search the FAISS index using the averaged feature vector
            D, I = index.search(np.array([avg_feature_vector]).astype('float32'), max(top_k))

            file_name = file.split('.')[0]

            # Retrieve the corresponding filenames
            retrieved_files = ([file_name.split('.')[0].split('_')[0]] if file_name.split('.')[0].split('_')[0] not in [file_names[idx].split('.')[0].split('_')[0] for idx in I[0]] else []) + [file_names[idx].split('.')[0].split('_')[0] for idx in I[0]]

            print(f"Results for file: {file.split('.')[0]}")
            for k in top_k:
                top_files = retrieved_files[:k]
                print(f"    Top {k} results: {top_files}")
                if file.split('.')[0] in top_files:
                    segment_accuracy[duration][k] += 1

        print(f"\nAccuracy for {duration}s segments:")
        for k in top_k:
            accuracy = segment_accuracy[duration][k] / total_segments[duration]
            print(f"  Top {k} accuracy: {accuracy * 100:.2f}%")