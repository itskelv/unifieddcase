import numpy as np
import librosa
import os

def stereo_to_foa_features(file_path, n_mels=64, hop_length=256, sr=16000):
    # Load stereo audio
    y, _ = librosa.load(file_path, sr=sr, mono=False)
    
    if y.ndim != 2 or y.shape[0] != 2:
        raise ValueError("Input must be stereo audio with shape (2, n_samples)")
    
    # Left and right channels
    L = y[0]
    R = y[1]
    
    # Compute mel-spectrogram for L and R
    S_L = librosa.feature.melspectrogram(L, sr=sr, n_mels=n_mels, hop_length=hop_length)
    S_R = librosa.feature.melspectrogram(R, sr=sr, n_mels=n_mels, hop_length=hop_length)
    
    # Compute sum and difference (pseudo-W/X channels)
    W = (S_L + S_R) / 2      # pseudo omni
    X = S_L - S_R            # pseudo horizontal
    Y = np.zeros_like(W)     # placeholder vertical channel
    Z = np.zeros_like(W)     # placeholder vertical channel
    
    # Intensity vectors (L+R, L-R, etc.) as extra channels
    I1 = W
    I2 = X
    I3 = Y
    
    # 4. Stack into 7-channel feature (time x n_mels x channels)
    features = np.stack([W, X, Y, Z, I1, I2, I3], axis=0)
    
    return features  # shape: (7, n_mels, n_frames)

input_folder = "../DCASE2025_SELD_dataset/stereo_dev"
output_folder = "../DCASE_FOA_DATASET_2024/stereo_dev"
os.makedirs(output_folder, exist_ok=True)

for file_name in os.listdir(input_folder):
    if file_name.endswith(".wav"):
        file_path = os.path.join(input_folder, file_name)
        feats = stereo_to_foa_features(file_path)
        np.save(os.path.join(output_folder, file_name.replace(".wav", ".npy")), feats)

input_folder = "../DCASE2025_SELD_dataset/stereo_eval"
output_folder = "../DCASE_FOA_DATASET_2024/stereo_eval"
os.makedirs(output_folder, exist_ok=True)

for file_name in os.listdir(input_folder):
    if file_name.endswith(".wav"):
        file_path = os.path.join(input_folder, file_name)
        feats = stereo_to_foa_features(file_path)
        np.save(os.path.join(output_folder, file_name.replace(".wav", ".npy")), feats)
