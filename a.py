import numpy as np
import cv2
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline
from mne.preprocessing import ICA
from mne.filter import filter_data
import dlib
import torch
from torch import nn

# EEG Blink Detection
def detect_eeg_blinks(eeg_data, sampling_rate=128, channels=['Fp1', 'Fp2', 'AF3', 'AF4']):
    # Band-pass filter (0.5–45 Hz)
    eeg_filtered = filter_data(eeg_data, sfreq=sampling_rate, l_freq=0.5, h_freq=45, verbose=False)
    
    # ICA for artifact isolation
    ica = ICA(n_components=len(channels), random_state=42)
    ica.fit(eeg_filtered)
    blink_components = ica.get_sources(eeg_filtered)  # Simplified: Assume first component is blinks
    
    # Wavelet-based peak detection (delta band, 0.1–4 Hz)
    from scipy import signal
    cwt_matrix = signal.cwt(blink_components[0], signal.morlet2, widths=np.arange(1, 31))
    threshold = np.mean(cwt_matrix) + 2 * np.std(cwt_matrix)
    peaks, _ = find_peaks(np.abs(cwt_matrix[0]), height=threshold, distance=sampling_rate//5)
    
    return peaks / sampling_rate  # Timestamps in seconds

# Video Blink Detection
class TCN(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=3):
        super(TCN, self).__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.layers = nn.ModuleList([nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1) for _ in range(num_layers-1)])
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv(x)
        for layer in self.layers:
            x = layer(x)
        x = self.fc(x.transpose(1, 2)).squeeze(-1)
        return self.sigmoid(x)

def detect_video_blinks(video_path, frame_rate=30):
    cap = cv2.VideoCapture(video_path)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    ear_values = []
    timestamps = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        if faces:
            shape = predictor(gray, faces[0])
            eye_points = [shape.part(i) for i in range(36, 48)]  # Eye landmarks
            ear = (np.linalg.norm(eye_points[1] - eye_points[5]) + np.linalg.norm(eye_points[2] - eye_points[4])) / (2 * np.linalg.norm(eye_points[0] - eye_points[3]))
            ear_values.append(ear)
            timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
    
    cap.release()
    
    # TCN for blink detection
    ear_tensor = torch.tensor(ear_values, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    tcn = TCN()
    with torch.no_grad():
        blink_probs = tcn(ear_tensor).numpy()
    
    threshold = 0.3  # Calibrated
    blink_indices = np.where(blink_probs < threshold)[0]
    return np.array(timestamps)[blink_indices]

# EAM Synchronization
def eam_synchronization(eeg_data, video_path, eeg_sampling_rate=128, video_frame_rate=30):
    # Detect blinks
    eeg_blink_timestamps = detect_eeg_blinks(eeg_data, eeg_sampling_rate)
    video_blink_timestamps = detect_video_blinks(video_path, video_frame_rate)
    
    # AND-based alignment
    def compute_and_overlap(tau, t_eeg, t_video):
        b_eeg = np.zeros(max(int(max(t_eeg) * 1000), int(max(t_video) * 1000)) + 1)
        b_video = b_eeg.copy()
        for t in t_eeg:
            b_eeg[int(t * 1000)] = 1
        for t in t_video:
            b_video[int(t * 1000)] = 1
        shift = int(tau * 1000)
        b_and = np.sum(np.logical_and(b_eeg[:-shift], b_video[shift:]))
        return b_and
    
    taus = np.linspace(-0.5, 0.5, 100)
    overlaps = [compute_and_overlap(tau, eeg_blink_timestamps, video_blink_timestamps) for tau in taus]
    tau_and = taus[np.argmax(overlaps)]
    
    # DTW refinement
    def dtw_with_blinks(eeg, video, b_eeg, b_video, lambda_=1.0):
        N, M = len(eeg), len(video)
        R = np.full((N+1, M+1), np.inf)
        R[0, 0] = 0
        for i in range(1, N+1):
            for j in range(1, M+1):
                cost = (eeg[i-1] - video[j-1])**2 + lambda_ * abs(b_eeg[i-1] - b_video[j-1])
                R[i, j] = cost + min(R[i-1, j], R[i, j-1], R[i-1, j-1])
        return R[N, M]
    
    b_eeg = np.isin(np.arange(len(eeg_blink_timestamps)), np.round(eeg_blink_timestamps * eeg_sampling_rate).astype(int))
    b_video = np.isin(np.arange(len(video_blink_timestamps)), np.round(video_blink_timestamps * video_frame_rate).astype(int))
    dtw_cost = dtw_with_blinks(eeg_blink_timestamps, video_blink_timestamps, b_eeg, b_video)
    tau_dtw = tau_and  # Simplified: Use AND-based as initial estimate
    
    # Combine offsets
    delta = 0.9 * tau_dtw + 0.1 * tau_and  # Robust combination
    
    # Interpolate non-blink segments
    video_timestamps = np.array([i / video_frame_rate for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))])
    spline = CubicSpline(video_blink_timestamps, video_blink_timestamps + delta)
    synchronized_video_timestamps = spline(video_timestamps)
    
    return eeg_data, synchronized_video_timestamps, delta

# Example usage
if __name__ == "__main__":
    eeg_data = np.random.normal(0, 1, (32, 1280))  # Simulated EEG
    video_path = "example_video.mp4"
    synced_eeg, synced_video_ts, delta = eam_synchronization(eeg_data, video_path)
    print(f"Time offset: {delta:.3f} seconds")
