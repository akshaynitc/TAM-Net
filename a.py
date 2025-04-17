import os
import numpy as np
import cv2
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline
import mne
from mne.preprocessing import ICA
from mne.io import RawArray
from mne import create_info
import dlib
import torch
from torch import nn

# -----------------------------
# EEG Blink Detection
# -----------------------------
def detect_eeg_blinks(eeg_data, sampling_rate=128, channel_names=None):
    """
    Detect eyeblink timestamps in EEG data using band‑pass filtering, ICA, and peak detection.
    
    eeg_data: np.ndarray, shape (n_channels, n_samples)
    sampling_rate: int, e.g. 128 Hz
    channel_names: list of str, length n_channels
    """
    n_channels, n_samples = eeg_data.shape
    if channel_names is None:
        # generate generic channel names
        channel_names = [f"EEG{i}" for i in range(n_channels)]
    
    # Wrap in an MNE RawArray
    info = create_info(ch_names=channel_names, sfreq=sampling_rate, ch_types='eeg')
    raw = RawArray(eeg_data, info)
    
    # 1) Band‑pass filter to retain 0.5–45 Hz
    raw_filtered = raw.copy().filter(l_freq=0.5, h_freq=45, verbose=False)
    
    # 2) ICA to isolate blink components
    ica = ICA(n_components=min(n_channels, 20), random_state=42, verbose=False)
    ica.fit(raw_filtered)
    
    # 3) Automatically find blink (EOG) components
    eog_inds, scores = ica.find_bads_eog(raw_filtered, threshold=3.0)
    if not eog_inds:
        # fallback: assume first component
        eog_inds = [0]
    sources = ica.get_sources(raw_filtered).get_data()
    blink_comp = sources[eog_inds[0], :]
    
    # 4) Band‑limit blink component to delta band (0.1–4 Hz)
    blink_delta = mne.filter.filter_data(
        blink_comp[np.newaxis, :], sfreq=sampling_rate,
        l_freq=0.1, h_freq=4.0, verbose=False
    ).squeeze()
    
    # 5) Peak detection on absolute delta‑band signal
    threshold = np.mean(np.abs(blink_delta)) + 2 * np.std(np.abs(blink_delta))
    peaks, _ = find_peaks(np.abs(blink_delta), height=threshold, distance=sampling_rate//5)
    
    # Convert sample indices to time (s)
    return peaks / sampling_rate


# -----------------------------
# Video Blink Detection
# -----------------------------
class TCN(nn.Module):
    """Temporal Convolutional Network for blink‑probability estimation."""
    def __init__(self, input_size=1, hidden_size=64, num_layers=3):
        super().__init__()
        layers = []
        in_ch = input_size
        for _ in range(num_layers):
            conv = nn.Conv1d(in_ch, hidden_size, kernel_size=3, padding=1)
            layers.append(conv)
            in_ch = hidden_size
        self.net = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: (batch=1, channels=1, seq_len)
        h = self.net(x)
        h = h.permute(0, 2, 1)              # (1, seq_len, hidden)
        out = self.fc(h).squeeze(-1)        # (1, seq_len)
        return self.sigmoid(out)           # blink probability per frame


def detect_video_blinks(video_path, ear_threshold=0.25, tcn_weights_path=None):
    """
    Detect eyeblink frame timestamps in a video via EAR and optional TCN model.
    
    video_path: str, path to .mp4/.avi file
    ear_threshold: float, EAR below this indicates blink (fallback)
    tcn_weights_path: str or None, path to pretrained TCN weights
    """
    # Initialize Dlib face detector & shape predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ear_values = []
    timestamps = []
    
    # 1) Extract EAR sequence
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        if faces:
            shape = predictor(gray, faces[0])
            pts = np.array([[p.x, p.y] for p in shape.parts()[36:48]])
            # EAR = (||p1–p5|| + ||p2–p4||) / (2 * ||p0–p3||)
            A = np.linalg.norm(pts[1] - pts[5])
            B = np.linalg.norm(pts[2] - pts[4])
            C = np.linalg.norm(pts[0] - pts[3])
            ear = (A + B) / (2.0 * C)
        else:
            ear = np.nan
        ear_values.append(ear)
        timestamps.append(t)
    cap.release()
    
    ear_values = np.array(ear_values, dtype=np.float32)
    timestamps = np.array(timestamps, dtype=np.float32)
    
    # 2) Optional TCN‑based detection
    use_tcn = False
    if tcn_weights_path and os.path.exists(tcn_weights_path):
        tcn = TCN()
        tcn.load_state_dict(torch.load(tcn_weights_path, map_location="cpu"))
        tcn.eval()
        seq = torch.tensor(np.nan_to_num(ear_values), dtype=torch.float32)
        seq = seq.unsqueeze(0).unsqueeze(0)  # (1,1,seq_len)
        with torch.no_grad():
            blink_prob = tcn(seq).squeeze(0).numpy()
        blink_idx = np.where(blink_prob < 0.5)[0]
        use_tcn = True
    # 3) Fallback: simple EAR thresholding
    if not use_tcn:
        blink_idx = np.where(ear_values < ear_threshold)[0]
    
    return timestamps[blink_idx]


# -----------------------------
# EAM Synchronization
# -----------------------------
def eam_synchronization(eeg_data, video_path,
                        eeg_sampling_rate=128, video_frame_rate=30,
                        ear_threshold=0.25, tcn_weights_path=None):
    """
    Perform eyeblink‑anchored synchronization of EEG & video data.
    
    Returns:
        eeg_data (unchanged),
        synchronized_video_timestamps (np.ndarray of length n_frames),
        time_offset_delta (float, seconds)
    """
    # 1) Detect blinks in each modality
    eeg_blinks = detect_eeg_blinks(eeg_data, sampling_rate=eeg_sampling_rate)
    video_blinks = detect_video_blinks(video_path,
                                       ear_threshold=ear_threshold,
                                       tcn_weights_path=tcn_weights_path)
    
    # 2) AND‑based coarse alignment over range [-0.5, +0.5] s
    def compute_and_overlap(tau):
        # bin size = 1/video_frame_rate (for video) and 1/eeg_sampling_rate
        max_time = max(eeg_blinks.max(), video_blinks.max()) + 1.0
        bins_eeg = np.zeros(int(max_time * eeg_sampling_rate) + 1, bool)
        bins_vid = np.zeros(int(max_time * video_frame_rate) + 1, bool)
        for t in eeg_blinks:
            bins_eeg[int(t * eeg_sampling_rate)] = True
        for t in video_blinks:
            bins_vid[int(t * video_frame_rate)] = True
        shift_eeg = int(tau * eeg_sampling_rate)
        if shift_eeg > 0:
            overlap = np.logical_and(bins_eeg[shift_eeg:], bins_vid[:len(bins_eeg)-shift_eeg])
        else:
            shift_v = -shift_eeg
            overlap = np.logical_and(bins_eeg[:len(bins_eeg)-shift_v], bins_vid[shift_v:])
        return overlap.sum()
    
    taus = np.linspace(-0.5, 0.5, 101)
    overlaps = [compute_and_overlap(t) for t in taus]
    tau_and = taus[np.argmax(overlaps)]
    
    # 3) (Optional) DTW refinement could be inserted here
    
    # 4) Combine offsets (here we just use the AND-based result)
    delta = tau_and
    
    # 5) Build full video timestamp array
    cap2 = cv2.VideoCapture(video_path)
    frame_count = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    cap2.release()
    video_times = np.arange(frame_count) / video_frame_rate
    
    # 6) Interpolate non‑blink segments via spline
    # Ensure at least two blink points
    if len(video_blinks) >= 2:
        sorted_idx = np.argsort(video_blinks)
        vb = video_blinks[sorted_idx]
        eb = video_blinks[sorted_idx] + delta
        spline = CubicSpline(vb, eb, extrapolate=True)
        synced_times = spline(video_times)
    else:
        # fallback: simple shift
        synced_times = video_times + delta
    
    return eeg_data, synced_times, delta


# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    # Simulate 32‑channel EEG for 10 seconds @128 Hz
    eeg_sampling_rate = 128
    duration = 10  # seconds
    n_samples = eeg_sampling_rate * duration
    eeg_data = np.random.normal(0, 1, (32, n_samples))
    
    video_path = "example_video.mp4"
    # If you have a trained TCN, specify its weights path; else leave as None
    tcn_weights = None  # e.g. "tcn_blink_model.pth"
    
    synced_eeg, synced_video_timestamps, offset = eam_synchronization(
        eeg_data, video_path,
        eeg_sampling_rate=eeg_sampling_rate,
        video_frame_rate=30,
        ear_threshold=0.25,
        tcn_weights_path=tcn_weights
    )
    
    print(f"EAM synchronization complete. Time offset: {offset:.3f} s")
    print(f"First 5 synchronized video timestamps: {synced_video_timestamps[:5]}")
