import os
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

class SpeechEnhancementDataset(Dataset):
    """
    Speech Enhancement Dataset that mixes clean speech files with noise files
    on-the-fly at random SNRs, then uses BarkSpeechDSP to extract noisy Bark bands
    and the Ideal Ratio Mask (IRM) in the Bark domain.
    """
    def __init__(self, clean_files, noise_files, dsp, seq_len=64, snr_range=(-5, 15)):
        self.clean_files = clean_files
        self.noise_files = noise_files
        self.dsp = dsp
        self.seq_len = seq_len
        self.snr_range = snr_range

    def __len__(self):
        # We can simulate a large epoch size by repeating files
        return max(len(self.clean_files), 100)

    def _mix_signals(self, clean, noise, snr_db):
        """Mixes clean speech and noise at a specified Signal-to-Noise Ratio (SNR)."""
        # Align lengths
        if len(noise) < len(clean):
            # Tile noise if it is too short
            repeats = int(np.ceil(len(clean) / len(noise)))
            noise = np.tile(noise, repeats)
        noise = noise[:len(clean)]
        
        # Calculate signal powers
        p_clean = np.mean(clean ** 2) + 1e-8
        p_noise = np.mean(noise ** 2) + 1e-8
        
        # Calculate required noise scaling factor
        # SNR_dB = 10 * log10(P_clean / (scalar^2 * P_noise))
        # P_clean / (scalar^2 * P_noise) = 10^(SNR/10)
        # scalar = sqrt(P_clean / (P_noise * 10^(SNR/10)))
        scalar = np.sqrt(p_clean / (p_noise * (10 ** (snr_db / 10.0))))
        
        noisy = clean + scalar * noise
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(noisy))
        if max_val > 1.0:
            clean = clean / max_val
            noisy = noisy / max_val
            
        return clean, noisy

    def __getitem__(self, idx):
        # Pick random clean and noise files
        clean_path = self.clean_files[idx % len(self.clean_files)]
        noise_path = self.noise_files[np.random.randint(len(self.noise_files))]
        
        # Load audio (downsample/resample if needed, assuming they are 16kHz)
        clean, fs_c = sf.read(clean_path)
        noise, fs_n = sf.read(noise_path)
        
        # Ensure mono
        if len(clean.shape) > 1: clean = np.mean(clean, axis=1)
        if len(noise.shape) > 1: noise = np.mean(noise, axis=1)
        
        # Select a random SNR
        snr = np.random.uniform(self.snr_range[0], self.snr_range[1])
        
        # Mix audio
        clean, noisy = self._mix_signals(clean, noise, snr)
        
        # Extract STFT magnitudes
        clean_mags, _ = self.dsp.stft_analysis(clean)
        noisy_mags, _ = self.dsp.stft_analysis(noisy)
        
        # Map to Bark bands
        clean_bark = self.dsp.linear_to_bark(clean_mags)
        noisy_bark = self.dsp.linear_to_bark(noisy_mags)
        
        # Compute target Ideal Ratio Mask (IRM) in Bark domain: G_b = clean_b / noisy_b
        # Clamped strictly between 0.0 and 1.0
        target_gains = np.clip(clean_bark / (noisy_bark + 1e-8), 0.0, 1.0)
        
        # Extract a random sub-sequence of length `seq_len` for training
        num_frames = noisy_bark.shape[0]
        if num_frames > self.seq_len:
            start_f = np.random.randint(0, num_frames - self.seq_len)
            noisy_bark = noisy_bark[start_f:start_f + self.seq_len, :]
            target_gains = target_gains[start_f:start_f + self.seq_len, :]
        else:
            # Pad sequence if too short
            pad_w = self.seq_len - num_frames
            noisy_bark = np.pad(noisy_bark, ((0, pad_w), (0, 0)), mode='edge')
            target_gains = np.pad(target_gains, ((0, pad_w), (0, 0)), mode='edge')
            
        return torch.tensor(noisy_bark, dtype=torch.float32), torch.tensor(target_gains, dtype=torch.float32)


def generate_synthetic_data(data_dir, num_clean=5, num_noise=3, fs=16000, duration=4.0):
    """
    Generates synthetic WAV files for test and offline development.
    mimics voice formants (clean speech) and background hums (noise).
    """
    os.makedirs(os.path.join(data_dir, 'clean'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'noise'), exist_ok=True)
    
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    # 1. Generate clean files (sweeping harmonics, simulating speech formants)
    clean_paths = []
    for idx in range(num_clean):
        # Synthesize sweeping fundamental frequency (F0) around 100-300Hz (vocal range)
        f0_start = np.random.uniform(120, 180)
        f0_end = np.random.uniform(180, 260)
        f0 = np.linspace(f0_start, f0_end, len(t))
        phase = 2 * np.pi * np.cumsum(f0) / fs
        
        # Harmonic structure
        wave = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.25 * np.sin(3 * phase)
        
        # Add vocal envelope modulation (simulate words/syllables)
        envelope = 0.5 * (1.0 + np.sin(2 * np.pi * np.random.uniform(1.0, 3.0) * t))
        wave *= envelope
        
        # Add high-frequency sibilant bursts (simulating unvoiced phonemes like 's' or 'f')
        bursts = np.zeros_like(t)
        for _ in range(3):
            b_start = np.random.randint(int(0.2*len(t)), int(0.8*len(t)))
            b_len = int(fs * 0.15)
            # High-pass filtered noise
            noise = np.random.normal(0, 0.05, b_len)
            bursts[b_start:b_start+b_len] += noise
            
        wave += bursts
        
        # Normalize and save
        wave /= np.max(np.abs(wave)) + 1e-6
        path = os.path.join(data_dir, 'clean', f'synth_clean_{idx+1}.wav')
        sf.write(path, wave, fs)
        clean_paths.append(path)
        
    # 2. Generate noise files (hum, rumble, white hiss)
    noise_paths = []
    
    # Noise A: Mechanical fan/hum (50Hz hum + white hiss)
    hum = 0.3 * np.sin(2 * np.pi * 50.0 * t) + np.random.normal(0, 0.1, len(t))
    hum /= np.max(np.abs(hum)) + 1e-6
    path = os.path.join(data_dir, 'noise', 'fan_hum.wav')
    sf.write(path, hum, fs)
    noise_paths.append(path)
    
    # Noise B: White static hiss
    hiss = np.random.normal(0, 0.2, len(t))
    hiss /= np.max(np.abs(hiss)) + 1e-6
    path = os.path.join(data_dir, 'noise', 'white_hiss.wav')
    sf.write(path, hiss, fs)
    noise_paths.append(path)
    
    # Noise C: Low frequency brown rumble (e.g. AC rumble)
    rumble = np.random.normal(0, 0.5, len(t))
    # Simple low pass integration to make brown noise
    rumble = np.cumsum(rumble)
    rumble -= np.mean(rumble)
    rumble /= np.max(np.abs(rumble)) + 1e-6
    path = os.path.join(data_dir, 'noise', 'ac_rumble.wav')
    sf.write(path, rumble, fs)
    noise_paths.append(path)
    
    print(f"Generated {num_clean} clean and {num_noise} noise synthetic files.")
    return clean_paths, noise_paths

if __name__ == "__main__":
    # Test generation and dataset
    from dsp import BarkSpeechDSP
    dsp = BarkSpeechDSP()
    c, n = generate_synthetic_data('./test_data')
    ds = SpeechEnhancementDataset(c, n, dsp)
    x, y = ds[0]
    print(f"Dataset Test passed. Sequence X shape: {x.shape}, Target Y shape: {y.shape}")
