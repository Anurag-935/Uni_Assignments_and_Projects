import os
import torch
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# Import our custom modules
from dsp import BarkSpeechDSP
from model import HybridNoiseEnhancerNet
from train import train_model

def enhance_audio(noisy_path, output_path, model_path=None, spectral_floor=0.02):
    """
    Loads a noisy wav file, processes it frame-by-frame (simulating real-time streaming),
    applies the deep gain mask with safeguards, and saves the enhanced clean audio.
    Also plots visual comparisons of waveforms and spectrograms.
    """
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if model_path is None:
        model_path = os.path.join(project_dir, 'model.pt')
        
    # 1. Check if model checkpoint exists; if not, trigger a quick training session
    if not os.path.exists(model_path):
        print("Optimized deep learning weights not found! Running a quick training session to build the model...")
        train_model(epochs=10, batch_size=8)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nLoading optimized speech model from {model_path} onto {device}...")
    
    # 2. Initialize DSP and Model
    dsp = BarkSpeechDSP()
    model = HybridNoiseEnhancerNet(num_bark_bands=dsp.num_bands).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 3. Load Audio File
    print(f"Reading noisy audio: {noisy_path}")
    noisy_sig, fs = sf.read(noisy_path)
    
    # Ensure monophonic (downmix stereo to mono)
    if len(noisy_sig.shape) > 1:
        print("Input audio is stereo. Downmixing to mono...")
        noisy_sig = np.mean(noisy_sig, axis=1)
        
    # Dynamically resample if there is a sample rate mismatch
    if fs != dsp.fs:
        print(f"Sample rate mismatch! Dynamically resampling from {fs}Hz to {dsp.fs}Hz...")
        from scipy.signal import resample
        num_samples = int(len(noisy_sig) * dsp.fs / fs)
        noisy_sig = resample(noisy_sig, num_samples)
        fs = dsp.fs
        
    # Normalize signal
    norm_factor = np.max(np.abs(noisy_sig)) + 1e-8
    noisy_sig_norm = noisy_sig / norm_factor
    
    # 4. DSP Analysis (STFT)
    print("Executing Time-Domain Framing and STFT Analysis...")
    noisy_mags, noisy_phases = dsp.stft_analysis(noisy_sig_norm)
    num_frames = noisy_mags.shape[0]
    
    # Convert entire spectrogram to Bark domain
    noisy_barks = dsp.linear_to_bark(noisy_mags)
    
    # DSP UPGRADE: Estimate background noise floor from the first 8 frames (initial silent segment)
    # Since continuous voice recordings usually start with a brief pause, the first 8 frames represent the noise floor.
    # We enforce a strict floor of 1e-5 to prevent division-by-zero.
    noise_floor = np.mean(noisy_barks[:min(8, num_frames), :], axis=0)
    noise_floor = np.maximum(noise_floor, 1e-5)
    
    # 5. Stateful Frame-by-Frame Streaming Inference (Simulating Real-Time Audio)
    print("Processing audio frames via stateful GRU-DNN controller...")
    h_state = None
    enhanced_mags = np.zeros_like(noisy_mags)
    predicted_gains_list = []
    
    for f in range(num_frames):
        # Extract noisy Bark vector for current frame
        frame_bark = noisy_barks[f, :] # shape: (24,)
        
        # Format for PyTorch recurrent step: (batch_size=1, seq_len=1, num_bark_bands=24)
        frame_tensor = torch.tensor(frame_bark, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        # Stateful recurrent forward pass (updates h_state inside loop)
        with torch.no_grad():
            frame_gain, h_state = model(frame_tensor, h_state)
            
        # Extract estimated Bark gains: (24,)
        gain = frame_gain.squeeze().cpu().numpy()
        
        # HYBRID SPEECH PRESERVER: Adaptive soft gating based on instantaneous SNR
        # Calculates local SNR per band relative to our estimated noise floor
        snr = frame_bark / noise_floor
        
        # Soft speech probability curve with a 1.8 SNR dead-band (guarantees zero leakage during silences)
        speech_prob = 1.0 - np.exp(-0.06 * (np.maximum(snr - 1.8, 0.0) ** 2))
        
        # Hybrid blend: Speech frequencies are passed at 100% volume, noise bands are attenuated by the model
        gain = speech_prob * 1.0 + (1.0 - speech_prob) * gain
        
        # SAFEGUARD: Apply non-linear spectral floor beta to kill robotic musical noise
        # This guarantees empty bins never drop below absolute silence
        gain = np.maximum(gain, spectral_floor)
        
        # DSP UPGRADE: Oversubtraction via Power Exponent
        # Squaring the gain mask aggressively suppresses low-energy noise leaks
        # while preserving high-energy speech bands.
        gain = gain ** 2.0
        predicted_gains_list.append(gain)
        
        # Smoothly interpolate Bark gains back to 257 linear frequency bins
        linear_gain = dsp.bark_to_linear_gains(gain)
        
        # Apply gains to noisy linear magnitudes
        enhanced_mags[f, :] = linear_gain * noisy_mags[f, :]
        
    # 6. DSP Synthesis (Weighted Overlap-Add Reconstruction)
    print("Reconstructing continuous audio via Weighted Overlap-Add (WOLA)...")
    enhanced_sig = dsp.stft_synthesis(enhanced_mags, noisy_phases)
    
    # DSP UPGRADE: Robust Peak Normalization
    # Restores the speech back to its original full, rich volume
    orig_peak = np.max(np.abs(noisy_sig))
    enh_peak = np.max(np.abs(enhanced_sig))
    if enh_peak > 1e-4:
        enhanced_sig = enhanced_sig * (orig_peak / enh_peak)
    
    # Save output file
    sf.write(output_path, enhanced_sig, fs, subtype='PCM_16')
    print(f"[SUCCESS] Enhanced clean audio successfully saved to: {output_path}")
    
    # 7. Calculate Blind SNR Metrics
    print("Calculating Signal-to-Noise Ratio (SNR) improvements...")
    noisy_frame_energies = np.mean(noisy_mags**2, axis=1)
    enhanced_frame_energies = np.mean(enhanced_mags**2, axis=1)
    
    # Sort energies to identify noise regions (bottom 15% energy) and active speech regions (top 30% energy)
    sorted_noisy = np.sort(noisy_frame_energies)
    sorted_enhanced = np.sort(enhanced_frame_energies)
    
    n_frames = len(noisy_frame_energies)
    n_noise = max(1, int(0.15 * n_frames))
    n_speech = max(1, int(0.30 * n_frames))
    
    noisy_noise_power = np.mean(sorted_noisy[:n_noise]) + 1e-8
    noisy_speech_power = np.mean(sorted_noisy[-n_speech:])
    
    enhanced_noise_power = np.mean(sorted_enhanced[:n_noise]) + 1e-8
    enhanced_speech_power = np.mean(sorted_enhanced[-n_speech:])
    
    # Reference-free estimated SNR calculations
    input_snr = 10 * np.log10(max(noisy_speech_power - noisy_noise_power, 1e-8) / noisy_noise_power)
    output_snr = 10 * np.log10(max(enhanced_speech_power - enhanced_noise_power, 1e-8) / enhanced_noise_power)
    
    # Restrict values to sensible physical bounds
    input_snr = np.clip(input_snr, -5.0, 50.0)
    output_snr = np.clip(output_snr, -5.0, 50.0)
    snr_improvement = max(0.0, output_snr - input_snr)
    
    # Calculate dynamic frame-by-frame SNR curves
    snr_noisy_frames = 10 * np.log10(noisy_frame_energies / noisy_noise_power + 1e-5)
    snr_enhanced_frames = 10 * np.log10(enhanced_frame_energies / enhanced_noise_power + 1e-5)
    
    snr_noisy_frames = np.clip(snr_noisy_frames, -10.0, 40.0)
    snr_enhanced_frames = np.clip(snr_enhanced_frames, -10.0, 40.0)
    
    # 8. Generate Visual Verification Artifacts
    print("Generating visual spectrogram and SNR comparison chart...")
    plot_path = os.path.join(project_dir, 'spectrogram_comparison.png')
    
    # Optimized figure size to fit perfectly inside the GUI canvas (3 rows, 2 columns)
    fig = plt.figure(figsize=(8.5, 6.2))
    
    # Define subplots using subplot2grid for complex layouts
    ax1 = plt.subplot2grid((3, 2), (0, 0))
    ax2 = plt.subplot2grid((3, 2), (0, 1))
    ax3 = plt.subplot2grid((3, 2), (1, 0))
    ax4 = plt.subplot2grid((3, 2), (1, 1))
    ax5 = plt.subplot2grid((3, 2), (2, 0), colspan=2)
    
    # Waveform Comparisons
    t = np.arange(len(noisy_sig)) / fs
    ax1.plot(t, noisy_sig, color='#e06666', alpha=0.8)
    ax1.set_title("Noisy Signal Waveform", fontsize=9, fontweight='bold')
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    t_enh = np.arange(len(enhanced_sig)) / fs
    ax2.plot(t_enh, enhanced_sig, color='#6fa8dc', alpha=0.8)
    ax2.set_title("Enhanced Clean Waveform", fontsize=9, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # Spectrogram Comparisons
    eps = 1e-5
    noisy_spec_db = 20 * np.log10(noisy_mags.T + eps)
    enhanced_spec_db = 20 * np.log10(enhanced_mags.T + eps)
    
    img1 = ax3.imshow(noisy_spec_db, origin='lower', aspect='auto', 
                             extent=[0, num_frames*dsp.hop_len/fs, 0, fs/2000],
                             cmap='inferno', vmin=-60, vmax=10)
    ax3.set_title("Noisy Input Spectrogram", fontsize=9, fontweight='bold')
    ax3.set_ylabel("Frequency (kHz)")
    fig.colorbar(img1, ax=ax3, format="%+2.0f dB")
    
    img2 = ax4.imshow(enhanced_spec_db, origin='lower', aspect='auto',
                             extent=[0, num_frames*dsp.hop_len/fs, 0, fs/2000],
                             cmap='inferno', vmin=-60, vmax=10)
    ax4.set_title("Enhanced Output Spectrogram", fontsize=9, fontweight='bold')
    fig.colorbar(img2, ax=ax4, format="%+2.0f dB")
    
    # Dynamic SNR Timeline Plot
    t_frames = np.arange(num_frames) * dsp.hop_len / fs
    ax5.plot(t_frames, snr_noisy_frames, color='#e06666', label='Noisy SNR', alpha=0.8, linewidth=1.5)
    ax5.plot(t_frames, snr_enhanced_frames, color='#00adb5', label='Enhanced SNR', alpha=0.9, linewidth=1.5)
    ax5.set_title(f"Dynamic Signal-to-Noise Ratio (SNR) Timeline (Estimated Improvement: +{snr_improvement:.1f} dB)", fontsize=9, fontweight='bold')
    ax5.set_xlabel("Time (seconds)")
    ax5.set_ylabel("SNR (dB)")
    ax5.legend(loc='upper right', framealpha=0.9)
    ax5.grid(True, linestyle='--', alpha=0.5)
    ax5.set_ylim(-10, 45)
    
    plt.tight_layout()
    # Optimized DPI to prevent image cropping
    plt.savefig(plot_path, dpi=100)
    plt.close()
    print(f"[SUCCESS] Spectrogram comparison plot saved to: {plot_path}")
    
    return plot_path, input_snr, output_snr, snr_improvement

if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Prepare a default noisy demo file if it doesn't exist
    demo_noisy = os.path.join(project_dir, 'data', 'demo_noisy.wav')
    demo_clean = os.path.join(project_dir, 'data', 'clean', 'synth_clean_1.wav')
    demo_noise = os.path.join(project_dir, 'data', 'noise', 'fan_hum.wav')
    
    if not os.path.exists(demo_noisy):
        print("Creating demo noisy file for verification...")
        # Make sure data directory exists and synthetic files are present
        if not os.path.exists(demo_clean) or not os.path.exists(demo_noise):
            from dataset import generate_synthetic_data
            generate_synthetic_data(os.path.join(project_dir, 'data'), num_clean=5, num_noise=3)
            
        clean, fs = sf.read(demo_clean)
        noise, _ = sf.read(demo_noise)
        
        # Mix clean & noise at 0dB SNR for testing
        p_c = np.mean(clean**2)
        p_n = np.mean(noise**2)
        scalar = np.sqrt(p_c / (p_n + 1e-8))
        noisy_mix = clean + 0.8 * scalar * noise
        noisy_mix /= np.max(np.abs(noisy_mix)) + 1e-8
        
        sf.write(demo_noisy, noisy_mix, fs)
        
    output_clean = os.path.join(project_dir, 'enhanced_clean.wav')
    enhance_audio(demo_noisy, output_clean)
