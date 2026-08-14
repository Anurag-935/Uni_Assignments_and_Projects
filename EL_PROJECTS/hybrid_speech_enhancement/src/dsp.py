import numpy as np
from scipy.signal import get_window

class BarkSpeechDSP:
    def __init__(self, sample_rate=16000, frame_size_ms=30, overlap=0.5, n_fft=512, num_bark_bands=24):
        self.fs = sample_rate
        self.frame_len = int(sample_rate * frame_size_ms / 1000)
        self.hop_len = int(self.frame_len * (1.0 - overlap))
        self.n_fft = n_fft
        self.num_bands = num_bark_bands
        
        # Windows
        self.window = get_window('hann', self.frame_len, fftbins=True)
        
        # Linear frequency grid
        self.num_linear_bins = n_fft // 2 + 1
        self.linear_freqs = np.fft.rfftfreq(n_fft, d=1.0/self.fs)
        
        # Precompute Bark Scale boundaries
        self.init_bark_scale()

    def hz_to_bark(self, f):
        """Analytical formula converting Hz to Bark scale."""
        return 13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)

    def init_bark_scale(self):
        """Precomputes the Linear-to-Bark transformation matrix and interpolation grids."""
        # Calculate Bark values for our linear frequency bins
        self.linear_barks = self.hz_to_bark(self.linear_freqs)
        
        # Define uniform critical band centers in the Bark domain
        min_bark = 0.0
        max_bark = self.hz_to_bark(self.fs / 2.0)
        
        # We define band edges and centers. For N bands, we need N+2 edge points to define overlapping triangles.
        self.bark_centers = np.linspace(min_bark, max_bark, self.num_bands)
        
        # Build the static Linear-to-Bark filterbank matrix of shape (num_bands, num_linear_bins)
        # We use overlapping triangular filters in the Bark domain.
        self.M_L2B = np.zeros((self.num_bands, self.num_linear_bins))
        
        # Calculate filter widths in Bark
        bark_step = (max_bark - min_bark) / (self.num_bands - 1) if self.num_bands > 1 else 1.0
        
        for b in range(self.num_bands):
            center = self.bark_centers[b]
            left = center - bark_step
            right = center + bark_step
            
            # Triangular filter shape
            filter_curve = np.zeros_like(self.linear_barks)
            # Left slope
            left_mask = (self.linear_barks >= left) & (self.linear_barks <= center)
            if left_mask.any():
                filter_curve[left_mask] = (self.linear_barks[left_mask] - left) / (center - left)
            # Right slope
            right_mask = (self.linear_barks > center) & (self.linear_barks <= right)
            if right_mask.any():
                filter_curve[right_mask] = (right - self.linear_barks[right_mask]) / (right - center)
                
            # Normalize filter to sum to 1.0 to preserve energy scale
            fs_sum = np.sum(filter_curve)
            if fs_sum > 0:
                filter_curve /= fs_sum
                
            self.M_L2B[b, :] = filter_curve

    def linear_to_bark(self, linear_mag):
        """
        Maps linear FFT magnitude bins to 24 critical Bark scale bands.
        linear_mag: array of shape (num_frames, num_linear_bins) or (num_linear_bins,)
        Returns: array of shape (num_frames, num_bands) or (num_bands,)
        """
        return np.dot(linear_mag, self.M_L2B.T)

    def bark_to_linear_gains(self, bark_gains):
        """
        Smoothly interpolates 24 Bark band gains back to 257 linear frequency bins.
        bark_gains: array of shape (num_frames, num_bands) or (num_bands,)
        Returns: array of shape (num_frames, num_linear_bins) or (num_linear_bins,)
        """
        is_1d = len(bark_gains.shape) == 1
        if is_1d:
            bark_gains = bark_gains[np.newaxis, :]
            
        num_frames = bark_gains.shape[0]
        linear_gains = np.zeros((num_frames, self.num_linear_bins))
        
        for f in range(num_frames):
            # Linearly interpolate the gains across the linear Bark grid
            linear_gains[f, :] = np.interp(self.linear_barks, self.bark_centers, bark_gains[f, :])
            
        return linear_gains[0] if is_1d else linear_gains

    def stft_analysis(self, x):
        """
        Splits signal into overlapping frames, applies Hann window, and computes RFFT.
        x: 1D monophonic audio array normalized in [-1.0, 1.0]
        Returns:
            mags: (num_frames, num_linear_bins) - linear FFT magnitudes
            phases: (num_frames, num_linear_bins) - noisy phase spectrum (radians)
        """
        # Padding input signal at the beginning and end to allow centered frames
        pad_len = self.frame_len // 2
        x_padded = np.pad(x, pad_len, mode='reflect')
        
        # Calculate number of frames
        num_frames = 1 + (len(x_padded) - self.frame_len) // self.hop_len
        
        mags = np.zeros((num_frames, self.num_linear_bins))
        phases = np.zeros((num_frames, self.num_linear_bins))
        
        for f in range(num_frames):
            start = f * self.hop_len
            end = start + self.frame_len
            frame = x_padded[start:end] * self.window
            
            # RFFT (zero-pads automatically if frame_len < n_fft)
            spectrum = np.fft.rfft(frame, n=self.n_fft)
            
            mags[f, :] = np.abs(spectrum)
            phases[f, :] = np.angle(spectrum)
            
        return mags, phases

    def stft_synthesis(self, mags, phases):
        """
        Executes Inverse FFT and Weighted Overlap-Add (WOLA) to reconstruct continuous audio.
        mags: (num_frames, num_linear_bins)
        phases: (num_frames, num_linear_bins)
        Returns:
            y: 1D enhanced audio array
        """
        num_frames = mags.shape[0]
        # Length of reconstructed padded signal
        recon_len = self.frame_len + (num_frames - 1) * self.hop_len
        y_padded = np.zeros(recon_len)
        window_sum = np.zeros(recon_len)
        
        for f in range(num_frames):
            start = f * self.hop_len
            end = start + self.frame_len
            
            # Reconstruct complex spectrum
            spectrum = mags[f, :] * np.exp(1j * phases[f, :])
            
            # IRFFT back to time domain
            frame_recon = np.fft.irfft(spectrum, n=self.n_fft)[:self.frame_len]
            
            # Apply synthesis window (Hann window again for smooth crossfades)
            y_padded[start:end] += frame_recon * self.window
            window_sum[start:end] += self.window ** 2
            
        # Normalize by window sum to achieve perfect reconstruction (WOLA)
        # Avoid division by zero at boundary tails
        safe_mask = window_sum > 1e-4
        y_padded[safe_mask] /= window_sum[safe_mask]
        
        # Remove padding
        pad_len = self.frame_len // 2
        y = y_padded[pad_len:-pad_len]
        return y
