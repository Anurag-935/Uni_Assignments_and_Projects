# AI-Driven Non-Linear Multi-Band Spectral Subtraction for Real-Time Speech Enhancement

This project implements a hybrid speech enhancement engine that combines classical Digital Signal Processing (DSP) pipelines with a Gated Recurrent Unit (GRU) and Deep Neural Network (DNN) controller to estimate the noise fingerprint and dynamically enhance audio in real-time.

## Project Structure
* `src/`
  * `dsp.py` - Core DSP components: Hann windowing, STFT, Linear-to-Bark scale mapping, Bark-to-Linear interpolation, and Overlap-Add (OLA) reconstruction.
  * `model.py` - Lightweight PyTorch joint GRU-DNN neural network architecture for real-time gain estimation.
  * `enhance.py` - Main enhancement streaming pipeline and CLI interface.
  * `dataset.py` - Synthetic noisy-clean data generator and PyTorch Dataset for offline training.
  * `train.py` - Training script for the GRU-DNN controller.
* `data/` - Directory for voice files (noisy/clean speech).
* `requirements.txt` - Dependencies (numpy, scipy, torch, soundfile).

## How It Works
1. **Time-Domain Framing & Windowing:** Monophonic 16kHz audio is framed into 30ms windows with 50% overlap, using a Hann window to satisfy the Constant-Overlap-Add (COLA) constraint.
2. **STFT Analysis:** Fast Fourier Transform (FFT) converts frames to linear frequency magnitudes (241 bins), preserving noisy phase.
3. **Psychoacoustic Bark Mapping:** The 241 linear magnitude bins are mapped to 24 critical bands on the human Bark Scale using a static projection matrix.
4. **GRU-DNN Controller:** The GRU tracks temporal noise changes across frames, and the DNN predicts a safeguarded spectral gain mask $G_b \in [\beta, 1]$ per band.
5. **Interpolation & Safegaurded Subtraction:** Bark gains are smoothly interpolated back to 241 linear bins and applied to the noisy magnitudes, eliminating musical noise.
6. **Overlap-Add (OLA) Reconstruction:** The enhanced magnitude is recombined with the noisy phase, inverted via IFFT, and stitched back using OLA.
