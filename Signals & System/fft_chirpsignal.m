clc;
clear all;
close all;

%% Radar Parameters
c = 3e8;          % Speed of light
Fs = 2e6;         % Sampling frequency
N = 1024;         % FFT size
B = 10e6;         % Bandwidth
T = 1e-3;         % Sweep time
S = B / T;        % Chirp slope

%% Simulated Beat Signal
fb_true = 100e3;              % Beat frequency (100 kHz)
t = (0:N-1) / Fs;             
signal = cos(2*pi*fb_true*t);

%% FFT
X = abs(fft(signal));
X_half = X(1:N/2);            % Half spectrum
%% Find Peak
[peak_val, idx] = max(X_half);

%% Convert index → frequency
fb = (idx - 1) * Fs / N;

%% Range Calculation
R = (c * fb) / (2 * S);

%% Display Results
fprintf('Peak Index = %d\n', idx);
fprintf('Beat Frequency = %.2f Hz\n', fb);
fprintf('Estimated Range = %.2f meters\n', R);

%% Plot
f = (0:N/2-1) * (Fs/N);
range_axis = (c * f) / (2 * S);

figure;
plot(range_axis, X_half);
xlabel('Range (meters)');
ylabel('Magnitude');
title('Range FFT');
grid on;
