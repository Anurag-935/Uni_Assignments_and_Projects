import os
import sys
import threading
import winsound
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# Ensure we can find the src files
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from dsp import BarkSpeechDSP
from enhance import enhance_audio

class SpeechEnhancerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI-Driven Non-Linear Multi-Band Speech Enhancer")
        self.root.geometry("1100x750")
        self.root.configure(bg="#121212")
        
        self.project_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Audio File Paths
        self.noisy_path = ""
        self.enhanced_path = os.path.join(self.project_dir, 'enhanced_clean.wav')
        self.plot_path = os.path.join(self.project_dir, 'spectrogram_comparison.png')
        
        # Audio Playing States
        self.playing_noisy = False
        self.playing_enhanced = False
        
        # Apply modern styling
        self.setup_styles()
        
        # Create Layout
        self.create_widgets()
        
        # Check for pre-existing synthetic data
        self.check_or_create_demo()

    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use("clam")
        
        # Configure frames and labels
        self.style.configure(".", bg="#121212", foreground="#ffffff")
        self.style.configure("TFrame", background="#121212")
        self.style.configure("Card.TFrame", background="#1e1e1e", relief="flat")
        self.style.configure("TLabel", background="#121212", foreground="#ffffff", font=("Segoe UI", 10))
        self.style.configure("Card.TLabel", background="#1e1e1e", foreground="#ffffff", font=("Segoe UI", 10))
        self.style.configure("Header.TLabel", background="#1e1e1e", foreground="#00adb5", font=("Segoe UI", 14, "bold"))
        self.style.configure("Title.TLabel", background="#121212", foreground="#00adb5", font=("Segoe UI", 18, "bold"))
        
        # Progressbar
        self.style.configure("TProgressbar", thickness=8, troughcolor="#2d2d2d", background="#00adb5")

    def create_widgets(self):
        # Top Title Bar
        title_frame = ttk.Frame(self.root, style="TFrame")
        title_frame.pack(fill="x", padx=30, pady=20)
        
        title_label = ttk.Label(
            title_frame, 
            text="AI-Driven Non-Linear Multi-Band Speech Enhancer", 
            style="Title.TLabel"
        )
        title_label.pack(side="left")
        
        subtitle_label = ttk.Label(
            title_frame, 
            text="Hybrid DSP + GRU-DNN Real-Time Engine", 
            foreground="#888888",
            font=("Segoe UI", 10, "italic")
        )
        subtitle_label.pack(side="left", padx=15, pady=8)
        
        # Main Work Area Splitter
        main_container = ttk.Frame(self.root, style="TFrame")
        main_container.pack(fill="both", expand=True, padx=30, pady=5)
        
        # Left Panel (Controls and Actions)
        left_panel = ttk.Frame(main_container, style="TFrame", width=380)
        left_panel.pack(side="left", fill="both", expand=False)
        left_panel.pack_propagate(False)
        
        # Control Card
        control_card = ttk.Frame(left_panel, style="Card.TFrame")
        control_card.pack(fill="both", expand=True, pady=5)
        
        # 1. File Selection Section
        file_header = ttk.Label(control_card, text="1. AUDIO SOURCE SELECTION", style="Header.TLabel")
        file_header.pack(anchor="w", padx=20, pady=(10, 5))
        
        btn_frame = ttk.Frame(control_card, style="Card.TFrame")
        btn_frame.pack(fill="x", padx=20, pady=2)
        
        self.btn_browse = tk.Button(
            btn_frame, text="Load Voice File (.wav)", font=("Segoe UI", 10, "bold"),
            bg="#2d2d2d", fg="#ffffff", activebackground="#444444", activeforeground="#ffffff",
            bd=0, padx=10, pady=6, cursor="hand2", command=self.browse_file
        )
        self.btn_browse.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        self.btn_demo = tk.Button(
            btn_frame, text="Use Demo Voice", font=("Segoe UI", 10, "bold"),
            bg="#00adb5", fg="#ffffff", activebackground="#008c9e", activeforeground="#ffffff",
            bd=0, padx=10, pady=6, cursor="hand2", command=self.load_demo
        )
        self.btn_demo.pack(side="right", fill="x", expand=True, padx=(5, 0))
        
        self.file_label = ttk.Label(
            control_card, text="No voice file loaded.", 
            foreground="#aaaaaa", font=("Segoe UI", 9, "italic"), style="Card.TLabel",
            wraplength=340
        )
        self.file_label.pack(anchor="w", padx=20, pady=3)
        
        # Divider
        divider1 = tk.Frame(control_card, height=1, bg="#2d2d2d")
        divider1.pack(fill="x", padx=20, pady=8)
        
        # 2. Parameters Section
        param_header = ttk.Label(control_card, text="2. PERFORMANCE CONTROLS", style="Header.TLabel")
        param_header.pack(anchor="w", padx=20, pady=(0, 5))
        
        # Slider for Spectral Floor
        floor_label_frame = ttk.Frame(control_card, style="Card.TFrame")
        floor_label_frame.pack(fill="x", padx=20, pady=1)
        ttk.Label(floor_label_frame, text="Spectral Floor (Beta):", style="Card.TLabel").pack(side="left")
        self.floor_val_label = ttk.Label(floor_label_frame, text="0.02", foreground="#00adb5", font=("Segoe UI", 10, "bold"), style="Card.TLabel")
        self.floor_val_label.pack(side="right")
        
        self.floor_slider = ttk.Scale(
            control_card, from_=0.001, to=0.1, value=0.02, orient="horizontal",
            command=self.update_slider_label
        )
        self.floor_slider.pack(fill="x", padx=20, pady=(2, 5))
        
        ttk.Label(
            control_card, 
            text="Note: Lower values drop more noise but may cause subtle gurgling. Default 0.02 is optimally balanced.",
            foreground="#888888", font=("Segoe UI", 8), style="Card.TLabel", wraplength=340
        ).pack(anchor="w", padx=20, pady=(0, 5))
        
        # Divider
        divider2 = tk.Frame(control_card, height=1, bg="#2d2d2d")
        divider2.pack(fill="x", padx=20, pady=8)
        
        # 3. Actions Section
        action_header = ttk.Label(control_card, text="3. SPEECH ENHANCEMENT ENGINE", style="Header.TLabel")
        action_header.pack(anchor="w", padx=20, pady=(0, 5))
        
        self.btn_enhance = tk.Button(
            control_card, text="🚀 ENHANCE & DE-NOISE AUDIO", font=("Segoe UI", 12, "bold"),
            bg="#00adb5", fg="#ffffff", activebackground="#008c9e", activeforeground="#ffffff",
            bd=0, pady=10, cursor="hand2", command=self.trigger_enhancement
        )
        self.btn_enhance.pack(fill="x", padx=20, pady=5)
        
        self.progress_bar = ttk.Progressbar(control_card, mode="determinate")
        self.progress_bar.pack(fill="x", padx=20, pady=5)
        self.progress_bar.pack_forget() # hide initially
        
        self.status_label = ttk.Label(
            control_card, text="Status: Ready", 
            foreground="#aaaaaa", font=("Segoe UI", 9), style="Card.TLabel"
        )
        self.status_label.pack(anchor="w", padx=20, pady=3)
        
        # Divider
        divider3 = tk.Frame(control_card, height=1, bg="#2d2d2d")
        divider3.pack(fill="x", padx=20, pady=8)
        
        # 4. Audio Playback Control
        play_header = ttk.Label(control_card, text="4. REAL-TIME PLAYBACK", style="Header.TLabel")
        play_header.pack(anchor="w", padx=20, pady=(0, 5))
        
        playback_frame = ttk.Frame(control_card, style="Card.TFrame")
        playback_frame.pack(fill="x", padx=20, pady=2)
        
        self.btn_play_noisy = tk.Button(
            playback_frame, text="Play Original Noisy", font=("Segoe UI", 9, "bold"),
            bg="#e06666", fg="#ffffff", activebackground="#cc4141", activeforeground="#ffffff",
            bd=0, padx=5, pady=6, cursor="hand2", command=self.toggle_play_noisy
        )
        self.btn_play_noisy.pack(side="left", fill="x", expand=True)
        
        self.btn_play_enhanced = tk.Button(
            playback_frame, text="Play Enhanced Clean", font=("Segoe UI", 9, "bold"),
            bg="#6fa8dc", fg="#ffffff", activebackground="#3d85c6", activeforeground="#ffffff",
            bd=0, padx=5, pady=6, cursor="hand2", command=self.toggle_play_enhanced
        )
        self.btn_play_enhanced.pack(side="right", fill="x", expand=True, padx=(10, 0))
        
        # Divider 4
        divider4 = tk.Frame(control_card, height=1, bg="#2d2d2d")
        divider4.pack(fill="x", padx=20, pady=8)
        
        # 5. Quantitative Speech Metrics
        metrics_header = ttk.Label(control_card, text="5. SPEECH METRICS", style="Header.TLabel")
        metrics_header.pack(anchor="w", padx=20, pady=(0, 3))
        
        metrics_frame = ttk.Frame(control_card, style="Card.TFrame")
        metrics_frame.pack(fill="x", padx=20, pady=2)
        
        self.lbl_input_snr = ttk.Label(
            metrics_frame, text="Input SNR: -- dB", 
            foreground="#aaaaaa", font=("Segoe UI", 10), style="Card.TLabel"
        )
        self.lbl_input_snr.grid(row=0, column=0, sticky="w", padx=(0, 30), pady=1)
        
        self.lbl_output_snr = ttk.Label(
            metrics_frame, text="Output SNR: -- dB", 
            foreground="#aaaaaa", font=("Segoe UI", 10), style="Card.TLabel"
        )
        self.lbl_output_snr.grid(row=0, column=1, sticky="w", pady=1)
        
        self.lbl_snr_improvement = ttk.Label(
            metrics_frame, text="SNR Improvement: -- dB", 
            foreground="#00adb5", font=("Segoe UI", 11, "bold"), style="Card.TLabel"
        )
        self.lbl_snr_improvement.grid(row=1, column=0, columnspan=2, sticky="w", pady=(3, 1))
        
        # Right Panel (Spectrogram and waveform visualizer)
        right_panel = ttk.Frame(main_container, style="TFrame")
        right_panel.pack(side="right", fill="both", expand=True, padx=(20, 0))
        
        visual_card = ttk.Frame(right_panel, style="Card.TFrame")
        visual_card.pack(fill="both", expand=True, pady=5)
        
        visual_title = ttk.Label(
            visual_card, text="REAL-TIME SPECTRAL SUBTRACTION VISUAL VERIFICATION", 
            style="Header.TLabel"
        )
        visual_title.pack(anchor="center", pady=(15, 10))
        
        # Image Display Area
        self.visual_canvas = tk.Canvas(visual_card, bg="#181818", bd=0, highlightthickness=0)
        self.visual_canvas.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
        # Initial Canvas Message
        self.canvas_text = self.visual_canvas.create_text(
            330, 250, text="Spectrogram comparisons will be rendered here\nafter executing the speech enhancer.",
            fill="#666666", font=("Segoe UI", 12, "bold"), justify="center"
        )
        
        # Bind canvas resize to automatically scale image if needed
        self.visual_canvas.bind("<Configure>", self.on_canvas_resize)
        
        self.tk_image = None

    def update_slider_label(self, val):
        self.floor_val_label.configure(text=f"{float(val):.3f}")

    def browse_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("WAV Audio Files", "*.wav")]
        )
        if file_path:
            self.noisy_path = file_path
            self.file_label.configure(text=os.path.basename(file_path), foreground="#00adb5")
            self.status_label.configure(text="Status: Custom voice file loaded.")

    def check_or_create_demo(self):
        # Create standard directories
        data_dir = os.path.join(self.project_dir, 'data')
        demo_noisy = os.path.join(data_dir, 'demo_noisy.wav')
        
        # If the demo exists, verify it is standard 16-bit PCM (needed for winsound)
        if os.path.exists(demo_noisy):
            try:
                import soundfile as sf
                info = sf.info(demo_noisy)
                if info.subtype == 'PCM_16':
                    return True
                else:
                    print("Found old 32-bit float demo. Re-generating as 16-bit PCM...")
                    os.remove(demo_noisy)
            except Exception as e:
                print(f"Old demo check warning: {e}")
            
        self.status_label.configure(text="Status: Synthesizing demo voice audio...")
        self.root.update()
        
        try:
            import soundfile as sf
            from dataset import generate_synthetic_data
            
            # Synthesize clean voice and noise on the fly (very fast, ~0.2 seconds)
            clean_paths, noise_paths = generate_synthetic_data(data_dir, num_clean=3, num_noise=2)
            
            if len(clean_paths) > 0 and len(noise_paths) > 0:
                clean, fs = sf.read(clean_paths[0])
                noise, _ = sf.read(noise_paths[0])
                
                # Mix them at 0dB SNR (equal power)
                p_c = np.mean(clean**2)
                p_n = np.mean(noise**2) + 1e-8
                scalar = np.sqrt(p_c / p_n)
                
                noisy_mix = clean + 0.8 * scalar * noise
                noisy_mix /= np.max(np.abs(noisy_mix)) + 1e-8
                
                # Write standard demo WAV file
                sf.write(demo_noisy, noisy_mix, fs, subtype='PCM_16')
                self.status_label.configure(text="Status: Demo voice ready.")
                return True
            else:
                self.status_label.configure(text="Status: Demo voice setup failed.")
                return False
        except Exception as e:
            print(f"Demo generation failed: {e}")
            self.status_label.configure(text=f"Status: Demo setup error: {e}")
            return False

    def load_demo(self):
        demo_noisy = os.path.join(self.project_dir, 'data', 'demo_noisy.wav')
        
        # Attempt to find or dynamically generate the demo WAV
        success = os.path.exists(demo_noisy) or self.check_or_create_demo()
        
        if success and os.path.exists(demo_noisy):
            self.noisy_path = demo_noisy
            self.file_label.configure(text="Loaded: Demo Noisy Speech (WAV)", foreground="#00adb5")
            self.status_label.configure(text="Status: Demo voice loaded.")
        else:
            messagebox.showerror("Error", "Could not synthesize or load the demo voice file. Please load a custom WAV file instead.")

    def trigger_enhancement(self):
        if not self.noisy_path:
            messagebox.showerror("Error", "Please load a voice file (.wav) or click 'Use Demo Voice' first!")
            return
            
        self.btn_enhance.configure(state="disabled", bg="#2d2d2d", text="⚙️ PROCESSING AUDIO...")
        self.btn_browse.configure(state="disabled")
        self.btn_demo.configure(state="disabled")
        
        self.progress_bar.pack(fill="x", padx=20, pady=10)
        self.progress_bar.configure(value=10)
        self.status_label.configure(text="Status: Running STFT Analysis...")
        
        # Run enhancement in a background thread to prevent UI lockup
        enhance_thread = threading.Thread(target=self.run_enhancement_process, daemon=True)
        enhance_thread.start()

    def run_enhancement_process(self):
        try:
            self.root.after(0, lambda: self.progress_bar.configure(value=30))
            self.root.after(0, lambda: self.status_label.configure(text="Status: Estimating noise temporal profiles (GRU)..."))
            
            # Read UI slider value for spectral floor
            beta = float(self.floor_slider.get())
            
            # Execute actual DSP & ML enhancement pipeline
            self.root.after(0, lambda: self.progress_bar.configure(value=60))
            self.root.after(0, lambda: self.status_label.configure(text="Status: Applying safeguarded gains & synthesis (WOLA)..."))
            
            _, input_snr, output_snr, snr_improvement = enhance_audio(self.noisy_path, self.enhanced_path, spectral_floor=beta)
            
            # Successful completion
            self.root.after(0, lambda: self.on_enhancement_success(input_snr, output_snr, snr_improvement))
        except Exception as e:
            self.root.after(0, lambda: self.on_enhancement_error(str(e)))

    def on_enhancement_success(self, input_snr, output_snr, snr_improvement):
        self.progress_bar.configure(value=100)
        self.status_label.configure(text="Status: ✓ Speech Enhancement Complete!", foreground="#00adb5")
        
        self.btn_enhance.configure(state="normal", bg="#00adb5", text="🚀 ENHANCE & DE-NOISE AUDIO")
        self.btn_browse.configure(state="normal")
        self.btn_demo.configure(state="normal")
        self.progress_bar.pack_forget()
        
        # Update SNR Labels
        self.lbl_input_snr.configure(text=f"Input SNR: {input_snr:.1f} dB")
        self.lbl_output_snr.configure(text=f"Output SNR: {output_snr:.1f} dB")
        self.lbl_snr_improvement.configure(text=f"SNR Improvement: +{snr_improvement:.1f} dB")
        
        # Load and render spectrogram
        self.load_spectrogram_plot()
        messagebox.showinfo("Success", "Audio has been cleaned! You can now play and visually inspect the differences.")

    def on_enhancement_error(self, err_msg):
        self.btn_enhance.configure(state="normal", bg="#00adb5", text="🚀 ENHANCE & DE-NOISE AUDIO")
        self.btn_browse.configure(state="normal")
        self.btn_demo.configure(state="normal")
        self.progress_bar.pack_forget()
        self.status_label.configure(text="Status: Error during execution.", foreground="#e06666")
        messagebox.showerror("Engine Failure", f"An error occurred during audio processing:\n{err_msg}")

    def load_spectrogram_plot(self):
        if os.path.exists(self.plot_path):
            try:
                # Load image using native Tkinter PhotoImage (natively supports PNG files!)
                self.tk_image = tk.PhotoImage(file=self.plot_path)
                
                # Clear text
                self.visual_canvas.delete("all")
                
                # Get canvas size
                cw = self.visual_canvas.winfo_width()
                ch = self.visual_canvas.winfo_height()
                
                # Draw image centered
                self.visual_canvas.create_image(cw//2, ch//2, image=self.tk_image, anchor="center")
            except Exception as e:
                print(f"Image load fail: {e}")

    def on_canvas_resize(self, event):
        # Re-render image when canvas is resized
        if self.tk_image:
            self.visual_canvas.delete("all")
            self.visual_canvas.create_image(event.width//2, event.height//2, image=self.tk_image, anchor="center")
        else:
            # Re-center placeholder text
            self.visual_canvas.delete("all")
            self.canvas_text = self.visual_canvas.create_text(
                event.width//2, event.height//2, 
                text="Spectrogram comparisons will be rendered here\nafter executing the speech enhancer.",
                fill="#666666", font=("Segoe UI", 12, "bold"), justify="center"
            )

    # ------------------ Audio Playback Functions ------------------
    def toggle_play_noisy(self):
        if self.playing_noisy:
            self.stop_audio()
        else:
            self.stop_audio()
            if self.noisy_path and os.path.exists(self.noisy_path):
                self.playing_noisy = True
                self.btn_play_noisy.configure(text="⏹ Stop Playing", bg="#2d2d2d")
                threading.Thread(target=self.play_audio_file, args=(self.noisy_path, "noisy"), daemon=True).start()
            else:
                messagebox.showerror("Error", "No noisy audio file loaded!")

    def toggle_play_enhanced(self):
        if self.playing_enhanced:
            self.stop_audio()
        else:
            self.stop_audio()
            if os.path.exists(self.enhanced_path):
                self.playing_enhanced = True
                self.btn_play_enhanced.configure(text="⏹ Stop Playing", bg="#2d2d2d")
                threading.Thread(target=self.play_audio_file, args=(self.enhanced_path, "enhanced"), daemon=True).start()
            else:
                messagebox.showerror("Error", "No enhanced clean file generated yet! Run enhancement first.")

    def play_audio_file(self, path, sig_type):
        try:
            import soundfile as sf
            import time
            
            # Fetch the actual WAV duration to control the play loop
            info = sf.info(path)
            duration = info.duration
            
            # Trigger asynchronous playback (non-blocking)
            winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            
            # Sleep in short intervals to monitor for cancellation or end of file
            elapsed = 0.0
            step = 0.1
            while elapsed < duration:
                # Instantly abort if user clicked Stop
                if sig_type == "noisy" and not self.playing_noisy:
                    break
                if sig_type == "enhanced" and not self.playing_enhanced:
                    break
                
                time.sleep(step)
                elapsed += step
                
            # Safely stop sound play upon loop completion or cancellation
            winsound.PlaySound(None, winsound.SND_PURGE)
            
        except Exception as e:
            print(f"Playback error: {e}")
        finally:
            # Reset button state on the main Tkinter thread
            self.root.after(0, lambda: self.on_play_finished(sig_type))

    def on_play_finished(self, sig_type):
        if sig_type == "noisy":
            self.playing_noisy = False
            self.btn_play_noisy.configure(text="Play Original Noisy", bg="#e06666")
        else:
            self.playing_enhanced = False
            self.btn_play_enhanced.configure(text="Play Enhanced Clean", bg="#6fa8dc")

    def stop_audio(self):
        # Stop all playing sounds instantly
        winsound.PlaySound(None, winsound.SND_PURGE)
        self.playing_noisy = False
        self.playing_enhanced = False
        self.btn_play_noisy.configure(text="Play Original Noisy", bg="#e06666")
        self.btn_play_enhanced.configure(text="Play Enhanced Clean", bg="#6fa8dc")

if __name__ == "__main__":
    root = tk.Tk()
    app = SpeechEnhancerGUI(root)
    
    # Gracefully stop audio playback on close
    def on_closing():
        app.stop_audio()
        root.destroy()
        
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
