import os
import urllib.request
import sys

def download_file(url, dest_path):
    """Downloads a file from a URL to a local destination with progress updates."""
    filename = os.path.basename(dest_path)
    print(f"Downloading {filename}...")
    try:
        # Custom user-agent to bypass any potential bot filters
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) SpeechEnhancerApp/1.0'}
        )
        with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        print(f"✓ Successfully saved to {dest_path}")
        return True
    except Exception as e:
        print(f"✗ Failed to download {filename}. Error: {e}", file=sys.stderr)
        return False

def setup_real_dataset():
    # Setup paths relative to this script
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(src_dir)
    
    clean_dir = os.path.join(project_dir, 'data', 'clean')
    noise_dir = os.path.join(project_dir, 'data', 'noise')
    
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(noise_dir, exist_ok=True)
    
    print("=" * 60)
    print("AUTOMATED REAL-WORLD DATASET DOWNLOADER (MS-SNSD)")
    print("Fetching high-quality human speech and noise profiles from Microsoft...")
    print("=" * 60)
    
    # 1. Microsoft MS-SNSD Clean Speech URLs
    clean_base_url = "https://github.com/microsoft/MS-SNSD/raw/master/CleanSpeech_training"
    clean_files = ["clnsp1.wav", "clnsp2.wav", "clnsp3.wav", "clnsp4.wav", "clnsp5.wav"]
    
    # 2. Microsoft MS-SNSD Noise URLs
    noise_base_url = "https://github.com/microsoft/MS-SNSD/raw/master/Noise_training"
    noise_files = ["AirConditioner_1.wav", "Babble_1.wav", "Cafeteria_1.wav", "CopyMachine_1.wav"]
    
    success_count = 0
    total_files = len(clean_files) + len(noise_files)
    
    # Download clean voice files
    print("\n[1/2] Fetching Clean Human Speech Recordings...")
    for f in clean_files:
        url = f"{clean_base_url}/{f}"
        dest = os.path.join(clean_dir, f)
        if download_file(url, dest):
            success_count += 1
            
    # Download background noise files
    print("\n[2/2] Fetching Real Background Noise Profiles...")
    for f in noise_files:
        url = f"{noise_base_url}/{f}"
        dest = os.path.join(noise_dir, f)
        if download_file(url, dest):
            success_count += 1
            
    print("\n" + "=" * 60)
    print(f"DOWNLOAD SUMMARY: {success_count}/{total_files} files successfully downloaded.")
    if success_count == total_files:
        print("✓ Your prototype is now loaded with real Microsoft speech and noise datasets!")
    else:
        print("⚠ Some downloads failed. Please check your internet connection and try again.")
    print("=" * 60)

if __name__ == "__main__":
    setup_real_dataset()
