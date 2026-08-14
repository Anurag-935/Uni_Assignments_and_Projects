import os
import shutil
import glob

def setup_real_dataset(num_clean_to_copy=600):
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    temp_dir = os.path.join(project_dir, 'ms_snsd_temp')
    
    clean_dest = os.path.join(project_dir, 'data', 'clean')
    noise_dest = os.path.join(project_dir, 'data', 'noise')
    
    os.makedirs(clean_dest, exist_ok=True)
    os.makedirs(noise_dest, exist_ok=True)
    
    print("=" * 60)
    print("REAL-WORLD DATASET INGESTION ENGINE (MS-SNSD)")
    print("=" * 60)
    
    # 1. Clear any old synthetic files
    print("Cleaning up old synthetic demo files...")
    old_clean_synths = glob.glob(os.path.join(clean_dest, 'synth_*'))
    old_noise_synths = glob.glob(os.path.join(noise_dest, '*_hum.wav')) + \
                       glob.glob(os.path.join(noise_dest, '*_hiss.wav')) + \
                       glob.glob(os.path.join(noise_dest, '*_rumble.wav'))
    
    for f in old_clean_synths + old_noise_synths:
        try:
            os.remove(f)
        except Exception as e:
            print(f"Warning: Could not remove old file {f}: {e}")
            
    # 2. Ingest clean speech files
    src_clean_dir = os.path.join(temp_dir, 'clean_train')
    if os.path.exists(src_clean_dir):
        clean_files = sorted(glob.glob(os.path.join(src_clean_dir, '*.wav')))
        copy_count = min(len(clean_files), num_clean_to_copy)
        print(f"Ingesting {copy_count} real human speech files...")
        for i in range(copy_count):
            shutil.copy(clean_files[i], clean_dest)
    else:
        print("Error: clean_train folder not found in cloned repository!")
        return False
        
    # 3. Ingest noise files
    src_noise_dir = os.path.join(temp_dir, 'noise_train')
    if os.path.exists(src_noise_dir):
        noise_files = glob.glob(os.path.join(src_noise_dir, '*.wav'))
        print(f"Ingesting all {len(noise_files)} environmental noise files...")
        for f in noise_files:
            shutil.copy(f, noise_dest)
    else:
        print("Error: noise_train folder not found in cloned repository!")
        return False
        
    # 4. Clean up temporary cloned repository to free up disk space (~3.6 GB)
    print("\nPurging temporary clone directory to reclaim disk space...")
    try:
        shutil.rmtree(temp_dir)
        print("✓ Temporary files successfully purged.")
    except Exception as e:
        print(f"Warning: Could not automatically purge temporary folder: {e}")
        print("You can manually delete the 'ms_snsd_temp' folder on your Desktop.")
        
    print("\n" + "=" * 60)
    print(f"SUCCESS: Real-world training dataset set up at Desktop\\hybrid_speech_enhancement\\data\\")
    print(f"  Clean voice files: {copy_count}")
    print(f"  Noise profiles:    {len(noise_files)}")
    print("=" * 60)
    return True

if __name__ == "__main__":
    setup_real_dataset()
