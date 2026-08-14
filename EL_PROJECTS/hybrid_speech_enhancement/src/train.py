import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# Import our custom modules
from dsp import BarkSpeechDSP
from model import HybridNoiseEnhancerNet
from dataset import SpeechEnhancementDataset, generate_synthetic_data

def train_model(epochs=50, batch_size=8, lr=0.003, seq_len=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using training device: {device}")
    
    # 1. Initialize DSP Engine
    dsp = BarkSpeechDSP()
    
    # 2. Setup Data Directories
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate synthetic training files if none exist
    clean_dir = os.path.join(data_dir, 'clean')
    noise_dir = os.path.join(data_dir, 'noise')
    
    clean_files = []
    noise_files = []
    
    if os.path.exists(clean_dir):
        clean_files = [os.path.join(clean_dir, f) for f in os.listdir(clean_dir) if f.endswith('.wav')]
    if os.path.exists(noise_dir):
        noise_files = [os.path.join(noise_dir, f) for f in os.listdir(noise_dir) if f.endswith('.wav')]
        
    if len(clean_files) == 0 or len(noise_files) == 0:
        print("Data directories empty. Automatically generating high-quality synthetic training corpus...")
        clean_files, noise_files = generate_synthetic_data(data_dir, num_clean=10, num_noise=4)
    else:
        print(f"Found {len(clean_files)} clean speech files and {len(noise_files)} noise profiles in local directory.")

    # 3. Create Dataset and DataLoader
    dataset = SpeechEnhancementDataset(clean_files, noise_files, dsp, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # 4. Initialize Joint GRU-DNN Network
    model = HybridNoiseEnhancerNet(num_bark_bands=dsp.num_bands).to(device)
    print(f"Model parameters to train: {model.count_parameters()}")
    
    # 5. Define Optimizer, Loss, and Scheduler (MSE is ideal for continuous gain masks)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # 6. Training Loop
    print("\nStarting Training Session...")
    print("-" * 60)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (noisy_barks, target_gains) in enumerate(dataloader):
            noisy_barks = noisy_barks.to(device)
            target_gains = target_gains.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass: model expects (batch_size, seq_len, num_bark_bands)
            # Output shape: (batch_size, seq_len, num_bark_bands)
            predicted_gains, _ = model(noisy_barks)
            
            loss = criterion(predicted_gains, target_gains)
            loss.backward()
            
            # Gradient clipping for stable training
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        
        # Step the learning rate scheduler based on average epoch loss
        scheduler.step(avg_loss)
        
        print(f"Epoch [{epoch+1:02d}/{epochs:02d}] | Average MSE Loss: {avg_loss:.5f}")
        
    print("-" * 60)
    print("Training complete!")
    
    # Save the trained model checkpoint
    model_path = os.path.join(project_dir, 'model.pt')
    torch.save(model.state_dict(), model_path)
    print(f"Saved optimized weights to {model_path}")
    
if __name__ == "__main__":
    # Standard training settings
    train_model(epochs=50, batch_size=8, lr=0.002)
