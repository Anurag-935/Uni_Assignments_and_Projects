import torch
import torch.nn as nn

class HybridNoiseEnhancerNet(nn.Module):
    """
    Hybrid GRU-DNN Speech Enhancement Network.
    - Input: Noisy Bark magnitude spectrum of shape (batch_size, seq_len, num_bark_bands)
    - GRU: Encodes temporal context and dynamics of noise statistics (implicit noise tracking)
    - DNN: Maps the spatial features (harmonic structure) and noise history to a clean spectral gain mask
    - Output: Attenuation gains G in [0, 1] of shape (batch_size, seq_len, num_bark_bands)
    """
    def __init__(self, num_bark_bands=24, gru_hidden_dim=128, num_gru_layers=1):
        super().__init__()
        self.num_bands = num_bark_bands
        self.gru_hidden_dim = gru_hidden_dim
        self.num_gru_layers = num_gru_layers
        
        # 1. GRU Layer (Temporal Noise History & Dynamics Tracker)
        self.gru = nn.GRU(
            input_size=num_bark_bands,
            hidden_size=gru_hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True
        )
        
        # 2. Feedforward DNN (Spatial Parameter Mapper)
        self.dnn = nn.Sequential(
            nn.Linear(num_bark_bands + gru_hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_bark_bands),
            nn.Sigmoid()  # Compresses output to [0.0, 1.0] representing spectral gains
        )
        
    def forward(self, x, h_state=None):
        """
        x: tensor of shape (batch_size, seq_len, num_bark_bands)
        h_state: tensor of shape (num_layers, batch_size, gru_hidden_dim) (optional)
        Returns:
            gains: tensor of shape (batch_size, seq_len, num_bark_bands) - Spectral Gain Mask
            h_state: updated GRU hidden state for real-time stateful inference
        """
        # GRU outputs (batch_size, seq_len, gru_hidden_dim) and the new hidden state
        gru_out, h_state = self.gru(x, h_state)
        
        # Concatenate noisy frame magnitude features and the GRU temporal tracking features
        # Shape: (batch_size, seq_len, num_bark_bands + gru_hidden_dim)
        combined = torch.cat([x, gru_out], dim=-1)
        
        # DNN estimates the target gain mask
        gains = self.dnn(combined)
        
        return gains, h_state

    def count_parameters(self):
        """Returns the total number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Self-test code to verify tensor shapes and parameter count
    model = HybridNoiseEnhancerNet()
    print(f"Hybrid Noise Enhancer Initialized.")
    print(f"Total Trainable Parameters: {model.count_parameters()}")
    
    # Test batch shape: (batch_size=4, seq_len=10, num_bark_bands=24)
    x_test = torch.randn(4, 10, 24)
    gains, h = model(x_test)
    print(f"Input Shape:  {x_test.shape}")
    print(f"Output Shape: {gains.shape}")
    assert gains.shape == (4, 10, 24)
    print("Shape assertion passed successfully!")
