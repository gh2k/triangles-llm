import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class ConvBlock(nn.Module):
    """Convolutional block with Conv2d, BatchNorm, ReLU, and optional MaxPool."""
    
    def __init__(self, in_channels: int, out_channels: int, use_pool: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.use_pool = use_pool
        if use_pool:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.use_pool:
            x = self.pool(x)
        return x


class TriangleGenerator(nn.Module):
    """CNN encoder + FC decoder for generating triangle parameters."""
    
    def __init__(self, 
                 triangles_n: int = 100,
                 encoder_depth: int = 5,
                 channels_base: int = 64,
                 channels_progression: Optional[List[int]] = None,
                 hidden_dim: int = 512,
                 fc_hidden_dim: int = 1024):
        """
        Initialize the TriangleGenerator.
        
        Args:
            triangles_n: Number of triangles to generate
            encoder_depth: Number of convolutional blocks in encoder
            channels_base: Base number of channels (used if channels_progression is None)
            channels_progression: Explicit channel sizes for each block
            hidden_dim: Size of the feature vector after global average pooling
            fc_hidden_dim: Hidden dimension of the FC decoder
        """
        super().__init__()
        
        self.triangles_n = triangles_n
        self.encoder_depth = encoder_depth
        
        # Determine channel progression
        if channels_progression is None:
            # Default progression: double channels each block, cap at 512
            channels_progression = []
            for i in range(encoder_depth):
                channels = min(channels_base * (2 ** i), 512)
                channels_progression.append(channels)
        else:
            assert len(channels_progression) == encoder_depth, \
                f"channels_progression length {len(channels_progression)} must match encoder_depth {encoder_depth}"
        
        self.channels_progression = channels_progression
        
        # Build encoder
        encoder_layers = []
        in_channels = 3  # RGB input
        
        for i, out_channels in enumerate(channels_progression):
            # Don't use pooling on the last block to preserve more spatial information
            use_pool = (i < encoder_depth - 1)
            encoder_layers.append(ConvBlock(in_channels, out_channels, use_pool=use_pool))
            in_channels = out_channels
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Feature projection to fixed hidden_dim if needed
        self.feature_dim = channels_progression[-1]
        if self.feature_dim != hidden_dim:
            self.feature_projection = nn.Linear(self.feature_dim, hidden_dim)
        else:
            self.feature_projection = None
        
        # FC decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, fc_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(fc_hidden_dim, triangles_n * 10)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images tensor of shape (B, 3, H, W)
            
        Returns:
            Triangle parameters tensor of shape (B, triangles_n, 10)
            where each triangle has:
            - 6 coordinate values (x1, y1, x2, y2, x3, y3) with tanh activation [-1, 1]
            - 4 color values (r, g, b, a) with sigmoid activation [0, 1]
        """
        batch_size = x.size(0)
        
        # Encode
        features = self.encoder(x)
        
        # Global average pooling
        features = self.global_pool(features)
        features = features.view(batch_size, -1)
        
        # Project features if needed
        if self.feature_projection is not None:
            features = self.feature_projection(features)
        
        # Decode to triangle parameters
        output = self.decoder(features)
        
        # Reshape to (B, triangles_n, 10)
        output = output.view(batch_size, self.triangles_n, 10)
        
        # Apply activations
        # Coordinates: tanh for [-1, 1] range
        output[:, :, :6] = torch.tanh(output[:, :, :6])
        
        # Colors: sigmoid for [0, 1] range
        output[:, :, 6:] = torch.sigmoid(output[:, :, 6:])
        
        return output
    
    def get_triangle_params(self, output: torch.Tensor) -> tuple:
        """
        Split output tensor into coordinates and colors.
        
        Args:
            output: Output tensor from forward pass (B, triangles_n, 10)
            
        Returns:
            coordinates: (B, triangles_n, 6) - triangle vertex coordinates
            colors: (B, triangles_n, 4) - RGBA colors
        """
        coordinates = output[:, :, :6]
        colors = output[:, :, 6:]
        return coordinates, colors