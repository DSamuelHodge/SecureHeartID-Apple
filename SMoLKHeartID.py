#SMoLK philosophy is applied to the SecureBiometricECGModel for lighter version for resource constrained devices.
#Refer to notebook for more training/testing information. https://colab.research.google.com/drive/1G57am1y_kPjRCJAqTzhSumuEpngeJpx4#scrollTo=23f2d7f1

import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedSecureBiometricECGModel(nn.Module):
    def __init__(self, input_size=1000, key_size=100, num_kernels=24):
        super(EnhancedSecureBiometricECGModel, self).__init__()
        
        self.kernel1 = nn.Conv1d(1, num_kernels, 192, stride=1, bias=True)
        self.kernel2 = nn.Conv1d(1, num_kernels, 96, stride=1, bias=True)
        self.kernel3 = nn.Conv1d(1, num_kernels, 64, stride=1, bias=True)
        
        self.power_spectrum_size = input_size // 2 + 1
        
        self.fc = nn.Linear(num_kernels * 3 + self.power_spectrum_size + key_size, 100)
    
    def forward(self, x, key):
        # Apply learned kernels
        c1 = F.leaky_relu(self.kernel1(x)).mean(dim=-1)
        c2 = F.leaky_relu(self.kernel2(x)).mean(dim=-1)
        c3 = F.leaky_relu(self.kernel3(x)).mean(dim=-1)
        
        # Compute power spectrum
        power_spectrum = torch.stft(x.squeeze(1), n_fft=self.power_spectrum_size*2-2, return_complex=True).abs().mean(dim=1)
        
        # Concatenate features
        features = torch.cat([c1, c2, c3, power_spectrum, key], dim=1)
        
        # Final fully connected layer
        output = self.fc(features)
        
        return output

class EnhancedSecureModel(nn.Module):
    def __init__(self, input_size=1000, key_size=100, num_kernels=24):
        super(EnhancedSecureModel, self).__init__()
        self.network = EnhancedSecureBiometricECGModel(input_size, key_size, num_kernels)

    def forward(self, xA, xP, xN, k1, k2):
        output_a = self.network(xA, k1)
        output_p1 = self.network(xP, k1)
        output_p2 = self.network(xP, k2)
        output_n1 = self.network(xN, k1)
        output_n2 = self.network(xN, k2)
        return output_a, output_p1, output_p2, output_n1, output_n2

    def get_embedding(self, x, k):
        return self.network(x, k)

# Helper function to generate random keys (unchanged)
def generate_key(key_size=100):
    key = torch.randint(0, 2, (1, key_size), dtype=torch.float32)
    return key / torch.norm(key)

# Secure Triplet Loss function (unchanged)
class SecureTripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(SecureTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive1, positive2, negative1, negative2):
        dist_p1 = torch.pairwise_distance(anchor, positive1)
        dist_p2 = torch.pairwise_distance(anchor, positive2)
        dist_n1 = torch.pairwise_distance(anchor, negative1)
        dist_n2 = torch.pairwise_distance(anchor, negative2)

        losses = torch.relu(dist_p1 - dist_n1 + self.margin) + \
                 torch.relu(dist_p1 - dist_n2 + self.margin) + \
                 torch.relu(dist_p2 - dist_n1 + self.margin) + \
                 torch.relu(dist_p2 - dist_n2 + self.margin)

        return losses.mean()

# Example usage
input_size = 1000  # Length of ECG signal
key_size = 100     # Size of the security key
num_kernels = 24   # Number of kernels in each learned kernel layer

model = EnhancedSecureModel(input_size, key_size, num_kernels)
criterion = SecureTripletLoss()

# Example forward pass (you would need to provide actual data)
xA = torch.randn(1, 1, input_size)  # Anchor
xP = torch.randn(1, 1, input_size)  # Positive
xN = torch.randn(1, 1, input_size)  # Negative
k1 = generate_key(key_size)
k2 = generate_key(key_size)

outputs = model(xA, xP, xN, k1, k2)
loss = criterion(*outputs)