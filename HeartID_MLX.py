# heartid_mlx.py

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_stft import STFT

class EnhancedSecureBiometricECGModel(nn.Module):
    def __init__(self, input_size=1000, key_size=100, num_kernels=24):
        super().__init__()
        
        self.kernel1 = nn.Conv1d(1, num_kernels, 192, stride=1, bias=True)
        self.kernel2 = nn.Conv1d(1, num_kernels, 96, stride=1, bias=True)
        self.kernel3 = nn.Conv1d(1, num_kernels, 64, stride=1, bias=True)
        
        self.stft = STFT(n_fft=input_size, win_length=input_size//2, hop_length=input_size//4, return_db=True, onesided=True)
        self.power_spectrum_size = input_size // 2 + 1
        
        self.fc = nn.Linear(num_kernels * 3 + self.power_spectrum_size + key_size, 100)
    
    def __call__(self, x, key):
        c1 = nn.leaky_relu(self.kernel1(x)).mean(axis=-1)
        c2 = nn.leaky_relu(self.kernel2(x)).mean(axis=-1)
        c3 = nn.leaky_relu(self.kernel3(x)).mean(axis=-1)
        
        power_spectrum = self.stft(x.squeeze(1)).mean(axis=-1)
        
        features = mx.concatenate([c1, c2, c3, power_spectrum, key], axis=1)
        
        output = self.fc(features)
        
        return output

class EnhancedSecureModel(nn.Module):
    def __init__(self, input_size=1000, key_size=100, num_kernels=24):
        super().__init__()
        self.network = EnhancedSecureBiometricECGModel(input_size, key_size, num_kernels)
    
    def __call__(self, xA, xP, xN, k1, k2):
        output_a = self.network(xA, k1)
        output_p1 = self.network(xP, k1)
        output_p2 = self.network(xP, k2)
        output_n1 = self.network(xN, k1)
        output_n2 = self.network(xN, k2)
        return output_a, output_p1, output_p2, output_n1, output_n2
    
    def get_embedding(self, x, k):
        return self.network(x, k)

def generate_key(key_size=100):
    key = mx.random.randint(0, 2, (1, key_size), dtype=mx.float32)
    return key / mx.linalg.norm(key)

class SecureTripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    
    def __call__(self, anchor, positive1, positive2, negative1, negative2):
        dist_p1 = mx.linalg.norm(anchor - positive1, axis=1)
        dist_p2 = mx.linalg.norm(anchor - positive2, axis=1)
        dist_n1 = mx.linalg.norm(anchor - negative1, axis=1)
        dist_n2 = mx.linalg.norm(anchor - negative2, axis=1)
        losses = mx.maximum(dist_p1 - dist_n1 + self.margin, 0) + \
                 mx.maximum(dist_p1 - dist_n2 + self.margin, 0) + \
                 mx.maximum(dist_p2 - dist_n1 + self.margin, 0) + \
                 mx.maximum(dist_p2 - dist_n2 + self.margin, 0)
        return losses.mean()

def train(model, optimizer, criterion, num_epochs, batch_size, input_size, key_size):
    for epoch in range(num_epochs):
        for batch in range(num_batches):  # You'll need to define num_batches based on your dataset
            # Load batch data (replace with your actual data loading)
            xA = mx.random.normal((batch_size, 1, input_size))
            xP = mx.random.normal((batch_size, 1, input_size))
            xN = mx.random.normal((batch_size, 1, input_size))
            k1 = generate_key(key_size)
            k2 = generate_key(key_size)

            # Compute loss and gradients
            loss, grads = loss_and_grad_fn(model, xA, xP, xN, k1, k2)

            # Update the model
            optimizer.update(model, grads)

            # Evaluate the new parameters and optimizer state
            mx.eval(model.parameters(), optimizer.state)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss}")

def main():
    # Hyperparameters
    input_size = 1000
    key_size = 100
    num_kernels = 24
    learning_rate = 0.001
    num_epochs = 10
    batch_size = 32

    # Initialize model, criterion, and optimizer
    model = EnhancedSecureModel(input_size, key_size, num_kernels)
    criterion = SecureTripletLoss()
    optimizer = optim.AdamW(learning_rate=learning_rate)

    # Force initialization of model parameters
    mx.eval(model.parameters())

    # Define loss and gradient function
    def loss_fn(model, xA, xP, xN, k1, k2):
        outputs = model(xA, xP, xN, k1, k2)
        return criterion(*outputs)

    global loss_and_grad_fn
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # Train the model
    train(model, optimizer, criterion, num_epochs, batch_size, input_size, key_size)

    # Save the model (optional)
    mx.save("heartid_model.npz", model.parameters())

    # Save the optimizer state (optional)
    from mlx.utils import tree_flatten
    state = tree_flatten(optimizer.state)
    mx.save_safetensors("optimizer.safetensors", dict(state))

if __name__ == "__main__":
    main()
