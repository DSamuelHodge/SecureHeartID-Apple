import torch
import torch.nn as nn
import torch.optim as optim

class SecureBiometricECGModel(nn.Module):
    def __init__(self, input_size=1000, key_size=100, dropout_rate=0.5):
        super(SecureBiometricECGModel, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),

            nn.Conv1d(16, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),

            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),

            nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        self.feature_size = self._get_conv_output(input_size)

        self.dropout = nn.Dropout(p=dropout_rate)

        self.fc_layers = nn.Sequential(
            nn.Linear(self.feature_size + key_size, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU()
        )

    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, 1, shape))
        output_feat = self.conv_layers(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    def forward(self, x, key):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = torch.cat((x, key), dim=1)
        x = self.fc_layers(x)
        return x

class SecureModel(nn.Module):
    def __init__(self, input_size=1000, key_size=100, dropout_rate=0.5):
        super(SecureModel, self).__init__()
        self.network = SecureBiometricECGModel(input_size, key_size, dropout_rate)

    def forward(self, xA, xP, xN, k1, k2):
        output_a = self.network(xA, k1)
        output_p1 = self.network(xP, k1)
        output_p2 = self.network(xP, k2)
        output_n1 = self.network(xN, k1)
        output_n2 = self.network(xN, k2)
        return output_a, output_p1, output_p2, output_n1, output_n2

    def get_embedding(self, x, k):
        return self.network(x, k)

# Helper function to generate random keys
def generate_key(key_size=100):
    key = torch.randint(0, 2, (1, key_size), dtype=torch.float32)
    return key / torch.norm(key)

# Secure Triplet Loss function
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

# Training function
def train_model(model, train_loader, val_loader, num_epochs=250, patience=25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = SecureTripletLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for anchor, positive, negative in train_loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            key1 = generate_key().to(device)
            key2 = generate_key().to(device)

            output_a, output_p1, output_p2, output_n1, output_n2 = model(anchor, positive, negative, key1, key2)

            loss = criterion(output_a, output_p1, output_p2, output_n1, output_n2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for anchor, positive, negative in val_loader:
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

                key1 = generate_key().to(device)
                key2 = generate_key().to(device)

                output_a, output_p1, output_p2, output_n1, output_n2 = model(anchor, positive, negative, key1, key2)

                loss = criterion(output_a, output_p1, output_p2, output_n1, output_n2)
                val_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

# Usage example
input_size = 1000  # 5 seconds of ECG at 200 Hz
key_size = 100
dropout_rate = 0.5
model = SecureModel(input_size, key_size, dropout_rate)

# Assuming you have your data loaders set up
# train_loader = ...
# val_loader = ...

# Train the model
train_model(model, train_loader, val_loader)