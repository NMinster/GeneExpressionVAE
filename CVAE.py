import numpy as np
from sklearn.metrics import precision_recall_curve, auc, f1_score, precision_score, recall_score
from tqdm import tqdm

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Convert df to tensors
train_data_resampled_tensor = torch.tensor(train_data_resampled.values, dtype=torch.float32).to(device)
train_labels_resampled_tensor = torch.tensor(train_labels_resampled.values, dtype=torch.float32).to(device)
minority_train_data_tensor = torch.tensor(minority_train_data.values, dtype=torch.float32).to(device)

# Create datasets and data loaders
train_dataset = TensorDataset(train_data_resampled_tensor, train_labels_resampled_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Convert pandas DataFrame to PyTorch tensors
test_data_tensor = torch.tensor(test_data.values, dtype=torch.float32).to(device)
test_labels_tensor = torch.tensor(test_labels, dtype=torch.float32).to(device)

# Create datasets and data loaders
test_dataset = TensorDataset(test_data_tensor, test_labels_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define CVAE model
class CVAE(nn.Module):
    def __init__(self, input_size, label_size, hidden_size, latent_size):
        super(CVAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size + label_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),  # Additional layer
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, latent_size * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_size + label_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),  # Additional layer
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, input_size),
            nn.Tanh()
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps *std

    def forward(self, x, y):
        y = y.view(-1, 1)  # Reshape the labels tensor
        xy = torch.cat([x, y], dim=1)
        h = self.encoder(xy)
    
        mu, logvar = h.chunk(2, dim=1)
        z = self.reparameterize(mu, logvar)
        zy = torch.cat([z, y], dim=1)
    
        return self.decoder(zy), mu, logvar


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# Initialize CVAE
input_size = train_data_resampled.shape[1]
label_size = 1
hidden_size = 64 
latent_size = 512

cvae = CVAE(input_size, label_size, hidden_size, latent_size).to(device)
cvae.apply(init_weights)

# Set loss function and optimizer
reconstruction_loss = nn.MSELoss(reduction='sum')
optimizer = optim.RMSprop(cvae.parameters(), lr=1.1779e-4)  # Changed optimizer and learning rate


def evaluate(model, dataloader, threshold):
    model.eval()
    labels_list = []
    predictions_list = []

    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            recon_data, _, _ = model(data, labels)  # Unpack the returned tuple
            
            # Calculate reconstruction errors
            recon_error = torch.mean((data - recon_data) ** 2, dim=1)
            # Classify data points based on the threshold
            predicted = (recon_error > threshold).int()
            
            labels_list.extend(labels.cpu().numpy())
            predictions_list.extend(predicted.cpu().numpy())

    return np.array(labels_list), np.array(predictions_list)


# Calculate reconstruction errors on the training data
train_errors = []
with torch.no_grad():
    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)
        recon_data, _, _ = cvae(data, labels)
        recon_error = torch.mean((data - recon_data) ** 2, dim=1)
        train_errors.extend(recon_error.cpu().numpy())

# Set the threshold at the 90th percentile of the training errors
threshold = np.percentile(train_errors, 90)



# Train the CVAE 
num_epochs = 50

for epoch in range(num_epochs):
    # Training step
    cvae.train()
    # Add a progress bar
    for batch_idx, (data, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
        data, labels = data.to(device), labels.to(device)

        optimizer.zero_grad()

        recon_data, mu, logvar = cvae(data, labels.unsqueeze(-1))
        loss = reconstruction_loss(recon_data, data)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss += kl_div

        loss.backward()
        torch.nn.utils.clip_grad_norm_(cvae.parameters(), max_norm=1.0)
        optimizer.step()

    # Calculate reconstruction errors on the training data
    train_errors = []
    with torch.no_grad():
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            recon_data, _, _ = cvae(data, labels)
            recon_error = torch.mean((data - recon_data) ** 2, dim=1)
            train_errors.extend(recon_error.cpu().numpy())


    threshold = np.percentile(train_errors, 80)

    # Evaluation step
    test_labels, test_predictions = evaluate(cvae, test_loader, threshold)
    precision = precision_score(test_labels, test_predictions)
    recall = recall_score(test_labels, test_predictions)
    f1 = f1_score(test_labels, test_predictions)
    precision_recall = precision_recall_curve(test_labels, test_predictions)
    auprc = auc(precision_recall[1], precision_recall[0])

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 score: {f1:.4f}, AUPRC: {auprc:.4f}')
