# Generate synthetic data
def generate_synthetic_data(gbm, n_samples, label):
    with torch.no_grad():
        z = torch.randn(n_samples, latent_size).to(device)
        y = torch.full((n_samples, 1), label, dtype=torch.float32).to(device)
        synthetic_data = cvae.decoder(torch.cat([z, y], dim=1)).cpu().numpy()
    
    return synthetic_data

# Generate synthetic data for both labels
n_samples = 10000
synthetic_data_pd1 = generate_synthetic_data(cvae, n_samples, 1)
synthetic_data_pd0 = generate_synthetic_data(cvae, n_samples, 0)

# Convert the generated data back to the original scale
synthetic_data_pd1 = scaler.inverse_transform(synthetic_data_pd1)
synthetic_data_pd0 = scaler.inverse_transform(synthetic_data_pd0)

# Create pandas DataFrames for the generated data
synthetic_data_pd1_df = pd.DataFrame(synthetic_data_pd1, columns=original_columns)
synthetic_data_pd0_df = pd.DataFrame(synthetic_data_pd0, columns=original_columns)

# Print results
print("Synthetic data with pd=1:")
print(synthetic_data_pd1_df.head())
print("\nSynthetic data with pd=0:")
print(synthetic_data_pd0_df.head())

#evaluate quality of synthetic data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def extract_latent_variables(model, dataloader):
    model.eval()
    latent_vars = []
    labels_list = []

    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            h = model.encoder(torch.cat([data, labels.unsqueeze(-1)], dim=1))
            mu, _ = h.chunk(2, dim=1)
            latent_vars.extend(mu.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    return np.array(latent_vars), np.array(labels_list)

train_latent_vars, train_labels = extract_latent_variables(cvae, train_loader)
test_latent_vars, test_labels = extract_latent_variables(cvae, test_loader)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(train_latent_vars, train_labels)

# Evaluate the classifier
train_preds = rf_classifier.predict(train_latent_vars)
test_preds = rf_classifier.predict(test_latent_vars)

train_accuracy = accuracy_score(train_labels, train_preds)
test_accuracy = accuracy_score(test_labels, test_preds)

print(f'Training accuracy: {train_accuracy:.4f}')
print(f'Test accuracy: {test_accuracy:.4f}')

import numpy as np
from sklearn.metrics import accuracy_score

# Combine synthetic data and create labels
synthetic_data = np.vstack([synthetic_data_pd1, synthetic_data_pd0])
synthetic_labels = np.hstack([np.ones(n_samples), np.zeros(n_samples)])

# Extract latent variables for the synthetic data
synthetic_latent_vars, _ = extract_latent_variables(cvae, DataLoader(TensorDataset(torch.tensor(synthetic_data, dtype=torch.float32).to(device), torch.tensor(synthetic_labels, dtype=torch.float32).to(device)), batch_size=32, shuffle=False))

# Make predictions using the Random Forest Classifier
synthetic_predictions = rf_classifier.predict(synthetic_latent_vars)

# Calculate accuracy
synthetic_accuracy = accuracy_score(synthetic_labels, synthetic_predictions)

print("Synthetic data accuracy: {:.4f}".format(synthetic_accuracy))
