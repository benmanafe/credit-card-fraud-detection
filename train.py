import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import pickle

# --- 1. MODEL DEFINITION ---
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 20), nn.Tanh(),
            nn.Linear(20, 14), nn.Tanh(),
            nn.Linear(14, 8), nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 14), nn.Tanh(),
            nn.Linear(14, 20), nn.Tanh(),
            nn.Linear(20, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# --- 2. LOAD & PROCESS DATA ---
print("Loading data...")
df = pd.read_csv('creditcard.csv') 

scaler_amount = RobustScaler()
scaler_time = RobustScaler()

df['scaled_amount'] = scaler_amount.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_time'] = scaler_time.fit_transform(df['Time'].values.reshape(-1, 1))
df.drop(['Time', 'Amount'], axis=1, inplace=True)

# Train on Normal data only
normal_df = df[df['Class'] == 0]
X_normal = normal_df.drop(['Class'], axis=1).values
X_train, X_test = train_test_split(X_normal, test_size=0.2, random_state=42)

X_train_tensor = torch.FloatTensor(X_train)

# --- 3. TRAIN MODEL ---
print("Training model...")
input_dim = X_train.shape[1] # 30
model = Autoencoder(input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 20
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train_tensor), batch_size=256, shuffle=True)

for epoch in range(num_epochs):
    for batch in train_loader:
        inputs = batch[0]
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()

# --- 4. CALCULATE THRESHOLD ---
print("Calculating threshold...")
model.eval()
with torch.no_grad():
    reconstructions = model(X_train_tensor)
    train_loss = torch.mean((X_train_tensor - reconstructions) ** 2, dim=1)
    
threshold = np.percentile(train_loss.numpy(), 99)

# --- 5. SAVE ARTIFACTS ---
print("Saving artifacts to 'fraud_detector.pth'...")
artifacts = {
    'model_state': model.state_dict(),
    'scaler_amount': scaler_amount,
    'scaler_time': scaler_time,
    'threshold': threshold,
    'input_dim': input_dim,
    'feature_names': [f'V{i}' for i in range(1, 29)] + ['Scaled_Amount', 'Scaled_Time']
}

with open('fraud_detector.pth', 'wb') as f:
    pickle.dump(artifacts, f)

print("Done! Ready for Streamlit.")