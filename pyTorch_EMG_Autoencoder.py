# -*- coding: utf-8 -*-
import numpy as np
import os
import kagglehub
import pandas as pd
import torch
import math
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

Train = False # If set to True, runs convolution, if set to False jumps to loading last model generated

# =============================================================================
# Download the Dataset
# =============================================================================
# Download latest version of the UCI EMG Signal for Gesture Recognition Dataset
path = kagglehub.dataset_download("sojanprajapati/emg-signal-for-gesture-recognition")

print("Path to dataset files:", path)

# Just dont add another csv to this file and we're good...
# Also the file is a quarter gig so make sure to delete after this project...
for fname in os.listdir(path):
    if fname.endswith(".csv"):
        fpath = os.path.join(path, fname)
        data = pd.read_csv(fpath)

# =============================================================================
# Remove the Dataset
# =============================================================================
# Removes the dataset file and upstreme directories. Do not run if you have other datasets from kaggle uploaded by sojanprajapati.
# if os.path.exists(path):
#     parent = os.path.join(path, *([os.pardir])*4)
#     os.remove(parent)
#     print(f"{parent} has been removed.")
# else:
#     print(f"The file {path} does not exist.")



# =============================================================================
# Separate the Data into Training, Validation, and Testing Sets
# =============================================================================

# Plan: Test autoencoder on subjects 1-30 test 1 both gestures and test 2 first gesture
# Validate on 1-30 test 2 second gesture, final eval on subjects 31-36

# Determine stretches of data by class shift.
# Since each gesture is separated by class 0 this allows for determination of whether the gesture observed
# Is the individual's first, second, etc. time performing the gesture
df_1 = data.copy()
df_1["class_changed"] = df_1.groupby("label")["class"].transform(lambda x: x.ne(x.shift()))
df_1["segment_id"] = df_1.groupby("label")["class_changed"].cumsum()

# Remove resting state and gesture 7 as gesture 7 is missing from some tests
gesture_df = df_1[(df_1["class"] != 0) & (df_1["class"] != 7)].copy()

# Rank by segment_id in ascending order to split gesture performance into training vs validation data
gesture_df["series"] = gesture_df.groupby(["label", "class"])["segment_id"].transform(
    lambda x: pd.factorize(x, sort=True)[0] + 1)

# Filter to series 1-3 and subjects 1–30
train_df = gesture_df[(gesture_df["series"].isin([1,2,3])) & (gesture_df["label"] < 31)].copy()

# Filter to series 4 and subjects 1-30
val_df = gesture_df[(gesture_df["series"] == 4) & (gesture_df["label"] < 31)].copy()

# Filter to subjects 31-36
test_df = gesture_df[(gesture_df["label"] >= 31)].copy()


# Extracts center window_size of the group. Please note the floor operator // may cause unexpected windows with odd numbers so they are precluded
def center_window(group, window_size):
    assert window_size % 2 == 0, "window_size must be even"
    n = len(group)
    mid = n // 2
    half = window_size // 2
    return group.iloc[mid - half : mid + half]

# Create empy results dict. Loop through classes and group by label and segment_id. Apply center_window to each group
# Stich data back together and put in a dict where result[id] stores the centered data for that gesture id
def split_by_gesture(df, window_size):
    result = {}
    for gesture_class in sorted(df["class"].unique()):
        gesture_df = df[df["class"] == gesture_class]
        trimmed = gesture_df.groupby(["label", "segment_id"], group_keys=False).apply(
            center_window, window_size=window_size, include_groups=False)
        result[gesture_class] = trimmed.reset_index(drop=True)
    return result

# List of channels for later use in pyTorch
channels = ["channel1", "channel2", "channel3", "channel4", "channel5", "channel6", "channel7", "channel8"]

# Creating gesture seperated and windowed datasets
train_windows = split_by_gesture(train_df, window_size=400)
val_windows   = split_by_gesture(val_df,   window_size=400)
test_windows  = split_by_gesture(test_df,  window_size=400)


# =============================================================================
# pyToch Setup
# =============================================================================
class EMGGestureDataset(Dataset):
    def __init__(self, windowed_df, gesture_class, channels, mean=None, std=None):
        super().__init__()
        
        self.gesture_class = gesture_class
        self.channels = channels

        # Convert to float 32
        data = windowed_df[channels].values.astype(np.float32)
        
        # Calculate the mean and std if it is a training set, otherwise pull from provided mean and std
        if mean is None and std is None:
            self.mean = data.mean(axis = 0)
            self.std = data.std(axis = 0)
        else:
            self.mean = mean
            self.std = std
            
        # Normalize
        self.data = (data - self.mean) / (self.std + 1e-8)
        
            
        # Reshape into 3D array split by window
        self.data = self.data.reshape(-1, 400, 8)
    
    # Loads and returns a sample from the dataset at the given index
    def __getitem__(self, idx):
        x = self.data[idx] # shape (400, 8)
        x = torch.tensor(x)
        return x.permute(1, 0)  # returns one (8, 400) block
    
    # The __len__ function returns the number of samples in the datset, previously set as the first dimension of self.date
    def __len__(self):
        return self.data.shape[0]
    
    
# =============================================================================
# Create pyToch Datasets and run ICA
# =============================================================================

# Training datasets — compute mean and std from training data
train_datasets = {g: EMGGestureDataset(train_windows[g], g, channels) for g in range(1, 7)}

# Val and test — pass in training statistics for consistent normalization
val_datasets = {g: EMGGestureDataset(val_windows[g],  g, channels, mean=train_datasets[g].mean, std=train_datasets[g].std) for g in range(1, 7)}

test_datasets = {g: EMGGestureDataset(test_windows[g], g, channels, mean=train_datasets[g].mean, std=train_datasets[g].std) for g in range(1, 7)}


# =============================================================================
# Convolution
# =============================================================================
# 4x reduction Convolution1D encoder
class EMGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=7, padding=3), # 8 channels to 16 features
            nn.ReLU(),
            nn.MaxPool1d(2), # Halves time dimension
            nn.Conv1d(16, 32, kernel_size=5, padding=2), # 16 Features to 32
            nn.ReLU(),
            nn.MaxPool1d(2), # Halves time dimension
            nn.Conv1d(32, 32, kernel_size=3, padding=1), # 32 Features to 16
            nn.ReLU(),
            nn.MaxPool1d(4), # quarters time dimension
        )
    def forward(self, x):
        return self.encoder(x)

# Decoder
class EMGDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=4),
            nn.ConvTranspose1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(32, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(16, 8, kernel_size=7, padding=3),
            # No ReLu since it would cut off negative outputs
        )
    def forward(self, x):
        return self.decoder(x)

# Autoencoder model
class EMGAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = EMGEncoder()
        self.decoder = EMGDecoder()
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch, X in enumerate(dataloader):
        X = X.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, X)
        
        loss.backward()
        optimizer.step()

def val_loop(dataloader, model, loss_fn):
    model.eval()
    num_batches = len(dataloader)
    val_loss = 0


    with torch.no_grad():
        for X in dataloader:
            X = X.to(device)
            pred = model(X)
            val_loss += loss_fn(pred, X).item()

    val_loss /= num_batches
    return val_loss

# =============================================================================
# Running the Training and Evaluating
# =============================================================================
batch_size = 9

train_loaders = {g: DataLoader(train_datasets[g], batch_size=batch_size, shuffle=True)  for g in range(1, 7)}
val_loaders   = {g: DataLoader(val_datasets[g],   batch_size=batch_size, shuffle=False) for g in range(1, 7)}
test_loaders  = {g: DataLoader(test_datasets[g],  batch_size=batch_size, shuffle=False) for g in range(1, 7)}
test_loop = val_loop
models = {}
test_results = {}

device = torch.device("cpu")
print(f"Using device: {device}")

if Train:
    for g in range(1, 7):
        model = EMGAutoencoder().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = F.mse_loss
        epochs = 500
        
        print(f"\n{'='*40}")
        print(f"Training Gesture {g}")
        print(f"{'='*40}")
        
        # Training loop — uses train and val loaders
        for t in range(epochs):
            # print(f"Gesture {g} | Epoch {t+1}")
            train_loop(train_loaders[g], model, loss_fn, optimizer)
            val_loss = val_loop(val_loaders[g], model, loss_fn)
            
            if (t + 1) % 100 == 0:
                print(f"Epoch {t+1:>3}/{epochs} | Val loss: {val_loss:.6f}")
        
       # Final test evaluation
        test_loss = test_loop(test_loaders[g], model, loss_fn)
        models[g] = model
        test_results[g] = test_loss
        print(f"  Final test loss: {test_loss:.6f}")
    
    # Summary
    print(f"\n{'='*40}")
    print("Final Test Results")
    print(f"{'='*40}")
    for g in range(1, 7):
        print(f"  Gesture {g}: {test_results[g]:.6f}")
    
    # Saving the models
    os.makedirs("models", exist_ok=True)
    for g in range(1, 7):
        torch.save(models[g].state_dict(), f"models/gesture_{g}.pth")
    print("Models saved.")


# =============================================================================
# Evaluating/Testing the Model
# =============================================================================
if not Train:
    # Reloding the models
    models = {}
    for g in range(1, 7):
        model = EMGAutoencoder().to(device)
        try:
            model.load_state_dict(torch.load(f"models/gesture_{g}.pth"))
        except:
            print("Model data does not exist")
            break
        model.eval()
        models[g] = model
    
    # Get the error of the model ran against it's own training data for normalization
    baseline_errors = {}
    for g, model in models.items():
        model.eval()
        total_error = 0
        num_batches = 0
        with torch.no_grad():
            for batch in train_loaders[g]:
                batch = batch.to(device)
                pred = model(batch)
                total_error += F.mse_loss(pred, batch).item()
                num_batches += 1
        baseline_errors[g] = total_error / num_batches
    
    print("Baseline errors per gesture:")
    for g, err in baseline_errors.items():
        print(f"  Gesture {g}: {err:.6f}")
    
    # Reset test_loaders to not be batched, allowing for better confusion matrix resolution
    batch_size=1
    test_loaders  = {g: DataLoader(test_datasets[g],  batch_size=batch_size, shuffle=False) for g in range(1, 7)}
    
    true_labels = []
    predicted_labels = []
    
    # Create confusion matrix normalizing error with self-baseline error
    for true_g in range(1, 7):
        for batch in test_loaders[true_g]:
            batch = batch.to(device)
            best_error = math.inf
            predicted_gesture = None
            
            for g, model in models.items():
                with torch.no_grad():
                    pred = model(batch)
                    error = F.mse_loss(pred, batch).item()/baseline_errors[g]
                    if error < best_error:
                        best_error = error
                        predicted_gesture = g
            
            true_labels.append(true_g)
            predicted_labels.append(predicted_gesture)
    
    cm = np.zeros((6, 6), dtype=int)
    
    # If prediction is the same as the true value add one to that cell in the 6x6 confusion matrix
    for true_g, pred_g in zip(true_labels, predicted_labels):
        cm[true_g - 1][pred_g - 1] += 1
        
    # Diag/Total of cm matrix gives accuracy percentage
    accuracy = cm.trace()/cm.sum()
    
    # Reframing to pandas df to simplify printing of cm matrix
    col_names = ["Pred1",  "Pred2",  "Pred3",  "Pred4",  "Pred5",  "Pred6"]
    row_names = ["True 1",  "True 2",  "True 3",  "True 4",  "True 5",  "True 6"]
    df = pd.DataFrame(cm, index=row_names, columns=col_names)
    
    print(f"Overall Accuracy: {accuracy:.1%}\n")
    print("Confusion Matrix:")
    print(df)
    
    # flattened_encode = []
    # encoded_gesture = []
    # for g in range(1, 7):
    #     for batch in test_loaders[g]:
    #         with torch.no_grad():
    #             encoded = models[g].encoder(batch)
    #             encoded_flat = encoded.view(encoded.size(0), -1)
    #             flattened_encode.append(encoded_flat)
    #             encoded_gesture.append(g)
    
    # bottleneck_matrix = np.vstack([v.cpu().numpy() for v in flattened_encode])
    # bottleneck_labels = np.array(encoded_gesture)
    
    # pca_full = PCA()
    # pca_full.fit(bottleneck_matrix)
    
    # cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    
    # # How many components to reach 80% and 90% variance explained?
    # n_80 = np.argmax(cumulative_variance >= 0.80) + 1
    # n_90 = np.argmax(cumulative_variance >= 0.90) + 1
    
    # print(f"Components needed for 80% variance: {n_80}")
    # print(f"Components needed for 90% variance: {n_90}")
    