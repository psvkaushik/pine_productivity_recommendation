# Necessary imports

import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.distributions import Normal
from tqdm import tqdm
import yaml
# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load Dara

train_data = pd.read_csv("../../data_ash/tr_data.csv")
validation_data = pd.read_csv("../../data_ash/te_data.csv")

# We will use the untouched test data for final evaluation
# It is included in this script for grading purposes
untouched_test_data = pd.read_csv("../../data_ash/test_data.csv")

# Process the data

drop_cols = [
    'Unnamed: 0', "env", "TestId", "latitude",
    "longitude", "date_initial", "date_final", "Feature"
]
target_class_col = 'Specie'
target_reg_col   = 'Productivity (y)'

feature_cols = (
    train_data
    .drop(columns=drop_cols + [target_class_col, target_reg_col], errors='ignore')
    .select_dtypes(include=[np.number])
    .columns
    .tolist()
)

# encode the species data
le = LabelEncoder()
train_data['species_encoded'] = le.fit_transform(train_data[target_class_col])
validation_data['species_encoded'] = le.transform(validation_data[target_class_col])
untouched_test_data['species_encoded'] = le.transform(untouched_test_data[target_class_col])

X_train = train_data[feature_cols].fillna(0).values
y_species_train = train_data['species_encoded'].values
y_prod_train    = train_data[target_reg_col].values

X_test  = validation_data[feature_cols].fillna(0).values
y_species_test = validation_data['species_encoded'].values
y_prod_test    = validation_data[target_reg_col].values

X_untouched  = untouched_test_data[feature_cols].fillna(0).values
y_species_untouched = untouched_test_data['species_encoded'].values
y_prod_untouched   = untouched_test_data[target_reg_col].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
X_untouched_scaled  = scaler.transform(X_untouched)

# Prepare the dataset
class PineDataset(Dataset):
    def __init__(self, X, y_species, y_productivity):
        self.X   = torch.tensor(X, dtype=torch.float32)
        self.yc  = torch.tensor(y_species, dtype=torch.long)
        self.yr  = torch.tensor(y_productivity, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.yc[idx], self.yr[idx]

train_loader = DataLoader(PineDataset(X_train_scaled, y_species_train, y_prod_train),
                          batch_size=512, shuffle=True)
test_loader  = DataLoader(PineDataset(X_test_scaled,  y_species_test,  y_prod_test),
                          batch_size=512)
untouched_loader  = DataLoader(PineDataset(X_untouched_scaled,  y_species_untouched,  y_prod_untouched),
                          batch_size=512)

# Define the Feature Attention Block
class FeatureAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        # normalize before computing attention scores
        x_norm = self.norm(x)
        scores = self.attn(x_norm)         # (batch, 1)
        return x + x * scores              

# Using MLP with residual features
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.fc1   = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2   = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.fc3   = nn.Linear(hidden_dim, hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.act   = nn.ReLU()

    def forward(self, x):
        x1 = self.act(self.norm1(self.fc1(x)))
        x2 = self.act(self.norm2(self.fc2(x1)))
        x3 = self.fc3(x2)
        return self.act(self.norm3(x3 + x1))  # residual connection + norm

class SpeciesProductivityModel(nn.Module):
    def __init__(self, input_dim, num_classes,
                 hidden_dim=128,
                 use_feature_attention=True,
                 use_classifier_inputs=True):
        super().__init__()
        self.fa = FeatureAttention(input_dim) if use_feature_attention else nn.Identity()
        self.shared = MLP(input_dim, hidden_dim)

        # classification head
        self.classifier = nn.Linear(hidden_dim, num_classes)

        # regression heads
        reg_dim = hidden_dim + (num_classes if use_classifier_inputs else 0)
        self.reg_mean   = nn.Linear(reg_dim, 1)
        self.reg_logvar = nn.Linear(reg_dim, 1)
        self.use_classifier_inputs = use_classifier_inputs

    def forward(self, x):
        x = self.fa(x)
        h = self.shared(x)
        logits = self.classifier(h)

        if self.use_classifier_inputs:
            probs = F.softmax(logits, dim=-1)
            reg_in = torch.cat([h, probs], dim=-1)
        else:
            reg_in = h

        mu     = self.reg_mean(reg_in).squeeze(1)
        logvar = self.reg_logvar(reg_in).squeeze(1)
        sigma  = torch.exp(0.5 * logvar)
        return logits, mu, logvar, sigma

# Create the loss functions
def gaussian_nll(mu, logvar, y):
    """
    Gaussian negative log-likelihood loss function for the regression task.
    """

    inv_var = torch.exp(-logvar)
    return 0.5 * (logvar + (y - mu)**2 * inv_var + math.log(2 * math.pi))

def gaussian_crps(mu, sigma, y):
    """
    Continuous Ranked Probability Score (CRPS) for evaluating the regression task.
    """
    z   = (y - mu) / sigma
    pdf = torch.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)
    cdf = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
    return sigma * (z * (2 * cdf - 1) + 2 * pdf - 1 / math.sqrt(math.pi))

def interval_metrics(mu, sigma, y, alpha=0.05):
    """
    Interval metrics for evaluating the regression task.
    """

    normal = Normal(0, 1)
    k      = normal.icdf(torch.tensor(1 - alpha/2, device=mu.device))
    lower  = mu - k * sigma
    upper  = mu + k * sigma
    inside = ((y >= lower) & (y <= upper)).float()
    return inside.mean().item(), (upper - lower).mean().item()

# Define the training and evaluation functions
def train_model(model, loader, epochs=50, cls_weight=1.0, reg_weight=1.0):
    model.to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = GradScaler()

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        for xb, ycls, yreg in loader:
            xb, ycls, yreg = xb.to(device), ycls.to(device), yreg.to(device)
            opt.zero_grad()
            with autocast():
                logits, mu, logvar, sigma = model(xb)
                loss_c = F.cross_entropy(logits, ycls)
                loss_r = gaussian_nll(mu, logvar, yreg).mean()
                loss   = cls_weight * loss_c + reg_weight * loss_r

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running_loss += loss.item()

        sched.step()
        avg = running_loss / len(loader)
        print(f"Epoch {epoch:02d}/{epochs} — loss: {avg:.4f}")

    return model

def evaluate_model(model, loader, alpha=0.05):
    model.eval()
    all_true_c, all_pred_c = [], []
    all_mu, all_sigma, all_true_r = [], [], []
    top3_ok, total = 0, 0

    with torch.no_grad():
        for xb, ycls, yreg in loader:
            xb, ycls, yreg = xb.to(device), ycls.to(device), yreg.to(device)
            logits, mu, logvar, sigma = model(xb)

            preds = logits.argmax(dim=1)
            all_true_c.extend(ycls.cpu().numpy())
            all_pred_c.extend(preds.cpu().numpy())

            top3 = torch.topk(logits, 3, dim=1).indices
            for t, p in zip(ycls.cpu(), top3.cpu()):
                if t.item() in p.tolist():
                    top3_ok += 1
            total += ycls.size(0)

            all_mu.extend(mu.cpu().numpy())
            all_sigma.extend(sigma.cpu().numpy())
            all_true_r.extend(yreg.cpu().numpy())

    # Calculate all required metrics
    acc   = accuracy_score(all_true_c, all_pred_c)
    top3  = top3_ok / total
    
    mse = mean_squared_error(all_true_r, all_mu)
    mae = mean_absolute_error(all_true_r, all_mu)
    r2  = r2_score(all_true_r, all_mu)

    mu_t    = torch.tensor(all_mu)
    sigma_t = torch.tensor(all_sigma)
    y_t     = torch.tensor(all_true_r)
    crps    = gaussian_crps(mu_t, sigma_t, y_t).mean().item()


    picp, mpiw = interval_metrics(mu_t, sigma_t, y_t, alpha)
    lvl = int((1 - alpha) * 100)

    # Print the results
    metrics = {
    "Species Acc":            acc,
    "Species Top-3 Acc":      top3,
    "Prod. MSE":              mse,
    "Prod. MAE":              mae,
    "Prod. R²":               r2,
    "Prod. CRPS":             crps,
    f"{lvl}% PICP":           picp,
    f"{lvl}% MPIW":           mpiw,
    }


    df_metrics = pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"])
    print(df_metrics.to_markdown()) 


input_dim   = X_train_scaled.shape[1]
num_classes = len(le.classes_)

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Extract settings
epochs               = cfg["training"]["epochs"]
use_feature_attention = cfg["model"]["use_feature_attention"]
use_classifier_inputs = cfg["model"]["use_classifier_inputs"]
hidden_dim           = cfg["model"]["hidden_dim"]
cls_weight           = cfg["training"].get("cls_weight", 1.0)
reg_weight           = cfg["training"].get("reg_weight", 0.5)
alpha               = cfg["evaluation"].get("alpha", 0.05)
model = SpeciesProductivityModel(
    input_dim, num_classes,
    hidden_dim=hidden_dim,
    use_feature_attention=use_feature_attention,
    use_classifier_inputs=use_classifier_inputs
)

train_model(model, train_loader, epochs=epochs, cls_weight=1.0, reg_weight=0.5)
evaluate_model(model, test_loader, alpha=alpha)
print("--"* 20)
print("Final evaluation on untouched test data")
evaluate_model(model, untouched_loader, alpha=0.05)