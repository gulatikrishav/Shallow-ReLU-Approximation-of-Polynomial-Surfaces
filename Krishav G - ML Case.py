import numpy as np              # numerical arrays, random sampling
import torch                    # machine learning library
import torch.nn as nn           # neural network layers & loss functions
import torch.optim as optim     # optimizers (Adam, SGD, etc.)
import matplotlib.pyplot as plt # plotting
import pandas as pd             # tabular results (nice summary tables)
from pathlib import Path        # file paths (for saving CSV)

# The true function we want to learn; trying to approximate with a neural network.
def f_true(xy, p):
    x = xy[:,0]; y = xy[:,1]
    # Returns a length-N vector of clean target values
    return (x**2 + x*y + 2*y**2)**p

# The model: a shallow ReLU neural network
# ReLU: Outputs the input value if it is positive and zero if it is negative
class ShallowReLU(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.hidden = nn.Linear(2, hidden)  # 2 inputs (x,y) -> M hidden neurons
        self.out    = nn.Linear(hidden, 1)  # M -> 1 output (prediction)
        nn.init.kaiming_normal_(self.hidden.weight, nonlinearity='relu')
        nn.init.zeros_(self.hidden.bias)
        nn.init.zeros_(self.out.bias)

    # Applies ReLU and sums hidden activations into one scalar output
    # Each hidden neuron is a kinked line (ReLU “hinge”) in 2D; the output layer adds them up to form a flexible surface
    def forward(self, x):
        return self.out(torch.relu(self.hidden(x)))

# Training Model for a given p (which true func. trying to approximate) and M (how large neural network is)
def train_once(p, M, N=800, epochs=400, lr=1e-2, sigma=0.03, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.uniform(-1, 1, (N, 2)).astype(np.float32)  # sample N points in [-1,1]^2
        y_clean = f_true(X, p).astype(np.float32)  # true function values
        y = y_clean + rng.normal(0, sigma, N).astype(np.float32)  # add mean-0 noise

        Xt = torch.tensor(X)  # -> torch tensors (required by PyTorch)
        yt = torch.tensor(y.reshape(-1, 1))

        model = ShallowReLU(M)
        opt = optim.Adam(model.parameters(), lr=lr)  # Adam optimizer
        loss_fn = nn.MSELoss()  # Mean Squared Error

        # Training Loop
        losses = []
        for _ in range(epochs):  # = number of passes over the whole dataset
            opt.zero_grad()  # clear old gradients
            pred = model(Xt)  # forward pass: p(x,y)
            L = loss_fn(pred, yt)  # compute MSE vs (noisy) targets
            L.backward()  # backprop: compute gradients dL/d(parameters)
            opt.step()  # gradient-based parameter update
            losses.append(float(L.item()))

        # Check how well it fit the training data
        with torch.no_grad():
            pred_train = model(Xt).numpy().reshape(-1)
        tr_noisy = float(((pred_train - y) ** 2).mean())  # vs noisy targets
        tr_clean = float(((pred_train - y_clean) ** 2).mean())  # vs true clean f
        # Return results
        return model, losses, tr_noisy, tr_clean

# Global Error - estimates the integral error by random sampling (Monte Carlo)
    # This tells you if the network generalizes.
    # Monte Carlo methods use random sampling to est. quantities that are too hard to calculate exactly.
def mc_errors(model, p, Q=5000, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (Q,2)).astype(np.float32)  # lots of random test points
    y = f_true(X, p).astype(np.float32)               # clean truth
    with torch.no_grad():
        pred = model(torch.tensor(X)).numpy().reshape(-1)
    dif = np.abs(pred - y)
    L2 = float((dif**2).mean())   # Monte-Carlo estimate of L2 error
    SUP = float(dif.max())        # sup norm proxy: max absolute error over samples
    return L2, SUP

# Experiment Loop
# For each p (function smoothness) and each M (model size), we:
    # train a model, compute global errors, save metrics into rows
ps = [1.0, 0.5, 0.25]   # the three cases in the prompt
Ms = [5, 10, 20, 50]    # model sizes to compare

rows = []
best_by_p = {}
for p in ps:
    best = None
    for M in Ms:
        model, losses, trN, trC = train_once(p, M, N=800, epochs=400, seed=42)
        L2, SUP = mc_errors(model, p, Q=6000, seed=9)
        row = {"p": p, "M (neurons)": M,
               "Train MSE vs noisy Y": trN,
               "Train MSE vs clean f": trC,
               "Global L2 (MC)": L2,
               "Sup norm (max |err|)": SUP}
        rows.append(row)
        # track best M by global L2 error
        if (best is None) or (L2 < best[0]):
            best = (L2, M, model, losses)
    best_by_p[p] = best

# Results Table + Saving
    # prints a clean table with: p, M, training errors (noisy/clean), Global L2, and Sup norm
df = pd.DataFrame(rows).sort_values(["p","M (neurons)"]).reset_index(drop=True)
print(df)

out = Path("relu_shallow_results.csv")
df.to_csv(out, index=False)
print("Saved results to", out.absolute())

# Diagnostic Plot
    # Loss curves for the "best M" per p
    # Loss curve: shows learning progress (should trend downward and flatten)
for p in ps:
    L2, M, model, losses = best_by_p[p]
    plt.figure(figsize=(6,4))
    plt.plot(losses)
    plt.title(f"Training loss (p={p}, best M={M})")
    plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.grid(True)
    plt.tight_layout(); plt.show()

# 3D plots for the target function, prediction, and error for each p
from mpl_toolkits.mplot3d import Axes3D  # noqa
from matplotlib import cm

def plot_3d_results(model, p, grid_n=80, save_prefix=None):
    # Create the grid over [-1,1]^2
    xs = np.linspace(-1, 1, grid_n)
    ys = np.linspace(-1, 1, grid_n)
    Xg, Yg = np.meshgrid(xs, ys)
    XY = np.stack([Xg.ravel(), Yg.ravel()], axis=1).astype(np.float32)

    # Compute target, prediction, and error
    with torch.no_grad():
        Z_pred = model(torch.tensor(XY)).numpy().reshape(grid_n, grid_n)
    Z_true = f_true(XY, p).reshape(grid_n, grid_n)
    Z_err  = np.abs(Z_true - Z_pred)

    # Common z-limits for fair comparison
        # Using the same z-axis range for Target and Prediction prevents optical illusions (e.g., one looks taller just because of autoscale).
    zmin = min(Z_true.min(), Z_pred.min())
    zmax = max(Z_true.max(), Z_pred.max())

    # Plot the Target Function
        # Creates a 3D figure and draws the true surface with a smooth colormap
    fig1 = plt.figure(figsize=(6,5))
    ax1 = fig1.add_subplot(111, projection="3d")
    ax1.plot_surface(Xg, Yg, Z_true, cmap=cm.viridis, linewidth=0, antialiased=True)
    ax1.set_title(f"Target Function f(x,y), p={p}")
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("f(x,y)")
    ax1.set_zlim(zmin, zmax)
    if save_prefix: fig1.savefig(f"{save_prefix}_Target_p{p}.png", dpi=200)

    # Plot the Model Prediction
        # Same style and same z-limits as the target to compare shapes one-to-one
    fig2 = plt.figure(figsize=(6,5))
    ax2 = fig2.add_subplot(111, projection="3d")
    ax2.plot_surface(Xg, Yg, Z_pred, cmap=cm.viridis, linewidth=0, antialiased=True)
    ax2.set_title(f"Model Prediction p(x,y), p={p}")
    ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("Prediction")
    ax2.set_zlim(zmin, zmax)
    if save_prefix: fig2.savefig(f"{save_prefix}_Prediction_p{p}.png", dpi=200)

    # Plot the Absolute Error
        # Uses a different colormap (inferno) to visually separate “error” from “surfaces”
    fig3 = plt.figure(figsize=(6,5))
    ax3 = fig3.add_subplot(111, projection="3d")
    surf = ax3.plot_surface(Xg, Yg, Z_err, cmap=cm.inferno, linewidth=0, antialiased=True)
    ax3.set_title(f"Absolute Error |f - p|(x,y), p={p}")
    ax3.set_xlabel("x"); ax3.set_ylabel("y"); ax3.set_zlabel("Error")
    # make error visible even if very small
    ax3.set_zlim(0, max(float(Z_err.max()), 1e-2))
    fig3.colorbar(surf, shrink=0.6, aspect=12)
    if save_prefix: fig3.savefig(f"{save_prefix}_Error_p{p}.png", dpi=200)

    # show all three at once
    plt.show()

# Generate all 3D plots for each p value using the "best" model found
for p in ps:
    L2, M, model, losses = best_by_p[p]
    print(f"3D graphs for p={p}, best M={M}")
    plot_3d_results(model, p, grid_n=80, save_prefix=f"surfaces_p{p}_M{M}")

# Notes
    # Pointwise absolute deviation between truth and prediction.
    # Shows where errors live (e.g., near high curvature).
    # The code forces a minimum z-axis range so tiny errors still look visible.
