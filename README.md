# Approximation Behavior of Shallow ReLU Networks

Repository: [https://github.com/gulatikrishav](https://github.com/gulatikrishav)  
(*Visit to check out code, issues, and future updates.*)

This project is a reproducible experiment in PyTorch studying how **shallow (1-hidden-layer) ReLU neural networks** approximate a known analytic target surface:

\[
f(x,y) = (x^2 + x\,y + 2y^2)^p
\]

for multiple values of the exponent **p** and multiple hidden layer widths **M**.

---

## What the script does

- Samples noisy training data from the true function over \([-1,1]^2\)  
- Trains a shallow ReLU network of width \(M\) for each setting  
- Tracks and reports:  
  - Train MSE vs noisy targets  
  - Train MSE vs the clean target  
  - Monte-Carlo global L2 error (generalization estimate)  
  - Sup-norm proxy (max absolute error over random sample)  
- Writes results to CSV (`relu_shallow_results.csv`)  
- Produces:  
  - Loss‐curve plots for each best model (per p)  
  - 3D surface plots (true target, network prediction, absolute error surface)

---

## Why this matters

- Even shallow ReLU networks are universal approximators—but in practice capacity is finite.  
- Varying the **smoothness** exponent \(p\) lets us see how *function regularity* affects learnability.  
- Using a Monte Carlo estimate gives insight into *generalization* (not just training fit).  
- Visual 3D surfaces offer geometric intuition about where errors concentrate (e.g., high curvature regions).

