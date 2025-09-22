# Extended-Stability Runge–Kutta Methods for Neural ODEs  
*SATML Position Paper — “Position: Extended-Stability Runge–Kutta Methods for Neural ODEs: Predictable Cost, Robustness, and Stability”*  

 

---

## 📌 Overview
This repository accompanies our SATML position paper on **Extended-Stability Runge–Kutta (ESRK) methods for Neural ODEs**.  

Our thesis:  
> The ML community has overlooked solver families whose stability domains vastly exceed those of RK4 and Euler.  
> By integrating ESRKs into Neural ODEs, we achieve **predictable cost, robustness through stability, and scalable continuous-depth models**.


---

## 🔬 Mathematical Validation and PDE Benchmarks

This repo does not just provide experiments — it also **verifies the mathematics behind ESRKs** and validates them on a **canonical PDE benchmark**.

- **`b_series_ver.py`** → Symbolic **B-series verification** of ESRK order conditions.  
  - Confirms that our reduced-parameter ESRK coefficients satisfy all third- and fourth-order rooted-tree conditions.  
  - Reproduces the formal proofs described in Sec. 3 of the paper.  
  - Output: pass/fail for each order condition, demonstrating mathematical soundness.  

- **`convergence_studie.py`** → **2D Brusselator PDE convergence study.**  
  - A nonlinear reaction–diffusion system used as a gold-standard benchmark in numerical analysis.  
  - Confirms that ESRK-15 achieves **third-order convergence in practice**, even on mildly stiff PDE dynamics with diffusion-driven instabilities.  
  - Reproduces the section of the mathematical background of ESRKS.  

Run them directly:

```bash
# Verify B-series order conditions
python3 b_series_ver.py

# Validate ESRK-15 on the 2D Bruss
python3 convergence_studie.py

```


## 🔬 CIFAR-10 and CIFAR-100

```bash
# Run for RK4 comparsion cifar-10
  python3 train.py --solver rk4 --steps 4 --h 7.5  --epochs 200

# Run for  ESRK-15 cifar-10
python3 train.py --solver esrk --steps 1 --h 30  --epochs 200

```




