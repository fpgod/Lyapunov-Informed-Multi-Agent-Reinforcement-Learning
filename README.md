



# Lyapunov-Informed Multi-Agent Reinforcement Learning

This repository contains the core code for the IEEE TASE submission:

> **"Lyapunov-Informed Multi-Agent Reinforcement Learning for Multi-Robot Cooperation Tasks"**

We propose a novel MARL framework that leverages Lyapunov-based stability principles to enhance the training efficiency and robustness of cooperative policies across various multi-robot tasks.

---

## 🧠 Highlights

- ✅ Lyapunov-based intrinsic reward formulation  
- 🤖 Compatible with multiple MARL baselines: MAPPO, MADDPG, MATD3    
- 📈 Improved training stability and convergence

---

## ⚙️ Hyperparameter Settings

### MAPPO / MAPPO_LYN

| Hyperparameter               | Value                              |
|-----------------------------|------------------------------------|
| Recurrent data chunk length | 10                                 |
| Gradient clip norm          | 10.0                               |
| GAE lambda                  | 0.95                               |
| Discount factor (gamma)     | 0.99                               |
| Value loss                  | Huber loss                         |
| Huber delta                 | 10.0                               |
| Optimizer                   | Adam                               |
| Optimizer epsilon           | 1e-5                               |
| Weight decay                | 0                                  |
| Network initialization      | Orthogonal                         |
| Learning rate               | 0.0005                             |
| Beta (β)                    | 100 (PP), 600 (RE), 300 (CA)       |
|                             | 10 (SMAC), 100 (CC)                |

### MADDPG / MATD3 and their LYN variants

| Hyperparameter         | Value              |
|------------------------|--------------------|
| Gradient clip norm     | 10.0               |
| Random episodes        | 5                  |
| Epsilon                | 1 → 0.05           |
| Epsilon anneal time    | 50000 timesteps    |
| Train interval         | 1 episode          |
| Gamma                  | 0.99               |
| Buffer size            | 5000 episodes      |
| Batch size             | 32 episodes        |
| Optimizer eps          | 1e-5               |
| Optimizer              | Adam               |
| Weight decay           | 0                  |
| Network initialization | Orthogonal         |
| Learning rate               | 0.0005                             |
| Beta (β)                    | 100 (PP), 600 (RE), 300 (CA)       |
|                             | 10 (SMAC), 100 (CC)                |

---










