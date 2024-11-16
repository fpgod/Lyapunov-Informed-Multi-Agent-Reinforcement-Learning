# README

## Installation

### Install lyapunov_marl

```
conda create -n lyapunov_marl python==3.9
conda activate lyapunov_marl
git clone ...
cd lyapunov_marl
pip install -r requirements.txt
```

### Install Environments Dependencies

**Install SMAC**

Please follow [the official instructions](https://github.com/oxwhirl/smac) to install SMAC.

**Install SMACv2**

Please follow [the official instructions](https://github.com/oxwhirl/smacv2) to install SMACv2.

**Install MPE**

```
pip install pettingzoo==1.22.2
pip install supersuit==3.7.0
```

## Training

Please take a look on the scripts at:

```
cd ~/lyapunov_marl  # Go to repo root.
python scripts/train/train_mujoco.py
python scripts/train/train_robots.py
python scripts/train/train_smac.py
```

As a quick start, you can start training lyapunov_marl in rendezvous task immediately after installation by running:

```
cd ~/lyapunov_marl/scripts
sh seed_robots.sh
```

You can train different algorithms on various tasks by altering the `algo` and `env` parameters in the scripts.

We support a range of algorithms, including:

- **MAPPO** (Multi-Agent Proximal Policy Optimization)
- **MADDPG** (Multi-Agent Deep Deterministic Policy Gradient)
- **MAPPO_lyapunov** (MAPPO combined with Lyapunov methods)
- **MADDPG_lyapunov** (MADDPG combined with Lyapunov methods)

Additionally, we provide various task environments, such as:

- **Rendezvous**: Multiple agents need to gather at a common location.
- **Pursuit**: A group of agents chase one or more evaders.
- **SMAC**: StarCraft Multi-Agent Challenge, where agents collaborate to compete against enemy units in the StarCraft II environment.
- **MPE**: Multi-Agent Particle Environment, a suite of simple environments for multi-agent reinforcement learning.