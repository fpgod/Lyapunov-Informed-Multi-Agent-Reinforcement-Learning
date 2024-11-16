### 1. Usage

The code consists of the following parts:

1. **Algorithms**: This section includes the algorithm structures, featuring both the trainer and policy for each type of algorithm.
2. **Environments (envs)**: This contains the experimental environments.
3. **Runners**: This includes files that manage the training processes.
4. **Configurations (configs)**: This contains the adjustable parameters for the algorithms.

After setting up the appropriate runtime environment, you can start the training in the corresponding environment by launching the `runner.py` file located in the `runners` directory. Training parameters can be adjusted in the `config.py` file. Modifications to the algorithm network can be made within the `algorithms` directory.



### 2. Installation

Here we give an example installation on CUDA == 10.1. For non-GPU & other CUDA version installation, please refer to the [PyTorch website](https://pytorch.org/get-started/locally/). We remark that this repo. does not depend on a specific CUDA version, feel free to use any CUDA version suitable on your own computer.

```
# create conda environment
conda create -n marl python==3.9.12
conda activate marl
pip install torch==1.13.0+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```



#### 2.1 MPE

```
# install this package first
pip install seaborn
```

There are 3 Cooperative scenarios in MPE:

- simple_spread
- simple_speaker_listener, which is 'Comm' scenario in paper
- simple_reference



#### 2.2 Webots

Webots provides a complete development environment to model, program and simulate robots, vehicles and mechanical systems.

- See the [Webots introduction video](https://www.youtube.com/watch?v=O7U3sX_ubGc).
- View online Webots simulations at [webots.cloud](https://webots.cloud/).
- Participate in the [IROS 2023 Simulated Humanoid Robot Wrestling Competition](https://webots.cloud/run?version=R2023a&url=https%3A%2F%2Fgithub.com%2Fcyberbotics%2Fwrestling%2Fblob%2Fmain%2Fworlds%2Fwrestling.wbt&type=competition) and win 1 Ethereum.

##### Download

Get pre-compiled binaries for the [latest release](https://github.com/cyberbotics/webots/releases/latest), as well as [older releases and nightly builds](https://github.com/cyberbotics/webots/releases).

Check out installation instructions:

[![Linux](https://camo.githubusercontent.com/848b1342527a2944a85ddb8685e73f4c74268f42bd181840c3faa85bce3804a3/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f4c696e75782d3066383063303f6c6f676f3d6c696e7578266c6f676f436f6c6f723d7768697465)](https://cyberbotics.com/doc/guide/installation-procedure#installation-on-linux) [![Windows](https://camo.githubusercontent.com/9b8bd81f931f610c84986b7e04302bfc0cc211c9f730828bd7641e6b80737862/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f57696e646f77732d3066383063303f6c6f676f3d77696e646f7773266c6f676f436f6c6f723d7768697465)](https://cyberbotics.com/doc/guide/installation-procedure#installation-on-windows) [![macOS](https://camo.githubusercontent.com/0bb488f7f9dcdc3dfec00943618fe70a9f6234cd572afa77ba04d8a7eb4ba135/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f6d61634f532d3066383063303f6c6f676f3d6170706c65266c6f676f436f6c6f723d7768697465)](https://cyberbotics.com/doc/guide/installation-procedure#installation-on-macos)

##### Build from Source

If you prefer to [compile Webots from source](https://github.com/cyberbotics/webots/wiki), read the [contributing guidelines](https://github.com/cyberbotics/webots/blob/master/CONTRIBUTING.md).

##### Continuous Integration Nightly Tests

[![master branch](https://camo.githubusercontent.com/e878a3c1ed5813dc541ea1b2968e6a11bab8579dab09719252864c90a688f743/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f6272616e63682d6d61737465722d626c7565)](https://github.com/cyberbotics/webots/tree/master) [![Linux build (master)](https://github.com/cyberbotics/webots/actions/workflows/test_suite_linux.yml/badge.svg?event=schedule)](https://github.com/cyberbotics/webots/actions/workflows/test_suite_linux.yml?query=event%3Aschedule) [![Windows build (master)](https://github.com/cyberbotics/webots/actions/workflows/test_suite_windows.yml/badge.svg?event=schedule)](https://github.com/cyberbotics/webots/actions/workflows/test_suite_windows.yml?query=event%3Aschedule) [![macOS build (master)](https://github.com/cyberbotics/webots/actions/workflows/test_suite_mac.yml/badge.svg?event=schedule&label=macOS)](https://github.com/cyberbotics/webots/actions/workflows/test_suite_mac.yml?query=event%3Aschedule)
[![develop branch](https://camo.githubusercontent.com/8857ff4274a29bb5d112875db578d4b95c114715c80041d21c7ed0a585ff3fe2/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f6272616e63682d646576656c6f702d626c7565)](https://github.com/cyberbotics/webots/tree/develop) [![Linux build (develop)](https://github.com/cyberbotics/webots/actions/workflows/test_suite_linux_develop.yml/badge.svg?event=schedule)](https://github.com/cyberbotics/webots/actions/workflows/test_suite_linux_develop.yml?query=event%3Aschedule) [![Windows build (develop)](https://github.com/cyberbotics/webots/actions/workflows/test_suite_windows_develop.yml/badge.svg?event=schedule)](https://github.com/cyberbotics/webots/actions/workflows/test_suite_windows_develop.yml?query=event%3Aschedule) [![macOS build (develop)](https://github.com/cyberbotics/webots/actions/workflows/test_suite_mac_develop.yml/badge.svg?event=schedule)](https://github.com/cyberbotics/webots/actions/workflows/test_suite_mac_develop.yml?query=event%3Aschedule)





### 3. Train

Here we use shared-policy trainer in webots environment as an example:

```
cd scripts
./train_mujoco.sh
```

Local results are stored in subfold scripts/results. Note that we use Weights & Bias as the default visualization platform; to use Weights & Bias, please register and login to the platform first. More instructions for using Weights&Bias can be found in the official [documentation](https://docs.wandb.ai/). Adding the in command line or in the .sh file will use Tensorboard instead of Weights & Biases.`--use_wandb`

