# Binary Classification

This directory contains the code for training and testing a binary classification model.

## Usage

### 1. GPUs

Before start training, you need to set the number of GPUs in the `train.sh` or `test.sh` file as follows:

```bash
# 2 GPUs are used (default)
torchrun --nproc_per_node=2 ...

# 4 GPUs are used
torchrun --nproc_per_node=4 ...
```

### 2. Train

Training settings are passed as arguments to the `train.py` file. To see the list of available options, run the following command:

```bash
$ python train.py --help
```

To train model with default settings, run the following command:

```bash
$ bash train.sh
```

All results will be saved in the `logs` directory.
- Checkpoints will be saved in the `logs/checkpoints` directory.
- Confusion matrices will be saved in the `logs/confusion_matrices` directory.
- Metrics will be saved in the `logs/metrics` directory.
- Tensorboard logs will be saved in the `logs/tensorboard` directory.

### 3. Test

Testing settings are passed as arguments to the `test.py` file. To see the list of available options, run the following command:

```bash
$ python test.py --help
```

To test model with default settings, run the following command:

```bash
$ bash train.sh
```

All results will be saved in the `results` directory.
- CAMs will be saved in the `results/cams` directory.
- Confusion matrices will be saved in the `results/confusion_matrices` directory.
- Metrics will be saved in the `results/metrics` directory.
