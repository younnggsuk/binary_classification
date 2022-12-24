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

Specifiying the specific GPUs to use is not supported yet.

### 2. Train

Training settings are passed as arguments to the `train.py` file. To see the list of available options, run the following command:

```bash
$ python train.py --help
```

To train model with default settings, run the following command:

```bash
$ bash train.sh
```

Or, you can run the script in the background using `nohup` as follows:

```bash
$ nohup bash train.sh &
```

All results will be saved in the `logs/exp_name` directory.  
(`exp_name` is the name of the experiment you can pass as an argument to the `train.py` file.)
- Training history will be logged in the `logs/exp_name/train.log` file.
- Checkpoints will be saved in the `logs/exp_name/checkpoints` directory.
- Confusion matrices will be saved in the `logs/exp_name/confusion_matrices` directory.
- Metrics will be saved in the `logs/exp_name/metrics` directory.
- Tensorboard logs will be saved in the `logs/exp_name/tensorboard` directory.

### 3. Test

Testing settings are passed as arguments to the `test.py` file. To see the list of available options, run the following command:

```bash
$ python test.py --help
```

To test model with default settings, run the following command:

```bash
$ bash test.sh
```

Or, you can run the script in the background using `nohup` as follows:

```bash
$ nohup bash test.sh &
```

All results will be saved in the `results/exp_name` directory.  
(`exp_name` is the name of the experiment you can pass as an argument to the `test.py` file.)
- Testing history will be logged in the `results/exp_name/test.log` file.
- CAMs will be saved in the `results/exp_name/cams` directory.
- Confusion matrices will be saved in the `results/exp_name/confusion_matrices` directory.
- Metrics will be saved in the `results/exp_name/metrics` directory.
