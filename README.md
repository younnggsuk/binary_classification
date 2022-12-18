# Classification Backbone

This directory contains the code for training and testing a classification model.

## Usage

### Training
To train a classification model, run the following command:

```bash
$ bash train.sh
```

All results will be saved in the `logs` directory.
- Checkpoints will be saved in the `logs/checkpoints` directory.
- Confusion matrices will be saved in the `logs/confusion_matrices` directory.
- Metrics will be saved in the `logs/metrics` directory.
- Tensorboard logs will be saved in the `logs/tensorboard` directory.

### Testing

To test a classification model with CAM, run the following command:

```bash
$ bash test.sh
```

All results will be saved in the `results` directory.
- CAMs will be saved in the `results/cams` directory.
- Confusion matrices will be saved in the `results/confusion_matrices` directory.
- Metrics will be saved in the `results/metrics` directory.
