# HeartID: Secure Biometric ECG Model

HeartID is an enhanced secure biometric model that uses Electrocardiogram (ECG) signals for identification purposes. This implementation is built using the MLX framework, offering efficient computation on Apple Silicon.

## Purpose

The HeartID model aims to provide a secure and efficient method for biometric identification using ECG signals. It incorporates several key features:

1. Convolutional layers for feature extraction from ECG signals
2. Short-Time Fourier Transform (STFT) for spectral analysis
3. A secure key-based approach for enhanced privacy
4. Triplet loss for learning discriminative embeddings

This model can be used in various applications where secure, personalized biometric identification is required, such as healthcare systems, secure access control, or personalized medical devices.

## Prerequisites

- Python 3.8 or higher
- Apple Silicon Mac (M1 chip or later) for optimal performance with MLX

## Installation

MLX and MLX STFT is available on PyPI. To install the Python API, run:

With pip:

Install the required dependencies:
   ```
   pip install mlx
   pip install mlx_stft
   ```

## Usage

To train the HeartID model, run the following command:

```
python HeartID_MLX.py
```

Note: The current implementation uses randomly generated data for demonstration purposes. For actual use, you'll need to modify the data loading section in the `train` function to use your ECG dataset.

## Dataset Requirements

The HeartID model expects ECG data in the following format:

- Input shape: `(batch_size, 1, input_size)`
- `input_size` is set to 1000 by default, but can be adjusted in the `main` function

To use your own dataset:

1. Prepare your ECG data in the required format
2. Modify the data loading section in the `train` function of `HeartID_MLX.py`
3. Adjust the `num_batches` variable based on your dataset size

## Customization

You can customize various aspects of the model by modifying the hyperparameters in the `main` function of `HeartID_MLX.py`:

- `input_size`: Length of the ECG signal
- `key_size`: Size of the security key
- `num_kernels`: Number of kernels in each convolutional layer
- `learning_rate`: Learning rate for the Adam optimizer
- `num_epochs`: Number of training epochs
- `batch_size`: Batch size for training

## Output

After training, the script will save:

1. The trained model parameters as `heartid_model.npz`
2. The optimizer state as `optimizer.safetensors`

These files can be used for later inference or to resume training.

## Acknowledgments

- This implementation uses the MLX framework developed by Apple.
- The STFT implementation is based on the mlx_stft package.
