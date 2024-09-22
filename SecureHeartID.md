## Documentation for Secure HeartID

This Python script defines a deep learning model for secure biometric ECG (electrocardiogram) identification. The model incorporates techniques to protect user privacy while performing user identification.

Refer to Google Colab Notebook for more information and future development of this model. 

Link: https://colab.research.google.com/drive/1wOsSJzol-tkB7uBSuYSE6sVj9aGGVUGy#scrollTo=kkfhV40W3wZp

Here's a breakdown of the code:

**Imports:**

* `torch`: The main PyTorch library for deep learning.
* `torch.nn`: Provides modules for building neural networks.
* `torch.optim`: Offers optimization algorithms for training models.

**Classes:**

* **SecureBiometricECGModel:** This class defines the core architecture of the neural network.
    * It takes input arguments for the ECG data size (`input_size`), key size (`key_size`), and dropout rate (`dropout_rate`).
    * The network consists of several convolutional layers for feature extraction followed by fully connected layers.
    * The `_get_conv_output` function helps determine the output size of the convolutional layers.
    * The `forward` function defines the data flow through the network. It takes the ECG data (`x`) and a key (`key`) as input and outputs the processed features.
* **SecureModel:** This class builds upon the `SecureBiometricECGModel` and incorporates secure user identification.
    * It takes the same input arguments as `SecureBiometricECGModel`.
    * It has a `network` attribute that is an instance of `SecureBiometricECGModel`.
    * The `forward` function takes multiple ECG data points for an anchor user (`xA`), positive users (`xP`), and negative users (`xN`), along with two keys (`k1` and `k2`). It utilizes the `network` to process each data point with a key and returns the corresponding outputs.
    * The `get_embedding` function retrieves the embedding (representation) of an ECG data point with a key.

**Helper Functions:**

* **generate_key:** This function generates a random key vector of a specified size (`key_size`).
* **SecureTripletLoss:** This class defines a custom loss function for secure triplet loss. It measures the distance between the anchor user, positive users, and negative users while incorporating keys for privacy protection.

**Training Function:**

* **train_model:** This function trains the model using the provided training and validation dataloaders.
    * It defines hyperparameters like the device (CPU or GPU) and optimizer.
    * It iterates through epochs, performing training and validation steps in each epoch.
        * During training, it calculates the secure triplet loss for each batch of data and updates the model weights using the optimizer.
        * During validation, it evaluates the model on the validation set without updating weights.
    * It implements early stopping to prevent overfitting. The model with the lowest validation loss is saved.

**Usage Example:**

* The script provides an example of how to use the model. It defines the model parameters and assumes you have already set up the data loaders for training and validation.
* Finally, it calls the `train_model` function to train the model.

**This script demonstrates a secure biometric ECG identification system using a deep learning model with key-based privacy protection.**
