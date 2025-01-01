# mnist_using_cnn
The MNIST dataset is a collection of 70,000 handwritten digit images (28x28 pixels) that is widely used in machine learning for benchmarking image classification models. It consists of 60,000 training images and 10,000 test images, each representing a digit (0-9).

A Convolutional Neural Network (CNN) is a deep learning architecture commonly used for image classification tasks. CNNs are especially powerful because they can automatically learn and extract spatial hierarchies of features from images.

Here's a step-by-step description of how CNNs are applied to the MNIST dataset:

1. Preprocessing the Data:
Reshaping: The MNIST images are 28x28 pixels, and CNNs require the data in a 4D format: (number_of_images, height, width, channels). Since MNIST images are grayscale, the shape becomes (60000, 28, 28, 1) for training data and (10000, 28, 28, 1) for test data.
Normalization: Pixel values are usually in the range [0, 255]. For better performance in neural networks, pixel values are normalized to a range of [0, 1] by dividing by 255.
One-hot encoding: The labels (digits 0-9) are usually converted to one-hot encoded vectors for classification tasks. For example, the label 5 becomes [0, 0, 0, 0, 0, 1, 0, 0, 0, 0].
2. CNN Architecture for MNIST:
A typical CNN model for MNIST classification consists of the following layers:

Input Layer:
The input is the reshaped and normalized image, typically of shape (28, 28, 1) for grayscale images.
Convolutional Layers (Conv2D):
First Convolutional Layer:
The convolutional layer applies a set of learnable filters (kernels) to the input image, which helps extract features such as edges, textures, and basic shapes.
For example, using 32 filters of size (3, 3) would result in 32 feature maps.
Activation Function:
After applying the filters, a ReLU (Rectified Linear Unit) activation function is commonly used to introduce non-linearity and enable the model to learn complex patterns.
Pooling:
MaxPooling is often applied after convolutional layers to reduce the spatial dimensions of the feature maps while retaining the important features. For example, using a (2, 2) pool size reduces the size by a factor of 2.
MaxPooling helps reduce computational complexity and prevents overfitting.
Additional Convolutional and Pooling Layers:
Deeper networks typically have multiple convolutional layers followed by pooling layers. These layers help the model learn more abstract and complex features at different levels (e.g., edges in early layers, shapes and objects in deeper layers).
Flattening:
After the convolutional and pooling layers, the 2D feature maps are flattened into a 1D vector, which can be fed into fully connected (dense) layers.
Fully Connected Layers (Dense Layers):
These layers consist of neurons connected to all neurons from the previous layer. These layers combine features learned by the convolutional layers to make final predictions.
Typically, a dense layer with a ReLU activation function is used.
Output Layer:
The final layer is a softmax layer with 10 neurons (one for each digit, 0-9). This layer converts the output into probabilities, where the highest probability corresponds to the predicted digit.
3. Training the CNN:
Loss Function: The typical loss function for classification tasks like MNIST is categorical cross-entropy, which compares the predicted probabilities with the actual labels.
Optimizer: Common optimizers like Adam or SGD (Stochastic Gradient Descent) are used to minimize the loss function.
Metrics: Accuracy is commonly used as a performance metric to evaluate how many predictions match the true labels.
4. Model Evaluation:
After training the CNN on the training data, the model is evaluated on the test data (10,000 images). The model’s accuracy on the test set is reported as an indicator of its generalization ability.
