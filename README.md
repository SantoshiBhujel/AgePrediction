# AgePrediction

The provided code encompasses a comprehensive approach to training and evaluating deep learning models for age prediction from images using DenseNet, VGG, and ResNet architectures. Here’s a detailed breakdown of each part:

# Libraries and Dependencies
Keras and TensorFlow: Essential for loading and preprocessing images, defining and training models, and visualizing results.
Matplotlib and NumPy: Utilized for plotting results and manipulating arrays.
PIL: Used for image processing.

# Utility Functions
### predict_and_plot:

Loads and preprocesses an image.
Predicts the age using the trained model.
Denormalizes the predicted age.
Plots the image with the predicted age.
### plot_label_distribution:

Plots the distribution of labels (ages) in the dataset.
Converts normalized labels back to their original values for visualization.
plot_model_structure:

Saves and displays the structure of the model as a PNG image.
### plot_training_history:

Plots the training and validation Mean Absolute Error (MAE) over epochs.
plot_real_vs_predicted:

Plots the real vs. predicted values for the test dataset.
# Data Preparation
Loading Images: Images are loaded from a specified directory, resized, and converted to arrays.
Labels Extraction: Labels (ages) are extracted from the folder names and converted to numeric values.
Normalization: Labels are normalized to a range of 0 to 1.
Data Splitting: The dataset is split into training and testing sets (80%-20%).
# Model Definition
## DenseNet Model
### densenet_model:

Loads a pre-trained DenseNet201 model without the top classification layer.
Freezes the DenseNet layers to prevent them from being trained.
Adds additional layers for regression (Conv2D, GlobalAveragePooling2D, Dropout, Flatten, Dense).
Compiles and trains the model.
Saves the trained model.
Plots the training history, model structure, and real vs. predicted values.
### train_test_densenet:

Constructs the model name based on the training parameters.
Trains the DenseNet model using the training data.
Loads the saved model.
Uses the trained model to predict the age of a sample image and plots the result.
## VGG Model
### vgg_model:

Loads a pre-trained VGG16 model without the top classification layer.
Freezes the VGG layers to prevent them from being trained.
Adds additional layers for regression (Conv2D, MaxPooling2D, Flatten, Dense, Dropout).
Compiles and trains the model.
Saves the trained model.
Plots the training history, model structure, and real vs. predicted values.
### train_test_vgg:

Constructs the model name based on the training parameters.
Trains the VGG model using the training data.
Loads the saved model.
Uses the trained model to predict the age of a sample image and plots the result.
## ResNet Model
### resnet_model:

Loads a pre-trained ResNet50 model without the top classification layer.
Freezes the ResNet layers to prevent them from being trained.
Adds additional layers for regression (Conv2D, MaxPooling2D, Flatten, Dense, Dropout).
Compiles and trains the model.
Saves the trained model.
Plots the training history, model structure, and real vs. predicted values.
### train_test_resnet:

Constructs the model name based on the training parameters.
Trains the ResNet model using the training data.
Loads the saved model.
Uses the trained model to predict the age of a sample image and plots the result.
# Running the Models
train_test_densenet, train_test_vgg, and train_test_resnet functions are called with specified parameters for optimizer, image size, batch size, epochs, and learning rate to start the training and evaluation process for each respective model architecture.
# Summary
This script demonstrates a comprehensive approach to developing deep learning models for age prediction using DenseNet, VGG, and ResNet architectures. The workflow includes data loading and preprocessing, model definition, training, evaluation, and visualization of results. This approach is particularly useful for applications requiring accurate age estimation from facial images using advanced deep learning techniques.