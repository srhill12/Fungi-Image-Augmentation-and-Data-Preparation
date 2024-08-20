

# Fungi Image Augmentation and Data Preparation

This project focuses on the augmentation and preparation of a dataset of fungal images for further use in training machine learning models. The dataset has been preprocessed and augmented to enhance the diversity and robustness of the training data, thereby improving model performance.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Data Augmentation](#data-augmentation)
- [Saving the Augmented Data](#saving-the-augmented-data)
- [Next Steps](#next-steps)
- [License](#license)

## Overview

This project involves augmenting a dataset of fungal images to create a more robust training set for machine learning models. The images are preprocessed, augmented using various transformations, and then saved for further use.

## Installation

1. **Clone the repository:**
   ```bash
   git clone repository_url
   ```

2. **Install the required libraries:**
   ```bash
   pip install tensorflow matplotlib pillow requests numpy pandas scikit-learn
   ```

3. **Open the notebook in Google Colab:**
   - This project is designed to be run in Google Colab. Click on the "Open in Colab" badge or open the notebook directly in Google Colab.

## Data Preparation

1. **Load and Inspect Data:**
   - The dataset is loaded from two URLs and contains preprocessed fungal images (`X`) and corresponding labels (`y`).
   - The images are numpy arrays representing pixel values, and the labels are class labels that have been label encoded.

2. **Data Splitting:**
   - The dataset is split into training and validation sets using `train_test_split`.

## Data Augmentation

1. **Image Augmentation:**
   - The `ImageDataGenerator` class from TensorFlow is used to perform various image augmentations, including:
     - Random rotation
     - Horizontal and vertical shifts
     - Shear and zoom transformations
     - Horizontal flipping
   - For each original image, 5 new augmented images are generated to expand the training set.

2. **Augmentation Process:**
   - The augmented images and their corresponding labels are stored in lists for later use.

## Saving the Augmented Data

1. **Mount Google Drive:**
   - The project uses Google Drive to store the augmented dataset in a pickle file.

2. **Save the Dataset:**
   - The final augmented dataset is saved in a dictionary and exported as a pickle file to Google Drive.

## Next Steps

- **Model Training:**
  - Use the augmented dataset to train a Convolutional Neural Network (CNN) or another machine learning model.
  - Experiment with different model architectures and hyperparameters to optimize performance.

- **Additional Augmentations:**
  - Explore other augmentation techniques, such as brightness adjustment, contrast adjustment, and image sharpening.

- **Transfer Learning:**
  - Apply transfer learning techniques using pre-trained models like ResNet or VGG to improve classification performance.

- **Model Evaluation:**
  - Evaluate the model's performance on the validation set using metrics such as accuracy, precision, recall, and F1 score.

## License

This project is licensed under the MIT License.

---
