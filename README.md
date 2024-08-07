# Automated Speech Recognition (ASR) System

This repository contains a Jupyter Notebook for an Automated Speech Recognition (ASR) system built using deep learning techniques. The notebook includes data preprocessing, data augmentation, model training, and optimization steps to develop and fine-tune a convolutional neural network (CNN) model for ASR.

1. **Downloading the Data:**
   The `curl` command is used to download files from the internet. The `-O` flag saves the file with its original name.

   ```bash
   !curl -O https://raw.githubusercontent.com/andrsn/data/main/speechImageData.zip
   ```

2. **Unzipping the Data:**
   The `unzip` command is used to extract files from a ZIP archive. The `-q` flag suppresses the output, making it quiet.

   ```bash
   !unzip -q speechImageData.zip
   ```

These commands together will download the ZIP file from the specified URL and then unzip it to extract the contents. 

### Steps in Detail:

1. **Open your Jupyter Notebook:**
   Start your Jupyter Notebook environment.

2. **Download the Data:**
   Use the `curl` command to download the ZIP file. Execute the following in a code cell:

   ```python
   !curl -O https://raw.githubusercontent.com/andrsn/data/main/speechImageData.zip
   ```

3. **Unzip the Data:**
   After the download is complete, unzip the file using:

   ```python
   !unzip -q speechImageData.zip
   ```

## Contents

1. **Preprocessing**
    - **Original Data Processing**: Loading and preprocessing of training and validation datasets.
    - **Spectrogram Augmentation**: Augmenting the data using frequency and time masks.
    - **Spectrogram Mixing**: Further augmenting the data using Mixup technique.

2. **Model Building**
    - **Basic Model**: Building and training a basic CNN model on the preprocessed data.
    - **Training with Augmented Data**: Training the model on various combinations of original and augmented data.

3. **Optimization**
    - **Grid Search Optimization**: Using GridSearchCV to find the best hyperparameters.
    - **Bayesian Optimization**: Using Tree-structured Parzen Estimator (TPE) for hyperparameter tuning.

## Getting Started

### Prerequisites

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- Sci-Keras (for Grid Search)
- Scikit-Learn (for Grid Search)
- Hyperopt (for Bayesian Optimization)

### Installation

Install the necessary packages using pip:

```bash
pip install tensorflow numpy matplotlib scikeras scikit-learn hyperopt
```

### Running the Notebook

1. **Clone the repository**:

    ```bash
    git clone https://github.com/ameer-alwadiya/ASR-deep-learning.git
    cd ASR-deep-learning
    ```

2. **Open the Jupyter Notebook**:

    ```bash
    jupyter notebook "automated speech recognition (ASR) system.ipynb"
    ```

3. **Run the Notebook**: Execute the cells in the notebook to preprocess the data, build and train the model, and perform optimization.

## Brief Description of the Notebook

### 1. Preprocessing

- **Original Data Processing**: 
  - Loads the dataset using `tf.keras.utils.image_dataset_from_directory`.
  - Prepares training and validation datasets with grayscale images of size (98, 50).
  - Extracts and concatenates images and labels into numpy arrays.
  - Visualizes some examples from the training dataset.

- **Spectrogram Augmentation**:
  - Applies frequency and time masking to the spectrograms.
  - Displays original and augmented spectrograms.
  
- **Spectrogram Mixing**:
  - Mixes spectrograms using the Mixup technique.
  - Displays original and mixed spectrograms.

### 2. Model Building

- **Basic Model**: 
  - Defines a CNN model with specified parameters.
  - Trains the model using different datasets.
  - Evaluates the model on the validation dataset.

### 3. Optimization

- **Grid Search Optimization**:
  - Uses `GridSearchCV` to find the best hyperparameters.
  - Trains the model with the best parameters and evaluates its performance.

- **Bayesian Optimization**:
  - Uses Hyperopt's Tree-structured Parzen Estimator (TPE) to optimize hyperparameters.
  - Trains and evaluates the model using the best hyperparameters found.

## Results

The notebook demonstrates different approaches to improve the ASR model's accuracy using data augmentation and hyperparameter optimization. It shows that combining original data with augmented data can improve model performance, and further optimization using grid search and Bayesian techniques can yield better results.

## Conclusion

This project provides a comprehensive approach to building and optimizing a deep learning-based ASR system. It highlights the importance of data preprocessing, augmentation, and hyperparameter tuning in achieving higher model accuracy.

## Acknowledgments

- The implementation details and results were adapted from various deep learning and ASR research papers and tutorials.
- Special thanks to the TensorFlow and Keras teams for their excellent libraries and documentation.

---

Feel free to explore the notebook, experiment with different parameters, and improve the model further. If you have any questions or suggestions, please open an issue or submit a pull request. Happy coding!
