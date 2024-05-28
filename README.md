## Using honey bee flight activity data and a deep learning model as a toxicovigilance tool


This repository contains the [source code](code/toxicovigilance_tool.py) of a Recurrent Neural Network to Classify pesticides on bees using activity data.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)

## Introduction

The tool is developed to facilitate the surveillance of toxicological information. It leverages machine learning algorithms and data visualization techniques to detect, analyze, and report toxicological trends and anomalies. This tool is essential for public health organizations, environmental agencies, and research institutions focused on toxicology.

## Features

- **Data Collection**: Seamless integration with various data sources for real-time data collection.
- **Data Preprocessing**: Robust data cleaning and preprocessing modules to ensure high-quality input.
- **Machine Learning Models**: A suite of pre-trained models for predictive analysis and anomaly detection.
- **Visualization**: Interactive dashboards and visualizations for easy interpretation of results.
- **Alerts**: Automated alerting system for early detection of toxicological hazards.
- **Scalability**: Designed to handle large datasets with efficient processing.

## Installation

To install the ToxicovigilanceTool, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/HpcDataLab/ToxicovigilanceTool.git
    cd ToxicovigilanceTool
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Set up the environment variables as per the configuration guidelines in `config/README.md`.

## Usage

### Running the Tool

To start using the ToxicovigilanceTool, execute the main script:
```sh
python main.py
```

### Examples

Check the `examples` directory for sample scripts demonstrating various functionalities of the tool.

### Using a Pretrained .h5 Model

#### Step 1: Load the Pretrained Model

Use the `load_model` function from Keras to load your `.h5` model.

```python
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the pretrained model
model = load_model('path/to/your/model.h5')
```

#### Step 2: Prepare the Data

Prepare your input data in the format that the model expects. This usually involves scaling, reshaping, and normalizing the data.

```python
import numpy as np

# Example: Preparing a single sample input
# Assuming the model expects images of shape (224, 224, 3)
input_data = np.random.random((1, 224, 224, 3))  # Replace with your actual data

# If your model expects normalized data, normalize it accordingly
input_data = input_data / 255.0
```

#### Step 3: Make Predictions

Use the `predict` method of the model to make predictions on the prepared data.

```python
# Making a prediction
predictions = model.predict(input_data)

# Interpreting the predictions
print(predictions)
```

#### Step 4: Evaluate the Model (Optional)

If you have a validation or test dataset, you can evaluate the model's performance.

```python
# Assuming you have X_test and y_test as your test data and labels
# Prepare the test data similarly to the input data
X_test = np.random.random((10, 224, 224, 3))  # Replace with your actual test data
y_test = np.random.randint(0, 2, (10, 1))  # Replace with your actual test labels

X_test = X_test / 255.0

# Evaluating the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')
```

#### Step 5: Save or Export Predictions (Optional)

You can save the predictions to a file or export them as needed.

```python
import pandas as pd

# Example: Saving predictions to a CSV file
pred_df = pd.DataFrame(predictions, columns=['Prediction'])
pred_df.to_csv('predictions.csv', index=False)
```


