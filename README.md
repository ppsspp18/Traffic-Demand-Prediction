# Traffic Demand Prediction Case Study

This project focuses on predicting traffic demand using a variety of machine learning and deep learning models. It implements both traditional regression techniques and advanced neural network architectures, including a hybrid Graph Neural Network (GNN) model.

## Project Overview

The goal of this case study is to forecast traffic demand based on historical data. The project compares the performance of several models using standard regression metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE).

## Dataset

The project relies on pre-processed demand data and graph structure data:

* **`processed_demand_datasetsMAN.npz`**: Contains the training, validation, and testing sets for features (`X`) and targets (`y`).
    * Shapes: Train (63, 886, 42), Val/Test (63, 101, 42).
* **`edges_GAman.npy`**: Contains the edge list defining the graph structure used for the GNN model.

## Models Implemented

The following models are implemented and evaluated in this project:

1.  **Random Forest Regressor (RF)**: An ensemble method using `sklearn`.
2.  **Gradient Boosting Regressor (GBDT)**: A boosting technique using `sklearn`.
3.  **MLP Regressor (MLP)**: A Multi-Layer Perceptron neural network.
4.  **GRU (Gated Recurrent Unit)**: A Recurrent Neural Network (RNN) tailored for time-series data.
5.  **GRU-GAT**: A hybrid model combining GRU for temporal feature extraction and Graph Attention Networks (GAT) for spatial dependency modeling.

## Requirements

To run the notebook, you will need the following Python libraries:

* `numpy`
* `pandas`
* `torch` (PyTorch)
* `scikit-learn`

## Usage

1.  Ensure the dataset files (`processed_demand_datasetsMAN.npz` and `edges_GAman.npy`) are in the same directory as the notebook or update the file paths in the code.
2.  Run the `Demand Prediction.ipynb` notebook.
3.  The notebook will:
    * Load and prepare the data.
    * Train each model sequentially.
    * Output evaluation metrics for each model.
    * Display a comparative error table at the end.

## Results

The models were evaluated on the test set. Below is the summary of the performance metrics obtained from the notebook:

| Model | MAE | RMSE | MAPE |
| :--- | :--- | :--- | :--- |
| **RF** | 2.164 | 4.394 | 21.693 |
| **GBDT** | 2.224 | 4.432 | 22.182 |
| **MLP** | 2.273 | 4.587 | 23.100 |
| **GRU** | 2.526 | 4.655 | 0.252 |
| **GRU-GAT** | 10.500 | 19.010 | 0.645 |

*Note: The traditional ML models (RF, GBDT, MLP) show lower MAE/RMSE scores compared to the deep learning implementations in this specific run. The GRU-GAT model, being more complex, may require further hyperparameter tuning or extended training to converge optimally.*
