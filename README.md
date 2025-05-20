# GNN Recommender System

This project implements a Graph Neural Network (GNN) based recommender system. The primary model used is LightGCN, which is well-suited for collaborative filtering tasks by learning user and item embeddings through graph convolutions. The system is designed to predict user-item interactions (e.g., movie ratings) and can be trained on various datasets.

## Methodology

The core of this recommender system is the LightGCN model. LightGCN simplifies the traditional GCN architecture by removing feature transformations and non-linear activation functions in the propagation layers. It learns user and item embeddings by linearly propagating them on the user-item interaction graph. The final embeddings for users and items are a weighted sum of the embeddings learned at different propagation layers. These embeddings are then used to predict the likelihood of a user interacting with an item, typically through a dot product.

Key aspects of the methodology include:
- **Graph Construction:** Building a bipartite graph where nodes represent users and items, and edges represent interactions (e.g., ratings).
- **Embedding Propagation:** Learning user and item embeddings by aggregating information from their neighbors in the graph over multiple layers.
- **Prediction:** Using the learned embeddings to predict user preferences for items they have not interacted with.
- **Loss Function:** Typically, a pairwise loss function like BPR (Bayesian Personalized Ranking) loss is used to optimize the model, aiming to rank observed interactions higher than unobserved ones.

## Project Structure

```
.
├── README.md                               # This file
├── best_lightgcn_model_movielens-100k.pth  # Example of a saved trained model
├── movielens-100k_processed_graph.pt     # Example of a processed graph
├── ml-100k/                                # MovieLens 100k dataset
│   ├── u.data
│   └── ...
├── ml-1m/                                  # MovieLens 1M dataset (if used)
│   ├── ratings.dat
│   └── ...
├── notebooks/                              # Jupyter notebooks for experimentation, analysis, and model training
│   ├── gnn-recsys.ipynb                    # Main notebook for LightGCN implementation and training
│   ├── movielens-eda.ipynb                 # Notebook for exploratory data analysis on MovieLens
│   ├── best_lightgcn_model_movielens-100k.pth # Saved model (can also be in root)
│   ├── movielens-100k_processed_graph.pt   # Processed graph (can also be in root)
│   └── ...
├── src/                                    # Source code for the GNN model and utilities
│   ├── __init__.py
│   ├── data_utils.py                     # Scripts for data loading, preprocessing, and graph creation
│   ├── evaluate.py                       # Scripts for evaluation metrics (e.g., Recall@K, AUC-ROC, AUC-PR)
│   ├── model.py                          # Definition of the LightGCN model and other GNN architectures
│   └── train.py                          # Training loops, loss functions, and optimization
└── yelp/                                   # Yelp dataset (if used)
    └── Yelp-JSON/
        └── ...
```

## Datasets

The system is designed to work with common recommendation datasets. The primary datasets used and demonstrated in the notebooks are:

-   **MovieLens 100k:** A classic dataset with 100,000 ratings from 943 users on 1682 movies.
-   **MovieLens 1M:** A larger dataset with 1 million ratings. (Presence inferred from directory structure)
-   **Yelp Dataset:** A dataset containing user reviews and business information. (Presence inferred from directory structure and notebook outputs)

The `src/data_utils.py` script and the notebooks contain utilities for downloading, processing these datasets, and constructing the necessary graph structures for the GNN model. Processed graph data is often saved (e.g., as `.pt` files) to speed up subsequent runs.

## Setup

1.  **Clone the Repository (if applicable):**
    ```bash
    # git clone <repository_url>
    # cd GNN-RecSys
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    The core dependencies include PyTorch, PyTorch Geometric, pandas, and scikit-learn.
    ```bash
    pip install -r requirements.txt
    ```
    Refer to the PyTorch and PyTorch Geometric official websites for specific installation commands tailored to your system (CPU/GPU, CUDA version).

## How to Use

The primary way to use this codebase is through the Jupyter Notebooks provided in the `notebooks/` directory.

1.  **Navigate to the `notebooks/` directory.**
2.  **Launch Jupyter Lab or Jupyter Notebook:**
    ```bash
    jupyter lab
    # or
    jupyter notebook
    ```
3.  **Open `gnn-recsys.ipynb`:** This notebook contains the end-to-end pipeline:
    *   Data loading and preprocessing for datasets like MovieLens or Yelp.
    *   Graph construction.
    *   Definition of the LightGCN model (and potentially other GNN models).
    *   Training procedures, including hyperparameter settings.
    *   Evaluation of the trained model.
    *   Saving and loading trained models.

4.  **Run the cells in the notebook:** Execute the cells sequentially to train the model and see the results. You can modify parameters, change datasets, or experiment with the model architecture within the notebook.

**Key scripts in `src/`:**
*   `model.py`: Defines the LightGCN architecture. You can inspect this to understand the model's layers and forward pass.
*   `train.py`: Contains the training loop, loss calculation, and optimization logic. This is often called from the notebook.
*   `data_utils.py`: Handles the fetching, parsing, and transformation of raw dataset files into graph data usable by PyTorch Geometric.
*   `evaluate.py`: Provides functions to calculate various recommendation metrics.

## Evaluation

The model's performance is evaluated using several standard recommendation metrics:

-   **AUC-ROC (Area Under the Receiver Operating Characteristic Curve):** Measures the model's ability to distinguish between positive and negative interactions.
-   **AUC-PR (Area Under the Precision-Recall Curve):** Particularly useful for imbalanced datasets, common in recommendation.
-   **Recall@K:** Measures the proportion of relevant items found in the top-K recommendations.

During training (as seen in `gnn-recsys.ipynb`), these metrics are typically calculated on a validation set at the end of each epoch. The model that performs best on a chosen validation metric (e.g., AUC-ROC) is often saved. The `evaluate.py` script can be used for more detailed offline evaluation if needed.
