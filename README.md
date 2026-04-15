# _NetPatrol: Where’s the Jam?_ A Federated-Learning Approach for Jamming Detection Using Multi-Layer Perceptron 
## Overview
While previous works have focused on classifying jamming attacks using ML and some focus on physically locating jammers, there is no unified framework to perform both tasks simultaneously. Furthermore, previous approaches have relied on centralized data processing, creating privacy risks for client devices and communications. We introduce a solution using federated learning to preserve client privacy while performing jamming classification and localization with a multi-task Multi-Layer Perceptron.

## Contents
```
6711-Project-Group14
├── README.md
├── requirements.txt                      # List of required Python packages (pip3 install -r requirements.txt)
├── charts/*                              # PNG figures of all evaluation results
├── data/WSN-DS.csv                       # The raw WSN-DS dataset
└── src/                                  # Contains all implementation code
    ├── compare_models.py                 # Overall comparison script to compare RF and MLP-FedAvg models.
    ├── mlp/                              # Source & evaluation for a single-task MLP model
    │   ├── compare_mlp.py                # Compare single-task MLP w/ RF
    │   ├── evaluate_centralized_mlp.py   # Evaluate centralized MLP
    │   ├── evaluate_fedAvg.py            # Evaluate FedAvg MLP
    │   ├── main_fedAvg.py                # Single-task MLP with FedAvg training
    │   ├── main_MLP.py                   # Single-task MLP with centralized training
    │   └── mlp_model.py                  # Single-task MLP model source code
    ├── multitask_mlp/                    # Source code and evaluation for a multitask MLP
    │   ├── args.py                       # Argument parser
    │   ├── localize.py                   # Node localization script
    │   ├── main_fedAvg.py                # Multi-task MLP with FedAvg training
    │   ├── main_MLP.py                   # Multi-task MLP with centralized training
    │   ├── mlp_model.py                  # Multi-task MLP model source code
    │   └── preprocess.py                 # Preprocessing with added localization features
    ├── preprocessing/                    # Data preprocessing and analysis scripts
    │   ├── args.py                       # Argument parser
    │   ├── dataset_analysis.py           # Dataset analysation
    │   └── preprocess.py                 # Data preprocessing
    └── rf/                               # Random Forest source code and evaluation
        ├── evaluate_rf.py                # RF model evaluation
        └── rf_model.py                   # RF model source code
```

## Usage
Scripts should be run from the main project directory, e.g. ```python src/multitask_mlp/main_fedAvg.py```.

All arguments are optional.

```
usage: main_<MLP|fedAvg>.py [-h] [-T T] [-C C] [-E E] [-B B] [-cls CLS]

options:
  -h, --help  show this help message and exit
  -T T        rounds of training T
  -C C        number of clients C
  -E E        the number of local epochs E
  -B B        local batch size B
  -cls CLS    binary (b) or multclass (mc) classification
```

Example (train multitask MLP for binary classification (b) with 5 rounds of FedAvg using 30/100 clients):  

```python src/multitask_mlp/main_fedAvg.py -T 5 -C 30 -cls 'b'```

## Dataset
This work uses the [WSN-DS](https://onlinelibrary.wiley.com/doi/epdf/10.1155/2016/4731953) dataset. It contains 374632 records from a 100-node WSN running the LEACH protocol over several rounds. Each record represents a communication from a single node and contains 18 features and a label denoting traffic type (normal, flooding attack, etc.).

## Sample Results
![confusion matrix](https://github.com/a-powelson/6711-Project-Group14/blob/main/charts/compare_mlp/comparison_mlp_vs_fedavg_multiclass_confusion.png?raw=true)

This confusion matrix shows the models' classification performance. It shows that Federated Learning has introduced uncertainty to most attack classifications, likely due to the smaller traffic sample sizes available to each client during training. 

![rf vs. mlp](https://github.com/a-powelson/6711-Project-Group14/blob/main/charts/compare_models/comparison_multiclass_rf_vs_fedavg_overall.png?raw=true)

This chart shows that the Random Forest achieves excellent classification results in a centralized setting, while MLP w/ FedAvg training struggles to maintain performance. Its accuracy likely benefits from the dominance of normal traffic in the dataset.
## References
- src/multitask_mlp
    - [giokezo](https://github.com/giokezo/Multi-Task-Learning-with-a-Two-Headed-MLP)
    - [TensorFlow](https://www.tensorflow.org/api_docs/python/tf/keras/Model)
- src/mlp
    - [shaoxiongji](https://github.com/shaoxiongji/federated-learning)
    - [TensorFlow](https://www.tensorflow.org/api_docs/python/tf/keras)
    - [AbuHamza773](https://github.com/AbuHamza773/Federated-Learning-with-LSTM-for-Intrusion-Detection-in-IoT/blob/main/preprocess_custom.py)
    -[Keras - save/load models](https://keras.io/guides/serialization_and_saving/)
    - localize.py
        - [ChatGPT prompt: how to convert inter-pair distances into 2D plot?](chatgpt.com)
        - [Reddit](https://www.reddit.com/r/algorithms/comments/cx1lpl/creating_a_grid_of_nodes_using_only_the_distance/)
        - [SciKit Learn: MDS](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html)
    - main_fedAvg.py
        - [educative.io](https://www.educative.io/answers/what-is-federated-averaging-fedavg)
    - mlp_model.py
        - [GeeksForGeeks: MLP Intro](https://www.geeksforgeeks.org/deep-learning/multi-layer-perceptron-learning-in-tensorflow/)
        - [GeeksForGeeks: Activation Functions](https://www.geeksforgeeks.org/machine-learning/activation-functions-neural-networks/)
        - [GeeksForGeeks: Tanh/Sigmoid/Relu](https://www.geeksforgeeks.org/deep-learning/tanh-vs-sigmoid-vs-relu/)
        - [GeeksForGeeks: ANN layers](https://www.geeksforgeeks.org/deep-learning/layers-in-artificial-neural-networks-ann/)
        - [SciKit Learn: Batch Size](https://sklearner.com/sklearn-mlpclassifier-batch_size-parameter/)
    - preprocess.py
        - [WSN-Project](https://github.com/m-zeeshan555/WSN-DS/blob/main/Wireless%20Sensor%20Network%20Project%20Notebook.pdf)
        - [SciKit Learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
- src/rf
    - [scikit-learn]( https://scikit-learn.org/stable/model_persistence.html)
