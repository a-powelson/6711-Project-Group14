# 6711-Project-Group14
**Problem Statement**: while previous works have focused on classifying jamming attacks using ML and some focus on physically locating jammers, there is no unified framework to perform both tasks simultaneously. Furthermore, previous approaches have relied on centralized data processing, creating privacy risks for client devices and communications. We introduce a solution using federated learning to preserve client privacy while performing jamming classification and localization with a multi-task Multi-Layer Perceptron.

## Contents

The project is divided into the following directories:
- **charts**: contains figures of various evaluation results
- **data**: contains the WSN-DS dataset in a csv
- **src**: contains all implementation code.
    - **mlp/**: contains a single-task (classification) MLP model with centralized (main_MLP.py) and federated (main_fedAvg.py) training, and evaluation scripts.
    - **multitask_mlp/**: contains a multitask MLP with federated training (main_FedAvg.py) and a node localization script (localize.py).
    - **preprocessing/**: contains data preprocessing and dataset analysis scripts.
    - **rf/**: contains the Random Forest implementation and evaluation scripts.
    - compare_models.py: overall comparison script to compare RF and MLP w/ FedAvg models.
    - requirements.txt: list of required Python packages, can be installed with ```pip3 install -r requirements.txt```

## Usage
Scripts should be run from the main project directory, e.g. python src/multitask_mlp/main_fedAvg.py.

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
