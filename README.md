# 6711-Project-Group14
**Problem Statement**: while previous works have focused on classifying jamming attacks using ML and some focus on physically locating jammers, there is no unified framework to perform both tasks simultaneously. Furthermore, previous approaches have relied on centralized data processing, creating privacy risks for client devices and communications. We introduce a solution using federated learning to preserve client privacy while performing jamming classification and localization with a multi-task Multi-Layer Perceptron.

## Contents

```
.
├── charts
│   ├── compare_mlp
│   │   ├── comparison_mlp_vs_fedavg_binary_confusion.png
│   │   ├── comparison_mlp_vs_fedavg_binary_overall.png
│   │   ├── comparison_mlp_vs_fedavg_binary_per_class_f1.png
│   │   ├── comparison_mlp_vs_fedavg_binary_roc.png
│   │   ├── comparison_mlp_vs_fedavg_binary_training_curves_and_time.png
│   │   ├── comparison_mlp_vs_fedavg_multiclass_confusion.png
│   │   ├── comparison_mlp_vs_fedavg_multiclass_overall.png
│   │   ├── comparison_mlp_vs_fedavg_multiclass_per_class_f1.png
│   │   └── comparison_mlp_vs_fedavg_multiclass_training_curves_and_time.png
│   ├── compare_models
│   │   ├── comparison_binary_rf_vs_fedavg_confusion.png
│   │   ├── comparison_binary_rf_vs_fedavg_overall.png
│   │   ├── comparison_binary_rf_vs_fedavg_per_class_f1.png
│   │   ├── comparison_binary_rf_vs_fedavg_roc.png
│   │   ├── comparison_multiclass_rf_vs_fedavg_confusion.png
│   │   ├── comparison_multiclass_rf_vs_fedavg_overall.png
│   │   └── comparison_multiclass_rf_vs_fedavg_per_class_f1.png
│   ├── dataset
│   │   ├── dataset_class_distribution_binary.png
│   │   ├── dataset_class_distribution_multiclass.png
│   │   ├── dataset_throughput_binary.png
│   │   ├── dataset_throughput_multiclass.png
│   │   └── MDS_grid.png
│   ├── fedAvg
│   │   ├── fedavg_binary_confusion_matrix.png
│   │   ├── fedavg_binary_metrics.csv
│   │   ├── fedavg_binary_per_class_metrics.png
│   │   ├── fedavg_binary_report.json
│   │   ├── fedavg_binary_roc_curve.png
│   │   ├── fedavg_binary_training_curves.png
│   │   ├── fedavg_multiclass_confusion_matrix.png
│   │   ├── fedavg_multiclass_metrics.csv
│   │   ├── fedavg_multiclass_per_class_metrics.png
│   │   ├── fedavg_multiclass_report.json
│   │   └── fedavg_multiclass_training_curves.png
│   ├── mlp
│   │   ├── mlp_binary_confusion_matrix.png
│   │   ├── mlp_binary_metrics.csv
│   │   ├── mlp_binary_per_class_metrics.png
│   │   ├── mlp_binary_report.json
│   │   ├── mlp_binary_roc_curve.png
│   │   ├── mlp_binary_training_curves.png
│   │   ├── mlp_multiclass_confusion_matrix.png
│   │   ├── mlp_multiclass_metrics.csv
│   │   ├── mlp_multiclass_per_class_metrics.png
│   │   ├── mlp_multiclass_report.json
│   │   └── mlp_multiclass_training_curves.png
│   └── rf
│       ├── rf_binary_confusion_matrix.png
│       ├── rf_binary_feature_importance.png
│       ├── rf_binary_metrics.csv
│       ├── rf_binary_per_class_metrics.png
│       ├── rf_binary_report.json
│       ├── rf_binary_roc_curve.png
│       ├── rf_multiclass_confusion_matrix.png
│       ├── rf_multiclass_feature_importance.png
│       ├── rf_multiclass_metrics.csv
│       ├── rf_multiclass_per_class_metrics.png
│       └── rf_multiclass_report.json
├── data
│   └── WSN-DS.csv
├── README.md
└── src
    ├── compare_models.py
    ├── mlp
    │   ├── compare_mlp.py
    │   ├── evaluate_centralized_mlp.py
    │   ├── evaluate_fedAvg.py
    │   ├── main_fedAvg.py
    │   ├── main_MLP.py
    │   └── mlp_model.py
    ├── multitask_mlp
    │   ├── args.py
    │   ├── localize.py
    │   ├── main_fedAvg.py
    │   ├── mlp_model.py
    │   └── preprocess.py
    ├── preprocessing
    │   ├── args.py
    │   ├── dataset_analysis.py
    │   └── preprocess.py
    ├── requirements.txt
    └── rf
        ├── evaluate_rf.py
        └── rf_model.py
```

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
