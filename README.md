# Kaggle-question-pairs-quora

Solution for the Quora Question Pair contest hosted on Kaggle[[1]](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs)  
Big thanks to the authors of all kernels & posts, which were of great inspiration and some features were derived based on them.  
Kaggle Profile : [Daft Vader](https://www.kaggle.com/syeddanish)

## Model Implementation

Final submission is the ensemble of **7 LSTM** based models and **6 xgboost** models with different hyperparameters. LSTM model architecture is based on lystdo[[2]](https://www.kaggle.com/lystdo/lb-0-18-lstm-with-glove-and-magic-features) model on kaggle kernel. Keras architecture of the model is shown below:


![[model_1.png]](model_1.png)


Nadam optimizer and binary cross-entropy is used as  the loss function, early stopping round by monitoring validation loss is used to avoid overfitting. To create different permutations of hyper-parameter in both xgboost model and lstm, values of hyper-parameter is chosen at random from a defined set.

## Experiments

Dataset is split into two parts in 90:10 ratio. Each LSTM model was trained for approx 19 eochs and optimal weights were saved using model checkpoint by monitoring validation loss, each epoch took around 220s on Nvidia GTX 1070 gpu and each xgboost model is trained using early stoping round of 50. The configration gives a logloss of 0.14672 on private leaderboard.

| Model | LogLoss |
|-----------:|:------------:|
| Avg(7 LSTM + 6 XGB) | 0.14672 |
| LSTM Model | 0.15918 |
| XGB Model | 0.15392 |

## Requirements

* Python 2.7.13
* jupyter 4.3.1

## Usage

This repository contains 5 notebooks performing different function, it can be used by simply running the notebook server by using standard Jupyter command:

```
$ jupyter notebook
```

`feature_engineering.ipynb` : Basic feature Generation  
`page_rank.ipynb` : Page rank feature generation  
`glove_840b+leaky_features.ipynb` : LSTM model training  
`xgboost.ipynb` : XG Boost model training  
`averagging.ipynb` : Ensemblling the models

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details
