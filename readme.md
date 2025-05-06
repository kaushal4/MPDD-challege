# How to run it

## Getting a python venv
make sure you have Python version `3.13.2` available on path as python3 and execute the following commands in folder

```
python3 -m venv .venv
```

This will create a virtual environment in a folder .venv in the current directory.

Now source the python version (The commands may differ if you are on windows)
```
source .venv/bin/activate
```

Now install the requirements

```
pip install -r requirements.txt
```

## Running the model

At this point you should be able to run any model in the base folder

For example:
```
python run_cross_attention_transformer.py
```

You should be able to run all models direcrtly now

## Note
If you are on MAC (M chips) you don't have to worry about device (mps selected by default).
Otherwise, you may have to change the device in config and some scripts require you to manually set this as a global variable.
Since we used MACOS to run the models we can't test on other devices

# Project Structures
Representation of folders:
- DataLoader: This folder contains standard data loader function to run training/eval on
- Datasets: This folder has code to generate a dataset for our data
- Dataclasses: Standard data holders to ease
- Results: Compilation of our results and scripts to generate some figures
- Utils: common utilities
- run_*.py: script to run the respective model

# Contributions
- MLP and baseline models : Aryamaan Saha
- LSTM and basic transformers: Niranjan
- Transformers and regularization: Kaushal

# Note
Since the data provided here is fake, you won't get good results.
To get the original data, please contact the MPDD challenge team

## Config explanation (not necesarry to understand)

-  data_root_path: path to your data directory (Elderly or Young)
-  window_split_time: 1 
-  audio_feature_method : Audio feature type, options {wav2vec, opensmile, mfccs}
-  video_feature_method : Video feature type, options {openface, resnet, densenet}
-  labelcount: Number of labels, e.g., 2 for binary classification {1,2,5}
-  track_option: Track option, e.g., "Track1"
-  feature_max_len: Maximum length of features, e.g., 26
-  batch_size: Batch size for training, e.g., 16 (Optional based on your model)
-  lr: Learning rate, e.g., 0.00002 (Optional based on your model)
-  num_epochs: Number of epochs for training, e.g., 200 (Optional Based on your model)
-  device: Device to use for computation, e.g., "mps"
-  seed: random seed