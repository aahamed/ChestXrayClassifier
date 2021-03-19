# Chest-XRay Multilabel Classification

* Multilabel classification of thoracic diseases using the Chest X-Ray 14 dataset provided by NIH
* Dataset paper: https://nihcc.app.box.com/v/ChestXray-NIHCC 

## Usage

* Define the configuration for your experiment. See `default.json` to see the structure and available options. You are free to modify and restructure the configuration as per your needs.
* Implement factories to return project specific models, datasets based on config. Add more flags as per requirement in the config.
* Implement `experiment.py` based on the project requirements.
* After defining the configuration (say `my_exp.json`) - simply run `python3 main.py my_exp` to start the experiment
* The logs, stats, plots and saved models would be stored in `./experiment_data/my_exp` dir. This can be configured in `contants.py`
* To resume an ongoing experiment, simply run the same command again. It will load the latest stats and models and resume training pr evaluate performance.

## Files
- main.py: Main driver class
- experiment.py: Main experiment class. Initialized based on config - takes care of training, saving stats and plots, logging and resuming experiments.
- dataset_factory: Factory to build datasets based on config
- model_factory.py: Factory to build models based on config
- constants.py: constants used across the project
- file_utils.py: utility functions for handling files
- chest_xray_dataloader.py: Class to create dataloader for NIH Chest X-Ray dataset
- models.py: Class containing ResNet50 based chest x-ray classifier
- models_all.py: Class containing all the U-Net based chest x-ray classifiers

## Data
- You will need to download the NIH dataset and pass its location to the `images_root_dir` key of the configuration json file. 
See `default.json` for an example.
- The `./data` directory contains the different splits we used for training and the `labels.csv` file.
- The different splits are: balanced, 5k, 10k, 20k, 40k, full and original.
- You may pass the desired split you would like to train on by updating the `data_split` key in the configuration json file.   
