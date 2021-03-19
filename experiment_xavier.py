# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 21:49:04 2021

@author: brand
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime

from sklearn.metrics import roc_auc_score, roc_curve


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.zeros_(m.bias)


# Class to encapsulate a neural experiment.
# The boilerplate code to setup the experiment, log stats, checkpoints and plotting have been provided to you.
# You only need to implement the main training logic of your experiment and implement train, val and test methods.
# You are free to modify or restructure the code as per your convenience.
class Experiment(object):
    def __init__(self, name):
        config_data = read_file_in_dir('./', name + '.json')
        if config_data is None:
            raise Exception("Configuration file doesn't exist: ", name)

        self.name = config_data['experiment_name']
        self.experiment_dir = os.path.join(ROOT_STATS_DIR, self.name)

        # Load Datasets
        self.train_loader, self.val_loader, self.test_loader = get_datasets(config_data)

        # Setup Experiment
        self.epochs = config_data['experiment']['num_epochs']
        lr = config_data['experiment']['learning_rate']
        wd = config_data['experiment']['weight_decay']
        momentum = config_data["experiment"]["momentum"]
        self.current_epoch = 0
        self.training_losses = []
        self.val_losses = []
        self.best_model = None  # Save your best model in this field and use this in test method.

        # Init Model
        self.model = get_model(config_data)

        # TODO: Set these Criterion and Optimizers Correctly
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=wd, momentum=momentum)

        self.init_model()

        # Load Experiment Data if available
        self.load_experiment()

    # Loads the experiment data if exists to resume training from last saved checkpoint.
    def load_experiment(self):
        os.makedirs(ROOT_STATS_DIR, exist_ok=True)

        if os.path.exists(self.experiment_dir):
            self.training_losses = read_file_in_dir(self.experiment_dir, 'training_losses.txt')
            self.val_losses = read_file_in_dir(self.experiment_dir, 'val_losses.txt')
            self.current_epoch = len(self.training_losses)

            state_dict = torch.load(os.path.join(self.experiment_dir, 'latest_model.pt'))
            self.model.load_state_dict(state_dict['model'])
            self.optimizer.load_state_dict(state_dict['optimizer'])

        else:
            os.makedirs(self.experiment_dir)

    def init_model(self):
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.model = self.model.apply(init_weights)
            self.criterion = self.criterion.cuda()

    # Main method to run your experiment. Should be self-explanatory.
    def run(self):
        start_epoch = self.current_epoch
        for epoch in range(start_epoch, self.epochs):  # loop over the dataset multiple times
            start_time = datetime.now()
            self.current_epoch = epoch
            train_loss = self.train()
            val_loss = self.val()
            self.record_stats(train_loss, val_loss)
            self.log_epoch_stats(start_time)
            self.save_model()

    # Perform one training iteration on the whole dataset and return loss value
    def train(self):
        self.model.train()
        self.optimizer.zero_grad()
        for i, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to( device ), labels.to( device )
            outputs = self.model( images )

            total_label_count = len(images)
            pos_count_per_class = torch.sum(labels, dim=0)
            pos_count_per_class[pos_count_per_class == 0] = total_label_count
            neg_count_per_class = total_label_count - pos_count_per_class
            pos_weights = neg_count_per_class / pos_count_per_class
            self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)
            loss = self.criterion( outputs, labels )

            # optimize
            loss.backward()

            if i % 2 == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
        
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.model.eval()
        training_loss = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.train_loader):
                total_label_count = len(images)
                pos_count_per_class = torch.sum(labels, dim=0)
                pos_count_per_class[pos_count_per_class == 0] = total_label_count
                neg_count_per_class = total_label_count - pos_count_per_class
                pos_weights = neg_count_per_class / pos_count_per_class
                self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)
                training_loss += loss.item()

        return training_loss / i

    # Perform one pass on the validation set and return loss value
    def val(self):
        self.model.eval()
        val_loss = 0

        with torch.no_grad():
            for i, (images, labels) in enumerate(self.val_loader):
                images, labels = images.to( device ), labels.to( device )
                outputs = self.model( images )
                
                total_label_count = len(images)
                pos_count_per_class = torch.sum(labels, dim=0)
                pos_count_per_class[pos_count_per_class == 0] = total_label_count
                neg_count_per_class = total_label_count - pos_count_per_class
                pos_weights = neg_count_per_class / pos_count_per_class
                self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)
                loss = self.criterion( outputs, labels )
                val_loss += loss.item()

        return val_loss / i

    # Perform one pass on the test set and return loss value
    def test(self):
        self.model.eval()
        test_loss = 0
        true_positives = torch.zeros(15).to(device)
        true_negatives = torch.zeros(15).to(device)
        false_positives = torch.zeros(15).to(device)
        false_negatives = torch.zeros(15).to(device)
        sigmoid = torch.nn.Sigmoid()
        all_labels = torch.empty(0, 15).to(device)
        all_probs = torch.empty(0, 15).to(device)

        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                images, labels = images.to( device ), labels.to( device )
                outputs = self.model( images )
                
                total_label_count = len(images)
                pos_count_per_class = torch.sum(labels, dim=0)
                pos_count_per_class[pos_count_per_class == 0] = total_label_count
                neg_count_per_class = total_label_count - pos_count_per_class
                pos_weights = neg_count_per_class / pos_count_per_class
                self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)
                loss = self.criterion( outputs, labels )
                test_loss += loss.item()

                probs = sigmoid(outputs)
                one_hot_outputs = torch.round(probs)
                
                #for label, one_hot in zip(labels, one_hot_outputs):
                    #true_positives += np.logical_and((label == 1).cpu(), (one_hot == 1).cpu()).to(device)
                    #true_negatives += np.logical_and((label == 0).cpu(), (one_hot == 0).cpu()).to(device)
                    #false_positives += np.logical_and((label == 0).cpu(), (one_hot == 1).cpu()).to(device)
                    #false_negatives += np.logical_and((label == 1).cpu(), (one_hot == 0).cpu()).to(device)
                
                all_labels = torch.cat((all_labels, labels))
                all_probs = torch.cat((all_probs, probs))

        test_loss = test_loss / i
        plt.figure()
        for i in range(15):
            fpr, tpr, _ = roc_curve(all_labels[:,i].cpu(), all_probs[:,i].cpu())
            plt.plot(fpr, tpr, label=f"{list(LABEL_ENCODINGS.keys())[i]}")

            auc = roc_auc_score(all_labels[:, i].cpu(), all_probs[:, i].cpu())
            self.log(f"{list(LABEL_ENCODINGS.keys())[i]} AUC: {auc:.3f}")
            
        plt.xlabel("Specificity")
        plt.ylabel("Sensitivity")
        plt.legend(loc="best")
        plt.title("ROC Curve")
        plt.savefig(os.path.join(self.experiment_dir, "roc_curve.png"))

        result_str = f'Test Performance: Loss: {test_loss}'
        self.log(result_str)

        return test_loss

    def save_model(self):
        root_model_path = os.path.join(self.experiment_dir, 'latest_model.pt')
        model_dict = self.model.state_dict()
        state_dict = {'model': model_dict, 'optimizer': self.optimizer.state_dict()}
        torch.save(state_dict, root_model_path)

    def record_stats(self, train_loss, val_loss):
        self.training_losses.append(train_loss)
        self.val_losses.append(val_loss)

        self.plot_stats()

        write_to_file_in_dir(self.experiment_dir, 'training_losses.txt', self.training_losses)
        write_to_file_in_dir(self.experiment_dir, 'val_losses.txt', self.val_losses)

    def log(self, log_str, file_name=None):
        print(log_str)
        log_to_file_in_dir(self.experiment_dir, 'all.log', log_str)
        if file_name is not None:
            log_to_file_in_dir(self.experiment_dir, file_name, log_str)

    def log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.epochs - self.current_epoch - 1)
        train_loss = self.training_losses[self.current_epoch]
        val_loss = self.val_losses[self.current_epoch]
        summary_str = "Epoch: {}, Train Loss: {}, Val Loss: {}, Took {}, ETA: {}\n"
        summary_str = summary_str.format(self.current_epoch + 1, train_loss, val_loss, str(time_elapsed),
                                         str(time_to_completion))
        self.log(summary_str, 'epoch.log')

    def plot_stats(self):
        e = len(self.training_losses)
        x_axis = np.arange(1, e + 1, 1)
        plt.figure()
        plt.plot(x_axis, self.training_losses, label="Training Loss")
        plt.plot(x_axis, self.val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        plt.title(self.name + " Stats Plot")
        plt.savefig(os.path.join(self.experiment_dir, "stat_plot.png"))