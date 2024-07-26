""" Model(s) """
import wandb
from src.config import EPOCHS, LEARNING_RATE, P, LAMBDA_, MAX_PATIENCE, DROPOUT, RANDOM_STATE

from typing import Optional

import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
import matplotlib.pyplot as plt

from src.utils import calculate_loss_gradients, compute_mse

wandb.require("core")

# ================================================================================================================ #

class LinearRegressionModel():
    """
    Class of linear regression models. 
    """
    def __init__(self,
                 weights_init:str='zero',
                 epochs:int=EPOCHS,
                 learning_rate:float=LEARNING_RATE,
                 p:int=P,
                 lambda_:float=LAMBDA_,
                 max_patience:int=MAX_PATIENCE,
                 dropout:float=DROPOUT,
                 random_state:int=RANDOM_STATE):

        self.weights = weights_init
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.p = p
        self.lambda_ = lambda_
        self.max_patience = max_patience
        self.dropout = dropout
        self.random_state = random_state
        self.losses_in_training_data = np.zeros(epochs)
        self.losses_in_validation_data = np.zeros(epochs)
        self.stopped_at = epochs # Records epoch stopped at. First assumes training using every epoch.

    def fit(self, X_train:jax.Array, y_train:jax.Array, X_val:Optional[jax.Array]=None, y_val:Optional[jax.Array]=None):
        """
        Fits model to data inputted.

        Args:
            X_train (jax.Array): training data of features.
            y_train (jax.Array): training data of labels.
            X_val (jax.Array): validation data of features.
            y_val (jax.Array): validation data of labels.
        """

        # Initializing parameters
        best_beta:Optional[jax.Array] = None

        n = len(X_train)

        best_mse = jnp.inf
        
        patience_counter = 0

        key = jax.random.key(self.random_state)

        # Defining the weights initializers
        weights_init_dict = {
            'zero': jnp.zeros(X_train.shape[1]),
            'random': jax.random.normal(key, shape=(X_train.shape[1],)),
            'lecun': jax.random.normal(key, shape=(X_train.shape[1],)) * jnp.sqrt(1/X_train.shape[0]),
            'xavier': jax.random.normal(key, shape=(X_train.shape[1],)) * jnp.sqrt(2/(X_train.shape[0]+y_train.shape[0])),
            'he': jax.random.normal(key, shape=(X_train.shape[1],)) * jnp.sqrt(2/X_train.shape[0])
        }

        self.weights = weights_init_dict[self.weights]

        # Training Loop
        for epoch in range(self.epochs):
            # Dropout
            key, subkey = jax.random.split(key)
            dropout_mask = jax.random.bernoulli(subkey, p=(1-self.dropout), shape=(n,1))
            X_train_dropout = X_train * dropout_mask

            # Calculating loss on training data
            mse_train = compute_mse(y_pred=jnp.dot(X_train_dropout, self.weights), y_true=y_train)
            wandb.log({'mse_train': mse_train})
            self.losses_in_training_data[epoch] = mse_train

            # Calculate loss gradients
            loss_gradient_wrt_beta = calculate_loss_gradients(self.weights, X_train_dropout, y_train, self.p, self.lambda_)

            # Optimiser step
            self.weights -= self.learning_rate * loss_gradient_wrt_beta


            if X_val is not None and y_val is not None:
                # Validation step
                mse_val = compute_mse(y_val, jnp.dot(X_val, self.weights))
                wandb.log({'mse_val': mse_val})
                self.losses_in_validation_data[epoch] = mse_val

                # Potential early stopping
                if mse_val < best_mse:
                    best_mse = mse_val
                    patience_counter = 0
                    best_beta = self.weights
                else:
                    patience_counter += 1

                if patience_counter >= self.max_patience:
                    print(f'Stopped at epoch {epoch+1}.')
                    self.stopped_at = epoch+1
                    break

        self.weights = best_beta

    def predict(self, X_test:jax.Array):
        """
        Predicts on test data.
        """
        return jnp.dot(X_test, self.weights)
    
    def plot_losses(self):
        """
        Plot losses.
        """
        plt.figure(figsize=(10,5))
        # Plotting training losses
        plt.title('MSE vs Epochs')
        plt.plot(range(self.stopped_at), self.losses_in_training_data[:self.stopped_at], c='blue', label='Training')
        # Plotting validation losses
        plt.plot(range(self.stopped_at), self.losses_in_validation_data[:self.stopped_at], c='orange', label='Valdation')
        plt.legend()
        plt.show()
