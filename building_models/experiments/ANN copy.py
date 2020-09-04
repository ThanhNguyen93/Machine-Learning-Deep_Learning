import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import Dense, Input, Dropout
from keras.models import Model
from keras.engine import Layer


class ANN:
    def __init__(
            self,
            x_dim,
            y_dim,
            h1=100,
            h2=50,
            h3=30,
            dropout_rate=None,
            y_loss="binary_crossentropy",
            y_activation="sigmoid",
            optimizer="adam"
    ):
        """
        Args:
           x_dim (int): number of features/columns in input data X
           y_dim (int): dimension of the predicted variable
           z_dim (int): dimension of the confounding variable
           h1 (int): number of neurons in the first hidden layer (from X to e)
           h2 (int): number of neurons in the second set of hidden layers (from e to y and e to z)
           z_inv_factor (float): scaler applied to the gradient at the reversal step
           loss_weights (dict or None): dictionary indicating the weight of losses on y and z (e.g. dict(y=10, z=3)).
                                        Default value (None) makes all the weights equal to 1.
           dropout_rate (float between 0 and 1): ratio of features to randomly drop between X and e at fitting time
           {y,z}_loss: loss function to associate to this variable (default to binary crossentropy)
           {y,z}_activation: activation function to associate to this variable (default to sigmoid)
           optimizer: optimizer to use during training
        """
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h1, self.h2, self.h3 = h1, h2, h3
        self.dropout_rate = dropout_rate
        self.y_loss = y_loss
        self.y_activation = y_activation
        self.optimizer = optimizer

    def _build_model(self):
        x_input = Input((self.x_dim,), name="x_input")
        h1 = Dense(self.h1, activation="relu", kernel_regularizer="l2", name="h1")(x_input)

        # Predict y
        h2 = Dense(self.h2, activation="relu", kernel_regularizer="l2")(h1)
        h3= Dense(self.h3, activation="relu", kernel_regularizer="l2")(h2)
        y = Dense(self.y_dim, name="y", activation=self.y_activation)(l3)

        # Create the full model and compile it
        self.model = Model(x_input, y)
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.y_loss,
            metrics=['accuracy']
        )

        # Expose a model that predicts the target variable only
        self.clf = Model(x_input, y)
        
    def summary(self):
        return self.summary()

    def fit(self, X_train, y_train, *args, **kwargs):
        # Reset the Tensorflow graph to avoid resource exhaustion
        K.clear_session()
        # Build a fresh model
        self._build_model()
        self.h = self.model.fit(X_train, y_train, *args, **kwargs)
        return self.h

    def predict(self, *args, **kwargs):
        return self.clf.predict(*args, **kwargs)
