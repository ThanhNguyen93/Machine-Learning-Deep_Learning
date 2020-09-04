import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import Dense, Input, Dropout
from keras.models import Model
from keras.engine import Layer

# Reverse gradient layer from https://github.com/michetonu/gradient_reversal_keras_tf/blob/master/flipGradientTF.py
# - Added compute_output_shape for Keras 2 compatibility
# - Fixed bug where RegisterGradient was raising a KeyError
def reverse_gradient(X, hp_lambda):
    """Flips the sign of the incoming gradient during training."""
    try:
        reverse_gradient.num_calls += 1
    except AttributeError:
        reverse_gradient.num_calls = 1

    while True:
        try:
            grad_name = "GradientReversal%d" % reverse_gradient.num_calls

            @tf.RegisterGradient(grad_name)
            def _flip_gradients(op, grad):
                return [tf.negative(grad) * hp_lambda]

            break
        except KeyError:
            reverse_gradient.num_calls += 1

    g = K.get_session().graph
    with g.gradient_override_map({"Identity": grad_name}):
        y = tf.identity(X)
    return y


class GradientReversal(Layer):
    """Flip the sign of gradient during training."""

    def __init__(self, hp_lambda, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        self.supports_masking = False
        self.hp_lambda = hp_lambda

    def build(self, input_shape):
        self.trainable_weights = []

    def call(self, x, mask=None):
        return reverse_gradient(x, self.hp_lambda)

    def get_output_shape_for(self, input_shape):
        return input_shape

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {"hp_lambda": self.hp_lambda}
        base_config = super(GradientReversal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class DANN:
    def __init__(
            self,
            x_dim,
            y_dim,
            z_dim,
            h1=100,
            h2=50,
            z_inv_factor=1,
            loss_weights=None,
            dropout_rate=.1,
            y_loss="binary_crossentropy",
            y_activation="sigmoid",
           z_loss="binary_crossentropy",
            z_activation="sigmoid",
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
        self.z_dim = z_dim
        self.h1, self.h2 = h1, h2
        self.z_inv_factor = z_inv_factor
        self.loss_weights = loss_weights
        self.dropout_rate = dropout_rate
        self.y_loss = y_loss
        self.y_activation = y_activation
        self.z_loss = z_loss
        self.z_activation = z_activation
        self.optimizer = optimizer

    def _build_model(self):
        x_input = Input((self.x_dim,), name="x_input")
        e = Dropout(self.dropout_rate)(x_input)
        e = Dense(self.h1, activation="relu", name="e")(e)

        # Predict y
        l = Dense(self.h2, activation="relu")(e)
        y = Dense(self.y_dim, name="y", activation=self.y_activation)(l)

        # Predict z with gradient reversal
        l = GradientReversal(self.z_inv_factor)(e)
        l = Dense(self.h2, activation="relu")(l)
        z = Dense(self.z_dim, name="z", activation=self.z_activation)(l)

        # Create the full model and compile it
        self.model = Model(x_input, [y, z])
        self.model.compile(
            optimizer=self.optimizer if self.optimizer is not None else Adam(lr=0.0001, beta_1=0.99),
            loss=[self.y_loss, self.z_loss],
            loss_weights=self.loss_weights,
            metrics=['accuracy']
        )

        # Expose a model that predicts the target variable only
        self.clf = Model(x_input, y)
        print(self.model.summary())

#     def fit(self, *args, **kwargs):
    def fit(self, X_train, y_train, z, *args, **kwargs):
        # Reset the Tensorflow graph to avoid resource exhaustion
        K.clear_session()
        # Build a fresh model
        self._build_model()
        
        vd = kwargs.get("validation_data", ())
        if type(vd) != tuple: 
            kwargs["validation_data"] = (vd.X, [vd.y, vd.z])
        inputs = X_train #d.X
        outputs = [y_train, z] #[d.y, d.z]
   
        self.h = self.model.fit(inputs, outputs, *args, **kwargs)
        return self.h

    def predict(self, *args, **kwargs):
        return self.clf.predict(*args, **kwargs)
