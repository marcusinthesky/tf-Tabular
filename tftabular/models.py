from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from .layers import TabNetDecoder, TabNetEncoder



@tf.function
def identity(x):
    return x

class TabNetClassifier(tf.keras.Model):
    def __init__(
        self,
        outputs: int = 1,
        n_steps: int = 3,
        n_features: int = 8,
        gamma: float = 1.3,
        epsilon: float = 1e-8,
        sparsity: float = 1e-5,
        feature_column: Optional[tf.keras.layers.DenseFeatures] = None,
        pretrained_encoder: Optional[tf.keras.layers.Layer] = None,
        virtual_batch_size: Optional[int] = 128,
        momentum: Optional[float] = 0.02,
    ):
        super(TabNetClassifier, self).__init__()

        self.outputs = outputs
        self.n_steps = n_steps
        self.n_features = n_features
        self.feature_column = feature_column
        self.pretrained_encoder = pretrained_encoder
        self.virtual_batch_size = virtual_batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.momentum = momentum
        self.sparsity = sparsity

        if feature_column is None:
            self.feature = tf.keras.layers.Lambda(identity)
        else:
            self.feature = feature_column

        if pretrained_encoder is None:
            self.encoder = TabNetEncoder(
                units=outputs,
                n_steps=n_steps,
                n_features=n_features,
                outputs=outputs,
                gamma=gamma,
                epsilon=epsilon,
                sparsity=sparsity,
                virtual_batch_size=self.virtual_batch_size,
                momentum=momentum,
            )
        else:
            self.encoder = pretrained_encoder

    def forward(
        self, X: Union[tf.Tensor, np.ndarray], training: Optional[bool] = None
    ) -> Tuple[tf.Tensor]:
        X = self.feature(X)
        output, encoded, importance = self.encoder(X)

        prediction = tf.keras.activations.sigmoid(output)
        return prediction, encoded, importance

    def call(
        self, X: Union[tf.Tensor, np.ndarray], training: Optional[bool] = None
    ) -> tf.Tensor:
        prediction, _, _ = self.forward(X)
        return prediction

    def transform(
        self, X: Union[tf.Tensor, np.ndarray], training: Optional[bool] = None
    ) -> tf.Tensor:
        _, encoded, _ = self.forward(X)
        return encoded

    def explain(
        self, X: Union[tf.Tensor, np.ndarray], training: Optional[bool] = None
    ) -> tf.Tensor:
        _, _, importance = self.forward(X)
        return importance


class TabNetAutoencoder(tf.keras.Model):
    def __init__(
        self,
        outputs: int = 1,
        inputs: int = 12,
        n_steps: int = 3,
        n_features: int = 8,
        gamma: float = 1.3,
        epsilon: float = 1e-8,
        sparsity: float = 1e-5,
        feature_column: Optional[tf.keras.layers.DenseFeatures] = None,
        virtual_batch_size: Optional[int] = 128,
        momentum: Optional[float] = 0.02,
    ):
        super(TabNetAutoencoder, self).__init__()

        self.outputs = outputs
        self.inputs = inputs
        self.n_steps = n_steps
        self.n_features = n_features
        self.feature_column = feature_column
        self.virtual_batch_size = virtual_batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.momentum = momentum
        self.sparsity = sparsity

        if feature_column is None:
            self.feature = tf.keras.layers.Lambda(identity)
        else:
            self.feature = feature_column

        self.encoder = TabNetEncoder(
            units=outputs,
            n_steps=n_steps,
            n_features=n_features,
            outputs=outputs,
            gamma=gamma,
            epsilon=epsilon,
            sparsity=sparsity,
            virtual_batch_size=self.virtual_batch_size,
            momentum=momentum,
        )

        self.decoder = TabNetDecoder(
            units=inputs,
            n_steps=n_steps,
            n_features=n_features,
            virtual_batch_size=self.virtual_batch_size,
            momentum=momentum,
        )

        self.bn = tf.keras.layers.BatchNormalization(
            virtual_batch_size=self.virtual_batch_size, momentum=momentum
        )

        self.do = tf.keras.layers.Dropout(0.25)

    def forward(
        self, X: Union[tf.Tensor, np.ndarray], training: Optional[bool] = None
    ) -> Tuple[tf.Tensor]:
        X = self.feature(X)
        X = self.bn(X)

        # training mask
        M = self.do(tf.ones_like(X), training=training)
        D = X * M

        # encoder
        output, encoded, importance = self.encoder(D)
        prediction = tf.keras.activations.sigmoid(output)

        return prediction, encoded, importance, X, M

    def call(
        self, X: Union[tf.Tensor, np.ndarray], training: Optional[bool] = None
    ) -> tf.Tensor:
        # encode
        prediction, encoded, _, X, M = self.forward(X)
        T = X * (1 - M)

        # decode
        reconstruction = self.decoder(encoded)

        # loss
        loss = tf.reduce_mean(
            tf.where(
                M != 0.0, tf.square(T - reconstruction), tf.zeros_like(reconstruction)
            )
        )

        self.add_loss(loss)

        return prediction

    def transform(
        self, X: Union[tf.Tensor, np.ndarray], training: Optional[bool] = None
    ) -> tf.Tensor:
        _, encoded, _, _, _ = self.forward(X)
        return encoded

    def explain(
        self, X: Union[tf.Tensor, np.ndarray], training: Optional[bool] = None
    ) -> tf.Tensor:
        _, _, importance, _, _ = self.forward(X)
        return importance
