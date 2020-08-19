from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf

import tensorflow_addons as tfa


class GLUBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        units: Optional[int] = None,
        virtual_batch_size: Optional[int] = 128,
        momentum: Optional[float] = 0.02,
    ):
        super(GLUBlock, self).__init__()
        self.units = units
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum

    def build(self, input_shape: tf.TensorShape):
        if self.units is None:
            self.units = input_shape[-1]

        self.fc_outout = tf.keras.layers.Dense(self.units, use_bias=False)
        self.bn_outout = tf.keras.layers.BatchNormalization(
            virtual_batch_size=self.virtual_batch_size, momentum=self.momentum
        )

        self.fc_gate = tf.keras.layers.Dense(self.units, use_bias=False)
        self.bn_gate = tf.keras.layers.BatchNormalization(
            virtual_batch_size=self.virtual_batch_size, momentum=self.momentum
        )

    def call(
        self, inputs: Union[tf.Tensor, np.ndarray], training: Optional[bool] = None
    ):
        output = self.bn_outout(self.fc_outout(inputs), training=training)
        gate = self.bn_gate(self.fc_gate(inputs), training=training)

        return output * tf.keras.activations.sigmoid(gate)  # GLU


class FeatureTransformerBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        units: Optional[int] = None,
        virtual_batch_size: Optional[int] = 128,
        momentum: Optional[float] = 0.02,
        skip: bool = False
    ):
        super(FeatureTransformerBlock, self).__init__()
        self.units = units
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self.skip = skip

    def build(self, input_shape: tf.TensorShape):
        if self.units is None:
            self.units = input_shape[-1]

        self.initial = GLUBlock(
            units=self.units,
            virtual_batch_size=self.virtual_batch_size,
            momentum=self.momentum,
        )
        self.residual = GLUBlock(
            units=self.units,
            virtual_batch_size=self.virtual_batch_size,
            momentum=self.momentum,
        )

    def call(
        self, inputs: Union[tf.Tensor, np.ndarray], training: Optional[bool] = None
    ):
        initial = self.initial(inputs, training=training)

        if self.skip:
            initial = (initial + inputs) * np.sqrt(0.5)
        
        residual = self.residual(initial, training=training)  # skip

        return (initial + residual) * np.sqrt(0.5)


class AttentiveTransformer(tf.keras.layers.Layer):
    def __init__(
        self,
        units: Optional[int] = None,
        virtual_batch_size: Optional[int] = 128,
        momentum: Optional[float] = 0.02,
    ):
        super(AttentiveTransformer, self).__init__()
        self.units = units
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum

    def build(self, input_shape: tf.TensorShape):
        if self.units is None:
            self.units = input_shape[-1]

        self.fc = tf.keras.layers.Dense(self.units, use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization(
            virtual_batch_size=self.virtual_batch_size, momentum=self.momentum
        )

    def call(
        self,
        inputs: Union[tf.Tensor, np.ndarray],
        priors: Union[tf.Tensor, np.ndarray],
        training: Optional[bool] = None,
    ) -> tf.Tensor:
        feature = self.bn(self.fc(inputs), training=training)
        output = feature * priors

        return tfa.activations.sparsemax(output)


class TabNetStep(tf.keras.layers.Layer):
    def __init__(
        self,
        units: Optional[int] = None,
        virtual_batch_size: Optional[int] = 128,
        momentum: Optional[float] = 0.02,
    ):
        super(TabNetStep, self).__init__()
        self.units = units
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum

    def build(self, input_shape: tf.TensorShape):
        if self.units is None:
            self.units = input_shape[-1]

        self.unique = FeatureTransformerBlock(
            units=self.units,
            virtual_batch_size=self.virtual_batch_size,
            momentum=self.momentum,
            skip=True
        )
        self.attention = AttentiveTransformer(
            units=input_shape[-1],
            virtual_batch_size=self.virtual_batch_size,
            momentum=self.momentum,
        )

    def call(self, inputs, shared, priors, training=None) -> Tuple[tf.Tensor]:
        split = self.unique(shared, training=training)
        keys = self.attention(split, priors, training=training)
        masked = keys * inputs

        return split, masked, keys


class TabNetEncoder(tf.keras.layers.Layer):
    def __init__(
        self,
        units: int = 1,
        n_steps: int = 3,
        n_features: int = 8,
        outputs: int = 1,
        gamma: float = 1.3,
        epsilon: float = 1e-8,
        sparsity: float = 1e-5,
        virtual_batch_size: Optional[int] = 128,
        momentum: Optional[float] = 0.02,
    ):
        super(TabNetEncoder, self).__init__()

        self.units = units
        self.n_steps = n_steps
        self.n_features = n_features
        self.virtual_batch_size = virtual_batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.momentum = momentum
        self.sparsity = sparsity

    def build(self, input_shape: tf.TensorShape):
        self.bn = tf.keras.layers.BatchNormalization(
            virtual_batch_size=self.virtual_batch_size, momentum=self.momentum
        )
        self.shared_block = FeatureTransformerBlock(
            units=self.n_features,
            virtual_batch_size=self.virtual_batch_size,
            momentum=self.momentum,
        )
        self.initial_step = TabNetStep(
            units=self.n_features,
            virtual_batch_size=self.virtual_batch_size,
            momentum=self.momentum,
        )
        self.steps = [
            TabNetStep(
                units=self.n_features,
                virtual_batch_size=self.virtual_batch_size,
                momentum=self.momentum,
            )
            for _ in range(self.n_steps)
        ]
        self.final = tf.keras.layers.Dense(units=self.units, use_bias=False)

    def call(
        self, X: Union[tf.Tensor, np.ndarray], training: Optional[bool] = None
    ) -> Tuple[tf.Tensor]:
        entropy_loss = 0.0
        encoded = 0.0
        output = 0.0
        importance = 0.0
        prior = tf.reduce_mean(tf.ones_like(X), axis=0)

        B = prior * self.bn(X, training=training)
        shared = self.shared_block(B, training=training)
        _, masked, keys = self.initial_step(B, shared, prior, training=training)

        for step in self.steps:
            entropy_loss += tf.reduce_mean(
                tf.reduce_sum(-keys * tf.math.log(keys + self.epsilon), axis=-1)
            ) / tf.cast(self.n_steps, tf.float32)
            prior *= self.gamma - tf.reduce_mean(keys, axis=0)
            importance += keys

            shared = self.shared_block(masked, training=training)
            split, masked, keys = step(B, shared, prior, training=training)
            features = tf.keras.activations.relu(split)

            output += features
            encoded += split

        self.add_loss(self.sparsity * entropy_loss)

        prediction = self.final(output)
        return prediction, encoded, importance


class TabNetDecoder(tf.keras.layers.Layer):
    def __init__(
        self,
        units=1,
        n_steps=3,
        n_features=8,
        outputs=1,
        gamma=1.3,
        epsilon=1e-8,
        sparsity=1e-5,
        virtual_batch_size=128,
        momentum=0.02,
    ):
        super(TabNetDecoder, self).__init__()

        self.units = units
        self.n_steps = n_steps
        self.n_features = n_features
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum

    def build(self, input_shape: tf.TensorShape):
        self.shared_block = FeatureTransformerBlock(
            units=self.n_features,
            virtual_batch_size=self.virtual_batch_size,
            momentum=self.momentum,
        )
        self.steps = [
            FeatureTransformerBlock(
                units=self.n_features,
                virtual_batch_size=self.virtual_batch_size,
                momentum=self.momentum,
            )
            for _ in range(self.n_steps)
        ]
        self.fc = [tf.keras.layers.Dense(units=self.units) for _ in range(self.n_steps)]

    def call(
        self, X: Union[tf.Tensor, np.ndarray], training: Optional[bool] = None
    ) -> tf.Tensor:
        decoded = 0.0

        for ftb, fc in zip(self.steps, self.fc):
            shared = self.shared_block(X, training=training)
            feature = ftb(shared, training=training)
            output = fc(feature)

            decoded += output
        return decoded
