from __future__ import print_function

from keras import activations, initializers, constraints
from keras import regularizers
from keras.layers import Layer
import keras.backend as K


import keras

import tensorflow.keras.backend as K
from tensorflow.keras import activations
from tensorflow.keras.layers import Layer, Dense


class MPNNLayer(Layer):
    def __init__(self, activation=None, hidden_units=128, **kwargs):
        super(MPNNLayer, self).__init__(**kwargs)
        self.activation = activations.get(activation)
        self.hidden_units = hidden_units

    def build(self, input_shape):
        node_feature_dim = input_shape[0][-1]

        self.W_message = self.add_weight(
            shape=(node_feature_dim, node_feature_dim),
            initializer="glorot_uniform",
            trainable=True,
            name="W_message"
        )

        self.update_network = keras.Sequential([
            Dense(self.hidden_units, activation="relu"),
            Dense(node_feature_dim)
        ])

        super(MPNNLayer, self).build(input_shape)

    def _aggregate(self, node_features, adjacency_matrix):
        messages = K.batch_dot(K.permute_dimensions(adjacency_matrix, (0, 2, 1)), node_features)
        aggregated_messages = K.dot(messages, self.W_message)
        return aggregated_messages

    def _update(self, aggregated_messages, node_features):
        combined_features = K.concatenate([aggregated_messages, node_features], axis=-1)
        updated_features = self.update_network(combined_features)
        return updated_features

    def call(self, inputs, **kwargs):
        node_features, adjacency_matrix = inputs

        aggregated_messages = self._aggregate(node_features, adjacency_matrix)
        updated_node_features = self._update(aggregated_messages, node_features)

        return updated_node_features

    def get_config(self):
        config = super(MPNNLayer, self).get_config()
        config.update({
            "activation": activations.serialize(self.activation),
            "hidden_units": self.hidden_units
        })
        return config


class GraphLayer(keras.layers.Layer):

    def __init__(self,
                 step_num=1,
                 activation=None,
                 **kwargs):
        """Initialize the layer.

        :param step_num: Two nodes are considered as connected if they could be reached in `step_num` steps.
        :param activation: The activation function after convolution.
        :param kwargs: Other arguments for parent class.
        """
        self.supports_masking = True
        self.step_num = step_num
        self.activation = keras.activations.get(activation)
        self.supports_masking = True
        super(GraphLayer, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'step_num': self.step_num,
            'activation': self.activation,
        }
        base_config = super(GraphLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _get_walked_edges(self, edges, step_num):
        """Get the connection graph within `step_num` steps

        :param edges: The graph in single step.
        :param step_num: Number of steps.
        :return: The new graph that has the same shape with `edges`.
        """
        if step_num <= 1:
            return edges
        deeper = self._get_walked_edges(K.batch_dot(edges, edges), step_num // 2)
        if step_num % 2 == 1:
            deeper += edges
        return K.cast(K.greater(deeper, 0.0), K.floatx())

    def call(self, inputs, **kwargs):
        features, edges = inputs
        edges = K.cast(edges, K.floatx())
        if self.step_num > 1:
            edges = self._get_walked_edges(edges, self.step_num)
        outputs = self.activation(self._call(features, edges))
        return outputs

    def _call(self, features, edges):
        raise NotImplementedError('The class is not intended to be used directly.')


class GraphConv(GraphLayer):
    """Graph convolutional layer.

    h_i^{(t)} = \sigma \left ( \frac{ G_i^T (h_i^{(t - 1)} W + b)}{\sum G_i}  \right )
    """

    def __init__(self,
                 units,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 use_bias=True,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 **kwargs):
        """Initialize the layer.

        :param units: Number of new states. If the input shape is (batch_size, node_num, feature_len), then the output
                      shape is (batch_size, node_num, units).
        :param kernel_initializer: The initializer of the kernel weight matrix.
        :param kernel_regularizer: The regularizer of the kernel weight matrix.
        :param kernel_constraint:  The constraint of the kernel weight matrix.
        :param use_bias: Whether to use bias term.
        :param bias_initializer: The initializer of the bias vector.
        :param bias_regularizer: The regularizer of the bias vector.
        :param bias_constraint: The constraint of the bias vector.
        :param kwargs: Other arguments for parent class.
        """
        self.units = units
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.use_bias = use_bias
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.bias_constraint = keras.constraints.get(bias_constraint)

        self.W, self.b = None, None
        super(GraphConv, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'units': self.units,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': keras.constraints.serialize(self.kernel_constraint),
            'use_bias': self.use_bias,
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'bias_constraint': keras.constraints.serialize(self.bias_constraint),
        }
        base_config = super(GraphConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        feature_dim = input_shape[0][2]
        self.W = self.add_weight(
            shape=(feature_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='{}_W'.format(self.name),
        )
        if self.use_bias:
            self.b = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='{}_b'.format(self.name),
            )
        super(GraphConv, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + (self.units,)

    def compute_mask(self, inputs, mask=None):
        return mask[0]

    def _call(self, features, edges):
        features = K.dot(features, self.W)
        if self.use_bias:
            features += self.b
        if self.step_num > 1:
            edges = self._get_walked_edges(edges, self.step_num)
        return K.batch_dot(K.permute_dimensions(edges, (0, 2, 1)), features) \
           / (K.sum(edges, axis=2, keepdims=True) + K.epsilon())


class GraphPool(GraphLayer):

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, mask=None):
        return mask[0]


class GraphMaxPool(GraphPool):

    NEG_INF = -1e38

    def _call(self, features, edges):
        node_num = K.shape(features)[1]
        features = K.tile(K.expand_dims(features, axis=1), K.stack([1, node_num, 1, 1])) \
            + K.expand_dims((1.0 - edges) * self.NEG_INF, axis=-1)
        return K.max(features, axis=2)


class GraphAveragePool(GraphPool):

    def _call(self, features, edges):
        return K.batch_dot(K.permute_dimensions(edges, (0, 2, 1)), features) \
            / (K.sum(edges, axis=2, keepdims=True) + K.epsilon())
