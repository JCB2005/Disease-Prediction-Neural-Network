import collections
from typing import List, Dict

import Layer
import Neuron


class NeuralNetwork:
    def __init__(self, inputs: type(Layer.Layer), hidden_layers: List[type(Layer.Layer)], outputs: type(Layer.Layer)):
        """
        Initialises the neural networks values
        :param inputs: The inputs the network has (Layer)
        :param hidden_layers: The hidden layers the network has (List[Layer])
        :param outputs: The output neurons the network has (Layer)
        """
        # Initialise the layers of the network
        self.input_layer = inputs
        self.hidden_layers = hidden_layers
        self.output_layer = outputs
        self.network_error = 0

    def __iadd__(self, other: Layer.Layer):
        self.hidden_layers.append(Layer.Layer())

    def neuron_loop(self) -> collections.Iterable:
        layers = self.input_layer + self.hidden_layers + self.output_layer
        for layer in layers:
            for neuron in layer:
                yield neuron

    def edit_input_layer_neuron(self, i: int, new_val: type(Neuron.Neuron)):
        """
        Used to replace a neuron in the input layer
        :param i: The index of the neuron to change (Layer)
        :param new_val: The new neuron (Neuron)
        :return: None
        """
        self.input_layer[i] = new_val

    def edit_hidden_layer_neuron(self, layer_i: int, i: int, new_val: type(Neuron.Neuron)):
        """
        Used to replace a neuron in a hidden layer
        :param layer_i: The index of the hidden layer to edit (int)
        :param i: The index of the neuron to change (int)
        :param new_val: The new neuron (Neuron)
        :return: None
        """
        self.hidden_layers[layer_i][i] = new_val

    def edit_output_layer_neuron(self, i: int, new_val: type(Neuron.Neuron)):
        """
        Used to replace a neuron in the input layer
        :param i: The index of the neuron to change (Layer)
        :param new_val: The new neuron (Neuron)
        :return: None
        """
        self.output_layer[i] = new_val

    def calculate_square_error(self, results: Dict[float: float]) -> float:
        """
        Calculates the accuracy of a network.
        :param results: A dictionary containing the output values of the training as the keys and the actual values of the training as the values. All values should be between 0 and 1. (Dict[float: float])
        :return: The error of the network as a float between 0 and 1. (float)
        """
        sum_err = sum(self.__calc_sqr_err_gen(results))
        self.network_error = sum_err / len(results)
        return self.network_error

    @staticmethod
    def __calc_sqr_err_gen(results: Dict[float: float]) -> iter:
        """
        A generator function used to calculate sum of the errors of each output neuron in a network.
        :param results: A dictionary containing the output values of the training as the keys and the actual values of the training as the values. All values should be between 0 and 1. (Dict[float: float])
        :return: An iterable object. (object)
        """
        for i, j in results.items():
            yield (j - i) ** 2
