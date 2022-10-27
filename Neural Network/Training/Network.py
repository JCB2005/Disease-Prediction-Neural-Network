import collections
import random
from typing import List, Dict

import Layer
import Neuron


class NeuralNetwork:
    """
    Base structure of a neural network.
    """

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
        self.layers = [self.input_layer] + self.hidden_layers + [self.output_layer]
        self.output_values = List[float]
        self.results: Dict[float: float] = dict()
        self.network_error = float()

    def __iadd__(self, other: Layer.Layer):
        """
        Used to add a hidden layer to the neural network (with the '+=' operator)
        :param other: The hidden layer to add to the network (Layer.Layer)
        :return: None
        """
        self.hidden_layers.append(Layer.Layer())

    def __len__(self) -> int:
        """
        Gets the amount of neurons in the network
        :return: The amount of neurons in the network (int)
        """
        return sum(len(x) for x in self.layers)

    def test_network(self, _inputs: List[float], _outputs: List[float]) -> Dict[float, float]:
        """
        Calculates the results of the network's testing.
        :param _inputs: The input data - Must be the same size as the input layer - (List[float])
        :param _outputs: The outputs values of the dataset (List[float])
        :return: The results of the network's testing Dict[testing_values: actual_values] (Dict[float: float])
        """
        network_output_values: Dict[int: List[float]] = dict()
        for i in range(len(self.input_layer)):
            network_output_values[i] = []

        for input_neuron, _inp in zip(self.input_layer, _inputs):
            inp_lyr_out = input_neuron.sigmoid(_inp)
            for hidden_lyr in self.hidden_layers:
                for hidden_lyr_neuron in hidden_lyr:
                    hidden_lyr_out = hidden_lyr_neuron.sigmoid(inp_lyr_out)
                    for i, out_neuron in enumerate(self.output_layer):
                        network_output_values[i].append(out_neuron.sigmoid(hidden_lyr_out))

        self.output_values = [sum(x) / len(x) for x in network_output_values.values()]

        for i, j in zip(self.output_values, _outputs):
            self.results[i] = j

        return self.results

    def neuron_loop(self) -> collections.Iterable:
        """
        Used to loop through all neurons in a network
        :return: An iterable that loops through each neuron in the network
        """
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

    def add_neuron_to_random_hidden_layer(self):
        self.hidden_layers[random.randint(0, len(self.hidden_layers) - 1)] += Neuron.Neuron()

    def calculate_square_error(self, results: Dict[float, float]) -> float:
        """
        Calculates the accuracy of a network.
        :param results: A dictionary containing the output values of the training as the keys and the actual values of the training as the values. All values should be between 0 and 1. (Dict[float: float])
        :return: The error of the network as a float between 0 and 1. (float)
        """
        sum_err = sum(self.__calc_sqr_err_gen(results))
        self.network_error = sum_err / len(results)
        return self.network_error

    @staticmethod
    def __calc_sqr_err_gen(results: Dict[float, float]) -> iter:
        """
        A generator function used to calculate sum of the errors of each output neuron in a network.
        :param results: A dictionary containing the output values of the training as the keys and the actual values of the training as the values. All values should be between 0 and 1. (Dict[float: float])
        :return: An iterable object. (object)
        """
        for i, j in results.items():
            yield (j - i) ** 2
