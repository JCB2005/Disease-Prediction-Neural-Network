from typing import List

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
