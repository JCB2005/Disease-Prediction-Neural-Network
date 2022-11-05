import random
import sys
from typing import List

import Network


class Generation:
    """
    Base structure of a generation of neural networks.
    """

    def __init__(self, networks: List[Network.NeuralNetwork]):
        self.networks: List[Network.NeuralNetwork] = networks
        self.best_network: Network.NeuralNetwork
        self.best_err = sys.float_info.max

    def __find_best_network(self, _networks: List[Network.NeuralNetwork]) -> Network.NeuralNetwork:
        """
        Used to find the best network in a generation.
        :param _networks: the networks in the generation (List[Network.NeuralNetwork])
        :return: The best network in that generation (Network.NeuralNetwork)
        """
        self.best_err_index = 0
        for i, network in enumerate(_networks):
            if (new_best_err := network.calculate_square_error(network.results)) < self.best_err:
                self.best_index = i
                self.best_err = new_best_err
                self.best_network = network

        return self.best_network

    def calculate_new_generation(self, neuron_err_threshold: float, weight_alter_amt: float, bias_alter_amt: float) -> \
            List[Network.NeuralNetwork]:
        """
        Used to create a new generation of 100 networks.
        :param bias_alter_amt: The amount to alter the bias by in either direction (float)
        :param weight_alter_amt: The amount to alter the weight by in either direction (float)
        :param neuron_err_threshold: The error threshold required to add an extra neuron to a layer (float)
        :return: A new Generation in the form of a list of networks (List[Network.NeuralNetwork])
        """
        base_network = self.__find_best_network(self.networks)
        new_networks: List[Network.NeuralNetwork] = list()

        for i in range(100):
            network = self.__edit_network(base_network, neuron_err_threshold, weight_alter_amt, bias_alter_amt)
            new_networks.append(network)

        return new_networks

    def __edit_network(self, network, neuron_err_threshold: float, weight_alter_amt: float,
                       bias_alter_amt: float) -> Network.NeuralNetwork:
        """
        Used to edit a network.
        :param network: The network to edit
        :param neuron_err_threshold: The error threshold required to add an extra neuron to a layer (float)
        :param weight_alter_amt: The amount to alter the weight by in either direction (float)
        :param bias_alter_amt: The amount to alter the bias by in either direction (float)
        :return: The new network (Network.NeuralNetwork)
        """
        for neuron in network.neuron_loop():
            if random.randint(1, len(network)) == 1:
                neuron.set_weight(neuron.get_weight() + random.uniform(-weight_alter_amt, weight_alter_amt))

            if random.randint(1, len(network)) == 1:
                neuron.set_bias(neuron.get_bias() + random.uniform(-bias_alter_amt, bias_alter_amt))

        if network.network_error < neuron_err_threshold:
            network.add_neuron_to_random_hidden_layer()

        return network

    def print_stats(self):
        print(f"Best Error: {self.best_err}")
