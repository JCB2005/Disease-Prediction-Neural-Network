import random
import math


class Neuron:
    """
    Base structure of a neuron in the neural network
    """

    def __init__(self):
        """
        Initialises the neuron's base values
        """
        self.weight: float = random.randint(-1, 1)
        self.bias: float = random.randint(-1, 1)

    def sigmoid(self, inp: float):
        """
        Uses the sigmoid function to get the output value from an input
        :param inp: float
        :return: None
        """
        out = abs((math.exp(inp) / (math.exp(inp) + 1)) * self.weight + self.bias)
        return out

    def set_weight(self, new_weight: float):
        """
        Allows the neuron's weight to be changed
        :param new_weight: float
        :return: None
        """
        self.weight = new_weight

    def get_weight(self) -> float:
        """
        Returns the weight of the neuron
        :return: float
        """
        return self.weight

    def set_bias(self, new_bias: float):
        """
        Allows the neuron's bias to be changed
        :param new_bias: float
        :return: None
        """
        self.bias = new_bias

    def get_bias(self) -> float:
        """
        Returns the bias of the neuron
        :return: float
        """
        return self.bias
