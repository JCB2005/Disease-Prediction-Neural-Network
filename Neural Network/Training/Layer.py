from typing import List
import Neuron

import sys


class Layer:
    def __init__(self, size: int = 10):
        """
        Initialises the layer
        :param size: size of the layer (int)
        """
        self.layer: List[type(Neuron.Neuron)] = [Neuron.Neuron for _ in range(size)]

    def get_layer_size(self) -> int:
        """
        Returns the size of the layer
        :return: The size of the layer (int)
        """
        return len(self.layer)

    def __iadd__(self, other: Neuron.Neuron):
        """
        Adds another neuron to the layer
        :param other: The neuron to add
        :return: None
        """
        self.layer.append(other)

    def edit_layer(self, neuron_i: int, new_weight: float = sys.float_info.min, new_bias: float = sys.float_info.min):
        self.layer[neuron_i].set_weight(
            new_weight if new_weight != sys.float_info.min else self.layer[neuron_i].get_weight())

        self.layer[neuron_i].set_bias(
            new_bias if new_bias != sys.float_info.min else self.layer[neuron_i].get_bias())
