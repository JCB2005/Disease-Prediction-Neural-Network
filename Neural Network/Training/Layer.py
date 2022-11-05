from typing import List
import Neuron

import sys


class Layer:
    """
    Base structure of a layer in a neural network
    """

    def __init__(self, size: int = 10):
        """
        Initialises the layer
        :param size: size of the layer (int)
        """
        self.layer: List[type(Neuron.Neuron)] = [Neuron.Neuron for _ in range(size)]
        self.max = size

    def __iter__(self):
        self.n = 0
        return self.n

    def __next__(self):
        if self.n <= self.max:
            result = self.layer[self.n]
            self.n += 1
            return result

        else:
            raise StopIteration

    def __iadd__(self, other: Neuron.Neuron):
        """
        Adds another neuron to the layer
        :param other: The neuron to add
        :return: None
        """
        self.layer.append(other)
        self.max += 1

    def __len__(self) -> int:
        """
        Gets the amount of neurons in the layer.
        :return: The amount of neurons in the layer (int)
        """
        return len(self.layer)

    def get_layer_size(self) -> int:
        """
        Returns the size of the layer
        :return: The size of the layer (int)
        """
        return len(self.layer)

    def edit_layer(self, neuron_i: int, new_weight: float = sys.float_info.min, new_bias: float = sys.float_info.min):
        """
        Used to change the weight and bias of a neuron in a layer to be changed.
        :param neuron_i: The index of the neuron to edit (int)
        :param new_weight: The new weight of the neuron (The weight will not be changed if the "new_weight" parameter is not changed) (float)
        :param new_bias: The new bias of the neuron (The bias will not be changed if the "new_bias" parameter is not changed) (float)
        :return: None
        """
        self.layer[neuron_i].set_weight(
            new_weight if new_weight != sys.float_info.min else self.layer[neuron_i].get_weight())

        self.layer[neuron_i].set_bias(
            new_bias if new_bias != sys.float_info.min else self.layer[neuron_i].get_bias())
