from typing import List
import Neuron


class Layer:
    def __init__(self):
        self.layer: List[Neuron.Neuron] = list()

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