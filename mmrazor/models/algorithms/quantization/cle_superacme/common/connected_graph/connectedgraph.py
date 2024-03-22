# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
""" Connected graph abstract class and utilities """

from abc import ABC, abstractmethod
from typing import List
from mmrazor.models.algorithms.quantization.cle_superacme.common.connected_graph.operation import Op


class ConnectedGraph(ABC):
    """ ConnectedGraph abstract class """

    def __init__(self):
        self._ops = dict()
        self._products = dict()

    @abstractmethod
    def get_op_from_module_name(self, name: str):
        """ Given the name of a operation/module, return the corresponding op in ops dict """

    def get_all_ops(self):
        """ Returns the ops dictionary """
        return self._ops

    def get_all_products(self):
        """ Returns the products dictionary """
        return self._products


def get_ordered_ops(list_of_starting_ops: List[Op]) -> List[Op]:
    """
    Function to get all the ops in connected graph based on occurrence by Depth First Traversal
    :param list_of_starting_ops: List of starting ops of the graph
    :return: List of connected graph ops in order of occurrence
    """

    def graph_traversal(current_op: Op, visited_ops: set, ordered_ops: List[Op]):
        """
        util function for Depth First Traversal
        :param current_op: tf.Operation
        :param visited_ops: Set of ops visited so far (to cut short duplicate traversals)
        :param ordered_ops: List of ops in order of occurrence
        """
        # Add current op to visited_ops set
        visited_ops.add(current_op)

        # iterate all the output tensors of current opchange_out_act_shape_to_channels_first
        if current_op.output:
            for consumer_op in current_op.output.consumers:
                # add consumer op to visited_ops list if not added previously and recursively call
                if consumer_op not in visited_ops:
                    graph_traversal(consumer_op, visited_ops, ordered_ops)

        # add to ordered_ops list only when all the children ops are traversed
        ordered_ops.append(current_op)

    #  Set of all ops that have been visited (to cut short duplicate traversals)
    visited_ops_set = set()

    # List of all ops in order of occurrence
    ordered_ops_list = []

    for op in list_of_starting_ops:
        graph_traversal(op, visited_ops_set, ordered_ops_list)

    # reverse the list because ops are in reverse order
    ordered_ops_list.reverse()

    return ordered_ops_list
