#!/usr/bin/env python3.5

""" Defines the Op class which represents an operation.
    For example, Conv2d, Fc, Add. """

from mmrazor.models.algorithms.quantization.cle_superacme.common.connected_graph.product import Product


class OpInformation:
    """ Additional Op specific information.
    This is temporary. Once MaskPropagation feature is completed,
    the OpInformation will be refactored. """

    def __init__(self):
        self._groups = None
        self._model_module = None
        self._num_in_channels = None
        self._num_out_channels = None

    @property
    def groups(self):
        """ Returns the groups information. """
        return self._groups

    @groups.setter
    def groups(self, groups):
        self._groups = groups

    @property
    def num_in_channels(self):
        """ Returns the number of in channels """
        return self._num_in_channels

    @num_in_channels.setter
    def num_in_channels(self, num_in_channels):
        """ Set the number of in channels """
        self._num_in_channels = num_in_channels

    @property
    def num_out_channels(self):
        """ Returns the number of out channels """
        return self._num_out_channels

    @num_out_channels.setter
    def num_out_channels(self, num_out_channels):
        """ Set the number of out channels """
        self._num_out_channels = num_out_channels

    @property
    def model_module(self):
        """
        Returns the model module associated with this op.
        Essentially the actual model object that this op represents.
        """
        return self._model_module

    @model_module.setter
    def model_module(self, model_module):
        self._model_module = model_module


class Op:    # pylint: disable=too-many-public-methods
    """An instance of this class represents an operation, being either a named
    module (instance variable), an anonymous module (local variable), or
    a function from torch.nn.functional."""

    def __init__(self, name, dotted_name, output_shape, is_anonymous, op_type):
        self.name_op = name
        self.dotted_name_op = dotted_name
        self._output_shape = output_shape
        self._is_anonymous = is_anonymous
        self._type = op_type
        self._inputs = []
        self._output = None
        self._op_info = OpInformation()

    def __repr__(self):
        """ Returns name. """
        return self.name_op

    @property
    def name(self):
        """ Returns name. """
        return self.name_op

    @property
    def dotted_name(self):
        """ Returns dotted name. """
        return self.dotted_name_op

    @dotted_name.setter
    def dotted_name(self, dotted_name):
        """ Sets the dotted name. """
        self.dotted_name_op = dotted_name

    @property
    def output_shape(self):
        """ Returns the output shape. """
        return self._output_shape

    @output_shape.setter
    def output_shape(self, shape):
        """ Sets the output shape of an Operation. """
        self._output_shape = shape

    # TODO: only used by old connected graph, remove in the future
    @property
    def is_anonymous(self):
        """ If the Operation is an anonymous operation, returns True. """
        return self._is_anonymous

    @property
    def type(self):
        """ Returns the type of the operation. For example, Conv2d, etc., """
        return self._type

    @property
    def inputs(self):
        """ Returns the inputs of an Operation. """
        return self._inputs

    @inputs.setter
    def inputs(self, inputs):
        """ Set the inputs list """
        self._inputs = inputs

    def add_input(self, product: Product):
        """ Adds a product to the inputs of an Operation."""
        self._inputs.append(product)

    @property
    def input_ops(self):
        """ Returns all the inputs of an Operation. """
        return [inp.producer for inp in self._inputs if inp.producer]

    @property
    def output(self):
        """ Returns the output of an operation. """
        return self._output

    @output.setter
    def output(self, product: Product):
        """ Sets a product as the output of an Operation. """
        self._output = product

    @property
    def groups(self):
        """ Returns the groups parameter.
        The groups parameter applies only to Conv modules. """
        return self._op_info.groups

    @groups.setter
    def groups(self, groups):
        self._op_info.groups = groups

    @property
    def num_in_channels(self):
        """ Returns the number of in channels for this op """
        return self._op_info.num_in_channels

    @num_in_channels.setter
    def num_in_channels(self, num_in_channels):
        """ Returns the number of in channels for this op """
        self._op_info.num_in_channels = num_in_channels

    @property
    def num_out_channels(self):
        """ Returns the number of out channels for this op """
        return self._op_info.num_out_channels

    @num_out_channels.setter
    def num_out_channels(self, num_out_channels):
        """ Returns the number of in channels for this op """
        self._op_info.num_out_channels = num_out_channels

    @property
    def model_module(self):
        """ Returns the model op associated with this op. """
        return self._op_info.model_module

    @model_module.setter
    def model_module(self, model_module):
        self._op_info.model_module = model_module

    def get_module(self):
        """ Return the module associated with this Op. """
        if self.model_module is not None:
            return self.model_module.get_module()
        return None

    def get_input_products(self):
        """ Return a list of products that are inputs for this operation (not parameters) """

        input_products = []
        for product in self.inputs:
            if not product.is_parm and not product.is_const:
                input_products.append(product)

        return input_products


def determine_preceding_op_input_product_index_in_multi_input_op(preceding_op, multi_input_op):
    """ Originally, the preceding op's product was one of the inputs for the Concat op. Since a Split Op
    is getting inserted in the  middle between them, Split Op's product must be inserted exactly in the same
    position as the preceding op's product. For that purpose, determine teh preceding op's product's
    index position. """

    preceding_op_dotted_name = preceding_op.dotted_name

    for index in range(len(multi_input_op.inputs)):
        if multi_input_op.inputs[index].producer is not None and \
                multi_input_op.inputs[index].producer.dotted_name == preceding_op_dotted_name:
            print("Preceding Op: %s, product index: %s, multi input Op: %s",
                         preceding_op.dotted_name, index, multi_input_op.dotted_name)
            return index
    return None


def determine_succeeding_op_output_product_index_in_multi_output_op(succeeding_op, multi_output_op):
    """

    :param succeeding_op:
    :param multi_output_op:
    :return:
    """
    succeeding_op_dotted_name = succeeding_op.dotted_name

    for index in range(len(multi_output_op.output.consumers)):
        if multi_output_op.output.consumers[index].dotted_name == succeeding_op_dotted_name:
            print("Succeeding Op: %s, product index: %s, multi output Op: %s",
                         succeeding_op_dotted_name, index, multi_output_op.dotted_name)
            return index

    return None
