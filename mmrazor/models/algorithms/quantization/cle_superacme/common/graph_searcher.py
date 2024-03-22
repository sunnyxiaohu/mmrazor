# /usr/bin/env python3.5
# -*- mode: python -*-
""" Main class for pattern match based graph searcher"""

import itertools
from collections import deque
from mmrazor.models.algorithms.quantization.cle_superacme.common.graph_pattern_matcher import PatternMatcher


class SlidingWindow:
    """
    Sliding window class implements a sliding window deque and provides api to update it's state
    """

    def __init__(self, window_size):
        """
        initializes params required for pattern matching
        :param window_size: max length of sliding window
        """
        # deque structure will be sized based on length provided
        self.current_op_window = deque(maxlen=window_size)

    def get_sub_graph_type_pattern_2(self) -> list:
        """
         sub_graph is stored as deque of ops
         this api converts it to list of op typ strings to aid with pattern matching
        :return: list of op types of the ops in the op_types_in_sliding_window deque.
        """

        return [op.type for op in self.current_op_window]

    def get_sub_graph_type_pattern(self) -> list:
        """
         sub_graph is stored as deque of ops
         this api converts it to list of op typ strings to aid with pattern matching
        :return: list of op types of the ops in the op_types_in_sliding_window deque.
        """

        return [op.type for op in self.current_op_window]

    def append_to_sliding_window(self, op) -> None:
        """
        appends op provided to op_types_in_sliding_window deque.
        :param op: Connected graph op type
        :return: None
        """

        assert op is not None, 'Error, op passed to append_to_sliding_window is None'
        self.current_op_window.append(op)

    def remove_op_from_sliding_window(self, op) -> None:
        """
        removes op provided from sliding_window deque.
        :param op: Connected graph op type
        :return: None
        """

        self.current_op_window.remove(op)

    def get_op_sliding_window(self) -> deque:
        """
        returns op_types_in_sliding_window
        :return: current op sliding window
        """
        return self.current_op_window


class GraphSearcher:
    """
    Graph searcher class performs graph search on connected graph.
    It uses SlidingWindow to maintain the search window and PatternMatcher to match sub graph patterns.
    """

    def __init__(self, conn_graph, patterns_with_callback):
        """
        initializes params required for pattern matching
        :param patterns_with_callback: patterns with corresponding call back functions
        """
        self._connected_graph = conn_graph
        self._patterns_with_callbacks = patterns_with_callback
        self.sliding_window = None

    def _find_patterns_apply_actions(self, op,
                                     pattern_matcher: PatternMatcher,
                                     visited_nodes) -> None:
        """
        Finds all patterns in the graph using DFS with sliding window based pattern matcher
        :param op: starting op as connected graph op
        :param pattern_matcher: pattern matcher instance
        :param visited_nodes: list of ops visited to avoid loops during search
        :return: None
        """

        if op and op in visited_nodes:
            return

        # sliding window stores the op and the type
        self.sliding_window.append_to_sliding_window(op)
        op_types_sliding_window = self.sliding_window.get_sub_graph_type_pattern()

        # we get the index in the sliding window and the matched pattern back from pattern matcher
        matched_patterns_start_indices_dict = pattern_matcher.get_matching_patterns(op_types_sliding_window)

        if matched_patterns_start_indices_dict:
            for matched_pattern in matched_patterns_start_indices_dict.keys():
                for i in matched_patterns_start_indices_dict[matched_pattern]:
                    # we need to call appropriate handler here based on the matched length and the starting op type
                    op_subset = list(itertools.islice(self.sliding_window.get_op_sliding_window(), i,
                                                      i+len(matched_pattern.pattern)))
                    print('...... subset to store %s', op_subset)
                    matched_pattern.action(matched_pattern, op_subset)

        # mark visited node
        visited_nodes.add(op)

        # recursively call if op has children
        # move the op_sliding_window if output ops found ; continue DFS
        if op.output:
            for consumer in op.output.consumers:
                GraphSearcher._find_patterns_apply_actions(self, consumer, pattern_matcher,
                                                           visited_nodes)
        # Done with the op, if this op in sliding window, remove it
        if op in self.sliding_window.current_op_window:
            self.sliding_window.remove_op_from_sliding_window(op)

    def find_all_patterns_in_graph_apply_actions(self):
        """
         Finds all conv and linear layers with bn and activations if present
         :return: None
        """

        # Find the input node(s) in the graph
        input_nodes = []
        for op in self._connected_graph.get_all_ops().values():
            if op.inputs and op.inputs[0].is_model_input:
                input_nodes.append(op)

        visited_nodes = set()

        # define pattern matcher for graph search and set the sliding window length
        pattern_matcher = PatternMatcher(self._patterns_with_callbacks)
        self.sliding_window = SlidingWindow(pattern_matcher.get_pattern_max_length())

        # find layers of interest
        for op in input_nodes:

            # perform DFS with sliding window
            GraphSearcher._find_patterns_apply_actions(self, op, pattern_matcher,
                                                       visited_nodes)
