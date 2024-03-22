# /usr/bin/env python3.5
# -*- mode: python -*-
""" Main class for pattern matcher"""

import sys


class PatternType:
    """
    structure to hold pattern data type
    """
    def __init__(self, pattern, action):
        """
        PatternType class holds a pattern with a corresponding actions
        :param pattern: pattern to be searched
        :param action: action to be applied upon finding pattern
        """
        self.pattern = pattern
        self.action = action


class PatternMatcher:
    """
    A pattern matcher class that performs custom sub graph matches
    """

    def __init__(self, patterns_and_callbacks):
        """
        initializes params required for pattern matching
        :param patterns_and_callbacks:
        """

        # list of PatternType elements
        # the order of elements serves as the match priority
        self.patterns = patterns_and_callbacks
        self.pattern_match_length = self.get_pattern_max_length()

    def get_pattern_max_length(self):
        """
        computes the length of longest pattern
        :return: max length among the pattern list provided
        """

        max_len = 0
        for item in self.patterns:
            if len(item.pattern) > max_len:
                max_len = len(item.pattern)

        return max_len

    def _get_pattern_min_length(self):
        """
        computes the length of shortest pattern
        :return: min length among the pattern list provided
        """

        min_len = sys.maxsize
        for item in self.patterns:
            if len(item.pattern) < min_len:
                min_len = len(item.pattern)

        if min_len == sys.maxsize:
            min_len = 1

        return min_len

    def _get_matched_sliced_pattern(self, s_pattern):
        """
        compare sliced pattern against all reference patterns
        :param s_pattern: sliced pattern to be matched against reference pattern set
        :return: bool, pattern_with_callback
        """
        pattern_with_callback = None

        for pattern_with_callback in self.patterns:
            if s_pattern == pattern_with_callback.pattern:
                return True, pattern_with_callback

        return False, pattern_with_callback

    def _get_all_sliced_patterns_and_match(self, pattern):
        """
        helper function that slices the pattern passed and matches it against reference set of patterns
        :param pattern: pattern to be matched
        :return: dictionary of matched pattern/ sliced patterns
        """
        index_patterns = []
        match_start_indices_patterns = {}

        slice_len = len(pattern)

        # Example to describe the match algorithm implemented below.
        # if max pattern length is 4
        # and we receive a pattern [OP_X, BN, CONV, OP_X] to be matched
        # we slice it into sub patterns below:
        # [OP_X, BN, CONV, OP_X],
        # [OP_X, BN, CONV,], [BN, CONV, OP_X],
        # [OP_X BN], [BN, CONV],
        # [OP_X],[BN], [Conv]
        # We accumulate all matches and the start offsets.
        # Suppose the "reference patterns" to be matched against were :
        # [OP_X, BN, CONV, OP_X], [OP_X] [BN, CONV]
        # This method returns matched patterns and corresponding list of
        # starting offset in the pattern provided.
        # Example output :
        # [OP_X, BN, CONV, OP_X] pattern with offset 0,
        # [OP_X] pattern with offset[0, 3] and
        # [BN, CONV] with offset [1]
        # Return type would be a dictionary with
        # Keys of type 'PatternType', values are a list of start offset indices.

        minimum_slice_length = self._get_pattern_min_length()
        while slice_len >= minimum_slice_length:
            for i in range(0, len(pattern)):
                if i+slice_len <= len(pattern):
                    sliced_pattern = pattern[slice(i, i+slice_len)]
                    print('.. sliced pattern %s ', sliced_pattern)
                    print('... sliced pattern start_index %s ', i)
                    found_match, matched_pattern = self._get_matched_sliced_pattern(sliced_pattern)
                    if found_match:
                        if matched_pattern not in match_start_indices_patterns.keys():
                            index_set = {i}
                            match_start_indices_patterns[matched_pattern] = index_set
                            index_patterns.append(matched_pattern.pattern)
                        else:
                            indices = match_start_indices_patterns[matched_pattern]
                            indices.add(i)
                            match_start_indices_patterns[matched_pattern] = indices
            slice_len = slice_len-1

        return match_start_indices_patterns

    def get_matching_patterns(self, sliding_window_op_type_pattern):
        """
        matches the pattern and its sliced forms with reference patterns
        :param sliding_window_op_type_pattern: ops to be searched as op types sliding window (list)
        :return: a dictionary of all matched patterns/sliced patterns and
        their start offsets in the pattern.
        """

        # we slice the given sub pattern and then compare against reference pattern set
        matched_patterns_start_indices = self._get_all_sliced_patterns_and_match(sliding_window_op_type_pattern)

        if matched_patterns_start_indices:
            return matched_patterns_start_indices

        return matched_patterns_start_indices

    @staticmethod
    def apply_custom_action(*args, **create_time_kwds):
        """
        curry funciton to perform custom actions on matched patterns
        :param args: input arguments
        :param create_time_kwds:
        :return:
        """
        action = args[0]
        create_time_args = args[1:]

        def apply_pattern_action(*call_time_args, **call_time_kwds):
            args = create_time_args + call_time_args
            kwds = create_time_kwds.copy()
            kwds.update(call_time_kwds)
            return action(*args, **kwds)
        return apply_pattern_action
