# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
'''
Unit tests for argument_parsing.py.
'''

import argparse
import unittest

from utils.argument_parsing import store_bool


class TestStoreBool(unittest.TestCase):

    def test_values(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--argument_name', action=store_bool)

        for value_string in ['false', 'FAlse', 'NO', 'no']:
            result = parser.parse_args(['--argument_name', value_string])
            self.assertEqual(result.argument_name, False)

        for value_string in ['true', 'tRUE', 'yes', 'YES']:
            result = parser.parse_args(['--argument_name', value_string])
            self.assertEqual(result.argument_name, True)

    def test_failure(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--argument_name', action=store_bool)

        # These raise argparse.ArgumentError and then argparse raises a
        # SystemExit exception instead.
        with self.assertRaises(SystemExit):
            parser.parse_args(['--argument_name'])

        with self.assertRaises(SystemExit):
            parser.parse_args(['--argument_name', 'absolutely'])

    def test_default(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--argument_name',
                            action=store_bool,
                            default=False)

        result = parser.parse_args([])
        self.assertEqual(result.argument_name, False)


if __name__ == '__main__':
    unittest.main()
