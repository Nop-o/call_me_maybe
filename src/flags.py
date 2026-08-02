from typing import Any
import argparse


def get_flags() -> Any:
    """Handle flags"""
    parser = argparse.ArgumentParser()

    parser.add_argument('--functions_definition',
                        default='data/input/functions_definition.json')
    parser.add_argument('--input',
                        default='data/input/function_calling_tests.json')
    parser.add_argument('--output',
                        default='data/output/function_calling_results.json')

    return parser.parse_args()
