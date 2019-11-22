import os
import sys

this_dir = os.path.abspath(os.path.dirname(__file__))


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


root_path = os.path.join(this_dir, '../')
add_path(root_path)
