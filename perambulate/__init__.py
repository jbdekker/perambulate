# flake8: noqa

__version__ = "0.1.4"

hard_dependencies = ("pandas", "scipy", "matplotlib")
missing_dependencies = []

for dependency in hard_dependencies:
    try:
        __import__(dependency)
    except ImportError as e:
        missing_dependencies.append(f"{dependency}: {e}")

if missing_dependencies:
    raise ImportError(
        "Unable to import required dependencies:\n"
        + "\n".join(missing_dependencies)
    )
del hard_dependencies, dependency, missing_dependencies


from perambulate.core.api import Condition, Periodic, ValueSearch

import perambulate.datasets as datasets
import perambulate.filters as filters

__doc__ = """
perambulate - a conditional timeseries analysis library for Python
==================================================================

**perambulate** is a Python package providing condition based slicing,
filtering and transformation capabilities on time-series data

Main Features
-------------
Here are a few to the things Perambulate does:

    - ValueSearch: Easy identification of periods within a time-series
      dataset based on time-based entry and exit criteria.
    - Condition: Slicing of time-series dataset based on value based criteria
    - Periodic: Slicing of time-series dataset based on date-time intervals

"""
