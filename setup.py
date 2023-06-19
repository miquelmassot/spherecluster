from __future__ import print_function

import sys

from setuptools import find_packages, setup

with open("requirements.txt") as f:
    INSTALL_REQUIRES = [line.strip() for line in f.readlines() if line]


try:
    import numpy  # NOQA
except ImportError:
    print("numpy is required during installation")
    sys.exit(1)

try:
    import scipy  # NOQA
except ImportError:
    print("scipy is required during installation")
    sys.exit(1)

setup(
    name="spherecluster",
    version="0.2.0",
    description="Clustering on the unit hypersphere in scikit-learn.",
    author="Jason Laska",
    author_email="jason@claralabs.com",
    maintainer="Miquel Massot",
    maintainer_email="miquel.massot@gmail.com",
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    url="https://github.com/miquelmassot/spherecluster",
    license="MIT",
)
