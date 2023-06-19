from setuptools import find_packages, setup

setup(
    name="spherecluster",
    version="0.2.0",
    description="Clustering on the unit hypersphere in scikit-learn.",
    author="Jason Laska",
    author_email="jason@claralabs.com",
    maintainer="Miquel Massot",
    maintainer_email="miquel.massot@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.23.4",
        "scipy>=1.8.0",
        "scikit-learn>=1.2.2",
        "pytest>=7.1.2",
        "nose>=1.3.7",
        "joblib>=1.2.0",
        "mlinsights>=0.3.649",
        "threadpoolctl>=3.1.0",
        "cython>=0.29.28",
        "pandas>=1.4.2",
        "matplotlib>=3.5.1",
        "pandas-streaming>=0.3.239",
        "attrs>=23.1.0",
        "iniconfig>=1.1.1",
        "packaging>=23.1",
        "pluggy<2.0,>=0.12 ",
        "ijson>=3.2.1",
        "six>=1.5",
    ],
    url="https://github.com/miquelmassot/spherecluster",
    license="MIT",
)
