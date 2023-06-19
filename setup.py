from pathlib import Path

from setuptools import find_packages, setup

# read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

GITHUB_URL = "https://github.com/miquelmassot/spherecluster"

classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Topic :: Software Development",
    "Topic :: Software Development :: Libraries",
    "Topic :: Software Development :: Libraries :: Python Modules",
]

setup(
    name="spherecluster2",
    version="0.2.0",
    description="Clustering on the unit hypersphere in scikit-learn.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Jason Laska",
    author_email="jason@claralabs.com",
    maintainer="Miquel Massot",
    maintainer_email="miquel.massot@gmail.com",
    packages=find_packages(),
    classifiers=classifiers,
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
    project_urls={"Bug Reports": GITHUB_URL + "/issues", "Source": GITHUB_URL},
    url=GITHUB_URL,
    license="MIT",
)
