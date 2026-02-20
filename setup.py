from setuptools import setup, find_packages

setup(
    name="oaktree",
    version="0.1.0",
    description="Optimized Analytic K-taxon Tree Reconstruction with EM Estimation",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "networkx>=3.0",
        "treeswift>=1.1",
    ],
    extras_require={
        "validation": ["msprime>=1.2", "dendropy>=4.6", "matplotlib>=3.7"],
        "dev": ["pytest>=7.4"],
    },
    entry_points={
        "console_scripts": [
            "oaktree=oaktree.cli:main",
        ],
    },
)
