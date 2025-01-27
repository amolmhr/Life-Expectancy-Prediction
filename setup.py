import os
from setuptools import setup, find_packages

from setuptools import setup, find_packages

setup(
    name="life-expectancy-prediction",  # Project name
    version="0.1.0",  # Initial version
    description="A Python package to predict life expectancy using machine learning",  # Short project description
    author="Amol Mehra",
    author_email="amolmhr@gmail.com",
    url="https://github.com/amolmhr/life-expectancy-prediction",  # Replace with your repo URL
    license="MIT",
    packages=find_packages(where="src"),  # Automatically find packages in the 'src' directory
    package_dir={"": "src"},  
    include_package_data=True,  # Include non-code files listed in MANIFEST.in
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "seaborn>=0.11.0",
        "matplotlib>=3.4.0",
        "scikit-learn>=1.0.0"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.8",  # Minimum Python version required
    entry_points={
        "console_scripts": [
            "life-expectancy-prediction=src.main:main",  # Replace with your main script entry point
        ],
    },
)
