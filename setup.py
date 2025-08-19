#!/usr/bin/env python3
"""
Setup script for ECHO package.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="echo-model",
    version="0.1.0",
    author="Yucong Zhang",
    author_email="yucong0428@outlook.com",
    description="ECHO: Enhanced Contextual Hierarchical Output for Audio Representation Learning",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yucongzh/ECHO",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="audio, machine-learning, deep-learning, representation-learning, mae, transformer",
    project_urls={
        "Bug Reports": "https://github.com/yucongzh/ECHO/issues",
        "Source": "https://github.com/yucongzh/ECHO",
        "Documentation": "https://github.com/yucongzh/ECHO#readme",
    },
)
