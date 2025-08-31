#!/usr/bin/env python3
"""
Setup script for ECHO package.
"""

from setuptools import setup, find_packages
import os
import sys

# Check Python version
if sys.version_info < (3, 8):
    raise RuntimeError("ECHO requires Python 3.8 or higher")

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
    version="0.2.0",
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
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
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
            "isort>=5.0",
            "pre-commit>=2.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.15",
        ],
        "audio": [
            "librosa>=0.9.0",
            "soundfile>=0.10.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="audio, machine-learning, deep-learning, representation-learning, mae, transformer, frequency-aware, hierarchical-encoding",
    project_urls={
        "Bug Reports": "https://github.com/yucongzh/ECHO/issues",
        "Source": "https://github.com/yucongzh/ECHO",
        "Documentation": "https://github.com/yucongzh/ECHO#readme",
        "Paper": "https://arxiv.org/abs/2508.14689",
        "Hugging Face": "https://huggingface.co/yucongzh/echo-small-0824",
    },
    entry_points={
        "console_scripts": [
            "echo-extract=ECHO.cli:main",
        ],
    },
)
