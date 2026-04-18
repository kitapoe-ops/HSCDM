from setuptools import setup, find_packages

setup(
    name="hscdm",
    version="1.0.0",
    description="Hormuz Strait Crisis Detection Model — Z-Score volatility anomaly scoring",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="kitapoe-ops",
    author_email="kitapoe@gmail.com",
    url="https://github.com/kitapoe-ops/hscdm",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21",
        "pandas>=2.0",
        "requests>=2.28",
        "feedparser>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "matplotlib>=3.5",
            "newspaper3k>=0.2",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
