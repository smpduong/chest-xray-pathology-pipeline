from setuptools import setup, find_packages

setup(
    name="chest-xray-pathology-pipeline",
    version="1.0.0",
    author="Steven Duong",
    author_email="",
    description="Production-oriented chest X-ray pathology detection pipeline",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/smpduong/chest-xray-pathology-pipeline",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "torchxrayvision>=1.2.0",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "requests>=2.31.0",
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0", "ruff>=0.1.0", "mypy>=1.5.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
