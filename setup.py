from setuptools import find_packages, setup

setup(
    name="cutile-forge",
    version="3.0.0",
    packages=find_packages(include=["cutile_forge", "cutile_forge.*"]),
    python_requires=">=3.11",
)

