from setuptools import find_packages, setup

setup(
    name="cutile-forge",
    version="0.4.0",
    packages=find_packages(include=["cutile_forge", "cutile_forge.*"]),
    python_requires=">=3.11",
)

