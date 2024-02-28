"""MoHa setup script."""

from setuptools import setup

from os import path


here = path.abspath(path.dirname(__file__))


long_description = open(path.join(here, "README.md"), encoding="utf-8").read()


setup(
    name="moha",
    version="0.0.0",
    description="A sample Python project",
    python_requires=">=3.6",
    install_requires=["numpy", "scipy"],
    packages=["moha", "moha.test"],
)
