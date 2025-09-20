from setuptools import setup, find_packages

setup(
    name="mf6pqc",
    version="0.1.0",
    packages=find_packages(where='src'),
    package_dir={"": "src"},
    include_package_data=True,
    python_requires=">=3.8",
)
