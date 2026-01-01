from setuptools import setup, find_packages

setup(
    name="intel-gpu-fine-tune",
    version="0.1.0",
    description="",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[],
    python_requires=">=3.12",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3 :: Only",
        "Operating System :: OS Independent",
    ],
)