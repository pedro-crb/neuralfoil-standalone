from setuptools import setup

setup(
    name='neuralfoil-standalone',
    version="0.2.3",
    description='neuralfoil-standalone is a fork of NeuralFoil without the explicit AeroSandbox dependency',
    long_description_content_type='text/markdown',
    packages=["neuralfoil_standalone"],
    python_requires='>=3.11',
    install_requires=[
        'numpy >= 1',
        'optisandbox >= 4.2.4'
    ],
    include_package_data=True,
    package_data={
        'NN parameters': ['*.npz'],
    },
)
