from setuptools import setup

setup(
    name='NeuralFoil',
    version="0.2.3",
    description='NeuralFoil is an airfoil aerodynamics analysis tool using physics-informed machine learning, '
                'in pure Python/NumPy.',
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
