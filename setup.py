from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='minimol',
    version='1.0',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'minimol.ckpts.minimol_v1': [
            'state_dict.pth',
            'config.yaml',
            'base_shape.yaml'
        ]
    },
    install_requires=[
        "graphium==2.4.7",
        "hydra-core"
    ],
    url='https://github.com/graphcore-research/minimol',
    author='Blazej Banaszewski, Kerstin Klaser',
    author_email='blazej@banaszewski.pl, kerstink@graphcore.ai',
    description='Molecular fingerprinting using pre-trained deep nets',
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires='>=3.9'
)
