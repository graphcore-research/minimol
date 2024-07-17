from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess
import sys
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


class CustomInstallCommand(install):
    def run(self):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch>=2.0"])
        install.run(self)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "graphium @ git+https://github.com/datamol-io/graphium.git@04fe66fed8dd30daa5d2820bb5e97c54700e32f7#egg=graphium"])

setup(
    name='minimol',
    version='0.3.1',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'minimol.ckpts.minimol_v1': [
            'state_dict.pth',
            'config.yaml',
            'base_shape.yaml'
        ],
        '': ['graphium_patch.diff'],
    },
    install_requires=[
        "typer",
        "loguru",
        "omegaconf >=2.0.0",
        "tqdm",
        "platformdirs",
        # scientific
        "numpy",
        "scipy >=1.4",
        "pandas >=1.0",
        "scikit-learn",
        "fastparquet",
        # ml
        "hydra-core",
        "lightning >=2.0",
        "torchmetrics >=0.7.0,<0.11",
        "ogb",
        "torch-geometric >=2.0",
        "wandb",
        "mup",
        "torch_sparse >=0.6",
        "torch_cluster >=1.5",
        "torch_scatter >=2.0",
        # viz
        "matplotlib >=3.0.1",
        "seaborn",
        # cloud IO
        "fsspec >=2021.6",
        "s3fs >=2021.6",
        "gcsfs >=2021.6",
        # chemistry
        "datamol >=0.10",
    ],
    url='https://github.com/graphcore-research/minimol',
    author='Blazej Banaszewski, Kerstin Klaser',
    author_email='blazej@banaszewski.pl, kerstink@graphcore.ai',
    description='Molecular fingerprinting using pre-trained deep nets',
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires='>=3.9',
    cmdclass={
        'install': CustomInstallCommand
    },
)
