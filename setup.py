from setuptools import setup, find_packages

setup(
    name='minimol',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'hydra-core',
        'graphium @ git+https://github.com/datamol-io/graphium.git@04fe66fed8dd30daa5d2820bb5e97c54700e32f7#egg=graphium'
    ],
    dependency_links=[
        'git+https://github.com/datamol-io/graphium.git@04fe66fed8dd30daa5d2820bb5e97c54700e32f7#egg=graphium'
    ],
    entry_points={
        'console_scripts': [
            'minimol=minimol:main'
        ],
    },
)
