from setuptools import setup
from os import path
from codecs import open

here = path.abspath(path.dirname(__file__))

# get version from version.py
__version__ = None
exec(open('prolif/version.py').read())

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='prolif',
    version=__version__,
    description='Protein-Ligand Interaction Fingerprints',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/chemosim-lab/ProLIF',
    author='Cédric Bouysset',
    author_email='bouysset.cedric@gmail.com',
    license='Apache License, Version 2.0',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Chemistry',
    ],
    keywords='science chemistry biology drug-design chemoinformatics virtual-screening',
    packages=['prolif'],
    entry_points = {
        'console_scripts': ['prolif=prolif.command_line:main'],
    },
    python_requires='>=3.6',
    install_requires=['numpy>=1.13.3','mdanalysis>=2.0.0'],
    extras_require={
        'sklearn': ['scikit-learn'],
    },
    test_suite="tests",
    tests_require="pytest",
    package_data={
        'tests': ['*.mol2'],
    },
    include_package_data=True,
    project_urls={
        'Bug Reports':  'https://github.com/chemosim-lab/ProLIF/issues',
    },
    zip_safe=False,
)
