from setuptools import setup
from os import path
from codecs import open

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='prolif',
    version=os.environ.get('PROLIF_VERSION', '0.0.0'),,
    description='Protein-Ligand Interaction Fingerprints',
    long_description=long_description,
    long_description_content_type='text/x-rst',
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
    python_requires='>=3.6',
    install_requires=[
        'pandas>=1.0.0',
        'numpy>=1.13.3',
        'scipy>=1.3.0',
        'mdanalysis @ git+https://github.com/cbouy/mdanalysis.git@prolif#subdirectory=package',
        'tqdm'],
    extras_require={
        'tests': ['pytest>=6.1.2', 'pytest-cov', 'codecov'],
        'docs': ['sphinx>=3.2.1', 'recommonmark', 'sphinx-rtd-theme'],
    },
    test_suite="tests",
    package_data={
        'tests': ['*.mol2', '*.pdb', '*.xtc'],
    },
    include_package_data=True,
    project_urls={
        'Issues':  'https://github.com/chemosim-lab/ProLIF/issues',
        'Discussions': 'https://github.com/chemosim-lab/ProLIF/discussions',
        'Documentation': 'https://prolif.readthedocs.io/en/latest/',
    },
    zip_safe=False,
)
