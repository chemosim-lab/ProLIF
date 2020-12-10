from setuptools import setup
import os

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with open(os.path.join(here, 'README.rst'), "r") as f:
    long_description = f.read()

# generate the metadata file
__version__ = os.environ.get('PROLIF_VERSION', '0.3.0')
__license__ = 'Apache License, Version 2.0'
__author__ = 'Cédric Bouysset'
metadata = f"""__version__ = {__version__!r}
__author__ = {__author__!r}
__license__ = {__license__!r}"""
with open(os.path.join(here, "prolif", "__about__.py"), "w") as f:
    f.write(metadata)

setup(
    name='prolif',
    version=__version__,
    description='Protein-Ligand Interaction Fingerprints',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    url='https://github.com/chemosim-lab/ProLIF',
    author=__author__,
    author_email='bouysset.cedric@gmail.com',
    license=__license__,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Chemistry',
    ],
    keywords='science chemistry biology drug-design chemoinformatics molecular-dynamics',
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
    },
    package_data={
        'prolif': ['data/*.mol2', 'data/*.pdb', 'data/*.xtc'],
    },
    include_package_data=True,
    project_urls={
        'Issues':  'https://github.com/chemosim-lab/ProLIF/issues',
        'Discussions': 'https://github.com/chemosim-lab/ProLIF/discussions',
        'Documentation': 'https://prolif.readthedocs.io/en/latest/',
    },
    zip_safe=False,
)
