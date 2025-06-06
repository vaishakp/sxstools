from setuptools import Extension, setup, Command, find_packages

VERSION = '0.1'

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='sxstools',
    version=VERSION,
    description = 'A sxstool package for generating SXS waveforms',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Kanchan Soni",
    author_email='ksoni01@syr.edu',
    url='https://github.com/Kanchan-05/sxstools',
    keywords=['gravitational waves', 'pycbc', 'sxs', 'nr data'],
    packages=find_packages(),
    python_requires='>=3.10',

    classifiers=[
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    ],

    install_requires=[
        "pycbc",
        "sxs>=2022.5.6",
        "numpy",
        "scipy",
        "h5py",  
        "lalsuite", 
    ]
)
