from setuptools import setup, find_packages

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='nifty_maria',
    version='0.0.1',
    description='Library for Nifty fits of maria data.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/jwuerzinger/CMB_denoising',
    author='Jonas Wuerzinger',
    author_email='jonas.wuerzinger@tum.de',
    packages=find_packages(),
    # package_data={'': ['data/*.csv']},
    install_requires=['numpy',
                      'jax[cuda12]'
                      ],

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    # entry_points={
    #     'console_scripts': [
    #         'niceplot = niceplot.__main__:niceplot',
    #     ],
    # },
)    
