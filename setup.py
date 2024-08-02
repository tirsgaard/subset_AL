from setuptools import setup

setup(
    name='subsetAL',
    version='0.1.0',    
    description='A library for implementing active learning for learning subsets under constraints',
    url='https://github.com/tirsgaard/subset_AL',
    author='Rasmus Tirsgaard',
    license='BSD 2-clause',
    packages=['src'],
    install_requires=[
        "torch",                  
                ],
)