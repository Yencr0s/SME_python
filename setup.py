from setuptools import setup

setup(
    name='functionsP',
    version='0.1.0',
    author='Roberto Riaño Hidalgo',
    packages=['functionsP'],
    url='https://github.com/Yencr0s/SME_python',
    license='LICENSE.txt',
    install_requires=[
        "pandas >= 2.1.0",
        "matplotlib >= 3.8.0",
        "numpy >=1.26.0",
        "seaborn >= 0.13.0"
    ],
    description='Functions for the SME project',
)