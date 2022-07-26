from setuptools import setup, find_packages


with open('readme.md') as f:
    readme = f.read()

setup(
    name='gmr',
    version='0.1.0',
    description='Gaussian mixture reduction algorithms',
    long_description=readme,
    author='Jeong Hyunho',
    author_email='me@kennethreitz.com',
    url='https://github.com/JeongHyunho/GaussianMixtureReduction',
    packages=find_packages(exclude=('tests', 'images'))
)
