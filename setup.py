from setuptools import setup, find_packages

# Read the dependencies from requirements.txt file
with open('requirements.txt', 'r') as file:
    install_requires = file.readlines()

setup(
    name='skacq',
    version='0.0.1',
    author='Surya',
    author_email='suryamailwork@gmail.com',
    description='Custom HMM model fitting and optimizing for time-series skill acquisition datasets',
    packages=find_packages(),
    install_requires=install_requires,
)
