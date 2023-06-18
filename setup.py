from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        # remove the slash n that gets by default appended when writing characters in new line in requirements.txt
        requirements = [req.replace("\n" , "") for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
        return requirements

setup(
    name = 'DiamondPricePrediction',
    version='0.0.1',
    author='Shafin',
    author_email='shafinmohammed315@gmail.com',
    # below code is a hardcoded code, instead use get_requirements function that will read the requirements.txt file and install the written files in that requirements.txt
    # install_requires = ['pandas' , 'numpy'],

    install_requires = get_requirements('requirements.txt'),
    packages=find_packages()

)