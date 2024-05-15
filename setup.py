from setuptools import setup, find_packages

setup(name='BUSCLEAN',
      version=0.1,
      author='Kailee Hung',
      author_email='abunnell@hawaii.edu',
      description='Cleaning and knowledge extraction pipeline from clinical BUS scans.',
      packages=find_packages(),
      license='cc-by-nc-sa 4.0',
      include_package_data=True,
      install_requires=[]) # kailee todo