from setuptools import setup, find_packages

setup(name='BUSCLEAN',
      version=0.1,
      author='Kailee Hung',
      author_email='abunnell@hawaii.edu',
      description='Cleaning and knowledge extraction pipeline from clinical BUS scans.',
      packages=find_packages(),
      license='cc-by-nc-sa 4.0',
      include_package_data=True,
      install_requires=[
          'easyocr >= 1.7.0', 'numpy >= 1.24.3', 'opencv-python >= 4.7.0.72','pandas >= 1.5.3',
          'pillow >= 9.4.0', 'scikit-image >= 0.21.0', 'scipy >= 1.10.1', 'typing-extensions >= 4.6.3', 'matplotlib >= 3.7.1'
      ])