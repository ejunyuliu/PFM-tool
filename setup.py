from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='PFM',
      version='0.1',
      description='Polarized microscope simulation tool.',
      long_description=readme(),
      classifiers=[
          'Development Status :: 2 - Pre-Alpha',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Natural Language :: English',
          'Programming Language :: Python :: 3.9',
          'Topic :: Scientific/Engineering :: Physics',
      ],
      url='https://github.com/ejunyuliu',
      author='Junyu Liu, Talon Chanler, Min Guo',
      author_email='ejunyuliu@gmail.com',
      license='MIT',
      packages=['PFM'],
      zip_safe=False,
      test_suite='tests',
      )