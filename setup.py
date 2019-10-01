from distutils.core import setup

long_description = """
## What is it?
Planetoids is a high level Python API for generating interactive, procedurally generated worlds from data in a pandas DataFrame.

## What does it do?
Currently, Planetoids is able to terraform a planet from two-dimensional data that has an optional cluster attribute. It's still very new and will be growing in capabilities, but for now the library can achieve the following when terraforming a new world:

+ generates somewhere in _space_ to render your creation
+ generates an ecology based on input data statistics
+ generates land masses
	+ these land masses have 
		+ topographic detail (contours) 
		+ relief detail (gradients)
+ generates lighting effects in the form of a hillshade

For further information please visit the Homepage in the Project links on the left.
"""

setup(
  name = 'planetoids',
  packages = ['planetoids'],
  version = '0.1-alpha',
  license='MIT',
  description = 'Planetoids is a high level Python API for generating interactive, procedurally generated worlds from data in a pandas DataFrame.',   # Give a short description about your library
  long_description=long_description,
  author = 'Paul dos Santos',
  author_email = 'paulds8@hotmail.com',
  url = 'https://github.com/paulds8/planetoids',
  download_url = 'https://github.com/paulds8/planetoids/archive/0.1-alpha.tar.gz',
  keywords = ['interactive', '3D', 'plotting', 'world-building', 'data science', 'GIS'],
  install_requires=[ 
            'matplotlib',
            'numpy',
            'opencv-python',
            'pandas',
            'plotly',
            'Pillow',
            'scikit-learn',
            'scipy',
            'Shapely',
            'tqdm'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers', 
    'Topic :: Software Development',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7'
  ],
)