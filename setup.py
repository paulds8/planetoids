from distutils.core import setup
setup(
  name = 'planetoids',         # How you named your package folder (MyLib)
  packages = ['planetoids'],   # Chose the same as "name"
  version = '0.1-alpha',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Planetoids is a high level Python API for generating interactive, procedurally generated worlds from data in a pandas DataFrame.',   # Give a short description about your library
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
    'Topic :: Software Development :: Artistic Software :: Scientific/Engineering',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7'
  ],
)