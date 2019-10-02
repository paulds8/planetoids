from distutils.core import setup

setup(
  name = 'planetoids',
  packages = ['planetoids'],
  version = '0.1-alpha.2',
  license='MIT',
  description = "Planetoids is a high level Python API for generating interactive, procedurally generated worlds from data in a pandas DataFrame.",
  long_description="For further information please visit https://github.com/paulds8/planetoids",
  long_description_content_type='text/markdown',
  author = 'Paul dos Santos',
  author_email = 'paulds8@hotmail.com',
  url = 'https://github.com/paulds8/planetoids',
  download_url = 'https://github.com/paulds8/planetoids/archive/0.1-alpha.2.tar.gz',
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