<h1>
  <a href="https://www.flaticon.com/authors/good-ware">
  <img src=./docs/astronaut.svg width=100px align="left" title="Icon made by Good Ware from Flaticon">
  </a>
  Planetoids
</h1>

Procedurally generated worlds.

![Build Status](https://img.shields.io/travis/com/paulds8/planetoids)
![Coverage Status](https://img.shields.io/codecov/c/github/paulds8/planetoids)
![Codacy Status](https://img.shields.io/codacy/grade/77b39d19f4c54647820cc7b7d22e2f41)

<h2>What is it?</h2>
Planetoids is a high level Python API for generating interactive, procedurally generated worlds from data in a pandas DataFrame.

<h2>What does it do?</h2>
Currently, Planetoids is able to terraform a planet from two-dimensional data that has an optional cluster attribute. It's still very new and will be growing in capabilities, but for now the library can achieve the following when terraforming a new world:

+ generates somewhere in _space_ to render your creation
+ generates an ecology based on input data statistics
+ generates land masses
   + these land masses have 	
      + topographic detail (contours) 	
      + relief detail (gradients)
+ generates lighting effects in the form of a hillshade
  

Your terraformed world can be rendered using many different map projections for different effects.

<h2>This is just the beginning</h2>

I'm hoping to add in hydrological effects, atmospheric effects, bathometry, vegetation, civilizations, animations and a <b>whole lot more</b>. If you'd be interested in helping shape and grow this library to its full potential, take a look at the issues with a [good first issue](https://github.com/paulds8/planetoids/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) label or raise an issue with features you think the library could benefit from.

<h2>How to create a Planetoid</h2>
Creating a Planetoid is as simple as

```python
import planetoids as pt
planet = pt.Planetoid(df, x="x_column", y="y_column", cluster_field="cluster_column").fit_terraform()
```

For full working examples, check out the interactive online [demo notebooks](https://nbviewer.jupyter.org/github/paulds8/planetoids/blob/master/examples).

<h2>Installing</h2>
Planetoids depends on:

+ pandas
+ sklearn
+ shapely
+ pyproj
+ plotly
+ opencv
+ and their related dependencies like numpy and scipy
 
<h3>Install Options</h3>

PyPI:

```python
pip install planetoids
```

Manual:

```bash
wget https://github.com/paulds8/planetoids/archive/master.zip
unzip master.zip
rm master.zip
cd planetoids
pip install -r requirements.txt
```

If you're on Windows and installing from PyPI or manually, you may need to install the following packages from the Windows binaries for your Python version here: 

+ [shapely](https://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely)
+ [opencv](https://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv)

<h2>Documentation</h2>

This library is super new, so there's not a ton of documentation to come by _just yet_, but the public-facing API is fully documented [here](https://paulds8.github.io/planetoids/planetoids.m).
