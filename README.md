
<h1>
  <a href="https://www.flaticon.com/authors/good-ware">
  <img src=./docs/astronaut.svg width=100px align="left" title="Icon made by Good Ware from Flaticon">
  </a>
  Planetoids
</h1>

_Procedurally generated worlds._

<h2>What is it?</h2>
Planetoids is a high level Python API for generating interactive, procedurally generated worlds from data in a pandas DataFrame.

<h2>Main Features</h2>
Planetoids is able to terraform a planet from two-dimensional data that has an optional cluster attribute. It's still very new and will be growing in capabilities, but for now the library can achieve the following when terraforming a new world:

+ generates somewhere in _space_ to render your creation
+ generates an ecology
+ generates land masses
	+ these land masses have 
		+ topographic detail (contours) 
		+ relief detail (gradients)
+ generates lighting effects in the form of a hillshade

Your terraformed world can be rendered using many different map projections for different effects.

<h2>How to create a Planetoid</h2>

<h2>Installing</h2>
Planetoids depends on:

 - pandas
 - sklearn
 - shapely
 - pyproj
 - plotly
 - open-cv
 - and their related dependencies like numpy and scipy

<h3>Install Options</h3>
Conda:

    ...

PyPI:

    ...

Manual:

    wget https://github.com/paulds8/planetoids/archive/master.zip
    unzip master.zip
    rm master.zip
    cd planetoids
    pip install -r requirements.txt

If you're on Windows, you will need to install the following dependencies from: ...

<h2>Documentation</h2>
This library is super new, so there's not a ton of documentation to come by just yet, you can view some basic documentation here:

<h2>Help & Support</h2>
Put the TODOs somewhere here as well
