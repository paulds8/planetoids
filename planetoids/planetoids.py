import numpy as np
import pandas as pd
import umap
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from shapely.geometry import asPoint
from shapely.geometry import asLineString
from shapely.geometry import asPolygon
from shapely.geometry import MultiPoint
from shapely.ops import transform
from functools import partial
import pyproj #pip install pyproj==2.2.1 --no-cache-dir

class Planetoid(object):
    def __init__(self, data, component1_field, component2_field, cluster_field):
        if isinstance(data, pd.DataFrame):
            self.data = data
        else:
            raise ValueError('Please provide a pandas DataFrame')
        if component1_field in self.data.columns:
            self.component1_field = component1_field
        else:
            raise ValueError('Component 1 field not in provided DataFrame')
        if component2_field in self.data.columns:
            self.component2_field = component2_field
        else:
            raise ValueError('Component 2 field not in provided DataFrame')
        if cluster_field in self.data.columns:
            self.cluster_field = cluster_field
        else:
            raise ValueError('Cluster field not in provided DataFrame')

        self.data = self.data[[component1_field, component2_field, cluster_field]]
        print(self.data)