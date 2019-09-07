import numpy as np
import pandas as pd
import umap
import pyproj #pip install pyproj==2.2.1 --no-cache-dir
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from shapely.geometry import asPoint
from shapely.geometry import asLineString
from shapely.geometry import asPolygon
from shapely.geometry import MultiPoint
from shapely.ops import transform
from functools import partial
import scipy.stats as st
from scipy.spatial import Delaunay
from functools import reduce

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

        # only keep what we need
        self.data = self.data[[component1_field, component2_field, cluster_field]]
        
        #set the rest
        self.contours = None


    def rescale_coordinates(self):
        #trying to prevent issues at the extremes
        lat_scaler = MinMaxScaler(feature_range=(-80, 80))
        long_scaler = MinMaxScaler(feature_range=(-170, 170))

        self.data['Latitude'] = lat_scaler.fit_transform(self.data[self.com].reshape(-1, 1)).reshape(-1)
        self.data['Longitude'] = long_scaler.fit_transform(self.data[:,1].reshape(-1, 1)).reshape(-1)

        # self.data.plot(kind='scatter',
        #                 x='Longitude',
        #                 y='Latitude',
        #                 c=self.cluster_field,
        #                 cmap='Spectral')
        # plt.show()
        
    
    def get_contour_verts(self, cn):
        """Get the vertices from the mpl plot to generate our own geometries"""
        contours = []
        # for each contour line
        for cc in cn.collections:
            paths = []
            # for each separate section of the contour line
            for pp in cc.get_paths():
                xy = []
                # for each segment of that section
                for vv in pp.iter_segments():
                    xy.append(vv[0])
                paths.append(np.vstack(xy))
            contours.append(paths)

        return contours


    def get_contours(self):
        """Generate contour lines based on density of points per cluster/class"""
        y=self.data['Latitude'].values
        x=self.data['Longitude'].values

        # Define the borders
        deltaX = (max(x) - min(x))/10
        deltaY = (max(y) - min(y))/10
        xmin = min(x) - deltaX
        xmax = max(x) + deltaX
        ymin = min(y) - deltaY
        ymax = max(y) + deltaY
        #print(xmin, xmax, ymin, ymax)
        # Create meshgrid
        xx, yy = np.mgrid[xmin:xmax:300j, ymin:ymax:300j]

        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([x, y])
        kernel = st.gaussian_kde(values)
        f = np.reshape(kernel(positions).T, xx.shape)

        fig = plt.figure(figsize=(8,8))
        ax = fig.gca()
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        #cfset = ax.contourf(xx, yy, f, cmap='coolwarm')
        ax.imshow(np.rot90(f), cmap='coolwarm', extent=[-180, 180, -90, 90])
        cset = ax.contour(xx, yy, f, colors='k', levels=20,)
        
        contours = get_contour_verts(cset)
        plt.close(fig)
        return contours


    def get_all_contours(self):
        contours = {}
        for cluster in np.unique(self['Cluster'].values):
            # Filter out the cluster points
            points_df = self.data.loc[self.data['Cluster'] == cluster, ['Longitude', 'Latitude']]
            contours[cluster] = get_contours(points_df)
        
        self.contours = contours