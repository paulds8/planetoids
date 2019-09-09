import numpy as np
import pandas as pd
import umap
import pyproj #pip install pyproj==2.2.1 --no-cache-dir
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.stats as st
import cv2 as cv
from tqdm import tqdm
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from shapely.geometry import asPoint
from shapely.geometry import asLineString
from shapely.geometry import asPolygon
from shapely.geometry import MultiPoint
from shapely.ops import transform
from functools import partial
from scipy.spatial import Delaunay
from functools import reduce
from plotly.subplots import make_subplots

class Planetoid(object):
    def __init__(self,
                 data,
                 component1_field,
                 component2_field,
                 cluster_field,
                 ecology = 'gist_earth'):
        
        self.data = None
        self.component1_field = None
        self.component2_field = None
        self.cluster_field = None
        self.ecology = None
        
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
        try:
            cm.get_cmap(ecology, 1)
            self.ecology = ecology
        except Exception as e:
            raise ValueError(e)

        # only keep what we need
        self.data = self.data[[component1_field, component2_field, cluster_field]]
        
        #set the rest
        self.contours = None
        self.ocean_colour = None
        self.fig = None
        self.cmap = None
        self.max_contour = None
        self.shadows = list()
        self.light_side = list()


    def rescale_coordinates(self):
        """Rescale provided components as pseudo latitudes and longitudes"""
        #trying to prevent issues at the extremes
        lat_scaler = MinMaxScaler(feature_range=(-80, 80))
        long_scaler = MinMaxScaler(feature_range=(-170, 170))

        self.data['Latitude'] = lat_scaler.fit_transform(self.data[self.component1_field].values.reshape(-1, 1)).reshape(-1)
        self.data['Longitude'] = long_scaler.fit_transform(self.data[self.component2_field].values.reshape(-1, 1)).reshape(-1)

        # self.data.plot(kind='scatter',
        #                 x='Longitude',
        #                 y='Latitude',
        #                 c=self.cluster_field,
        #                 cmap='Spectral')
        # plt.show()
        
    
    def get_contour_verts(self, cn):
        """Get the vertices from the mpl plot to generate our own geometries"""
        cntr = []
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
            cntr.append(paths)

        return cntr


    def get_contours(self, subset, topography_levels, shadow_levels):
        """Generate contour lines based on density of points per cluster/class"""
                
        topography_levels += 5
        
        y=subset['Latitude'].values
        x=subset['Longitude'].values

        # Define the borders
        deltaX = (max(x) - min(x))/3
        deltaY = (max(y) - min(y))/3
        xmin = max(-180, min(x) - deltaX)
        xmax = min(180, max(x) + deltaX)
        ymin = max(-90, min(y) - deltaY)
        ymax = min(90, max(y) + deltaY)
        #print(xmin, xmax, ymin, ymax)
        # Create meshgrid
        xx, yy = np.mgrid[xmin:xmax:(20*topography_levels + 1j), ymin:ymax:(20*topography_levels + 1j)]

        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([x, y])
        kernel = st.gaussian_kde(values, bw_method='silverman')
        f = np.reshape(kernel(positions).T, xx.shape)
        
        hillshade = np.rot90(self.calculate_hillshade(f, 315, 45))

        fig = plt.figure(figsize=(8,8))
        ax = fig.gca()
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        #cfset = ax.contourf(xx, yy, f, cmap='coolwarm')
        #ax.imshow(np.rot90(f), cmap='coolwarm', extent=[-180, 180, -90, 90])
        cset = ax.contour(xx, yy, f, colors='k', levels=topography_levels, )
        
        cntrs = self.get_contour_verts(cset)
        plt.close(fig)
        
        self.generate_hillshade_polygons(hillshade, xx, yy, xmin, xmax, ymin, ymax, shadow_levels)
        self.generate_lightside_polygons(hillshade, xx, yy, xmin, xmax, ymin, ymax, shadow_levels)
        
        return cntrs
    
    
    def generate_hillshade_polygons(self, hillshade, xx, yy, xmin, xmax, ymin, ymax, shadow_levels):
        
        #self.shadows = list()
        
        #we have to strech it for the opencv function to catch the edges properly
        hs_array = (hillshade - hillshade.min())/(hillshade.max()-hillshade.min()) * 255
        hist, bin_edges = np.histogram(hs_array, bins=shadow_levels+5) #add some since we toss many of these away currently
        #bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
        
        #still need to refine this, but this piece here should help catch only the shadows and not the 'light side'
        bin_edges = [x for x in bin_edges if x > 170]

        cluster_shadows = []
        for b in list(zip(bin_edges[:-1], bin_edges[1:])):
            hs_array_binary_slice = hs_array.copy()
            hs_array_binary_slice[(hs_array_binary_slice<b[0]) & (hs_array_binary_slice != 1)] = 0
            hs_array_binary_slice[(hs_array_binary_slice>=b[0]) & (hs_array_binary_slice<b[1])] = 1
            #hs_array_binary_slice[(hs_array_binary_slice>=b[1]) & (hs_array_binary_slice != 1)] = 0
            
            hs_array_binary_slice = np.flipud(hs_array_binary_slice)
            hs_array_binary_slice = hs_array_binary_slice.astype(np.uint8)  

            #plt.imshow(hs_array_binary_slice,cmap='Greys', extent=[xmin, xmax, ymin, ymax])
            #plt.show()
            
            contours, hierarchy = cv.findContours(hs_array_binary_slice.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            for cntr in contours:
                x_loc = [xx[pair[0][0], pair[0][1]] for pair in cntr]
                y_loc = [yy[pair[0][0], pair[0][1]] for pair in cntr]
                
                #get rid of polygons that touch the bondary of the calculated extent
                if xmin not in x_loc\
                    and xmax not in x_loc\
                    and ymin not in y_loc\
                    and ymax not in y_loc:
                        coords = list(zip(x_loc + [x_loc[0]], y_loc + [y_loc[0]]))
                        if len(coords) > 3:
                            #attempt some smoothing
                            coords = list(asPolygon(coords).buffer(1, join_style=1).buffer(-1, join_style=1).exterior.coords)
                            cluster_shadows.append(coords)       
        self.shadows.append(cluster_shadows)
        
        
    def generate_lightside_polygons(self, hillshade, xx, yy, xmin, xmax, ymin, ymax, shadow_levels):
        
        #self.shadows = list()
        
        #we have to strech it for the opencv function to catch the edges properly
        hs_array = (hillshade - hillshade.min())/(hillshade.max()-hillshade.min()) * 255
        hist, bin_edges = np.histogram(hs_array, bins=shadow_levels+5) #add some since we toss many of these away currently
        #bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
        
        #still need to refine this, but this piece here should help catch only the 'light side'
        bin_edges = [x for x in bin_edges if x > 10 and x <= 84]

        light_side = []
        for b in list(zip(bin_edges[:-1], bin_edges[1:])):
            hs_array_binary_slice = hs_array.copy()
            hs_array_binary_slice[(hs_array_binary_slice<b[0]) & (hs_array_binary_slice != 1)] = 0
            hs_array_binary_slice[(hs_array_binary_slice>=b[0]) & (hs_array_binary_slice<b[1])] = 1
            #hs_array_binary_slice[(hs_array_binary_slice>=b[1]) & (hs_array_binary_slice != 1)] = 0
            
            hs_array_binary_slice = np.flipud(hs_array_binary_slice)
            hs_array_binary_slice = hs_array_binary_slice.astype(np.uint8)  

            #plt.imshow(hs_array_binary_slice,cmap='Greys', extent=[xmin, xmax, ymin, ymax])
            #plt.show()
            
            contours, hierarchy = cv.findContours(hs_array_binary_slice.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            for cntr in contours:
                x_loc = [xx[pair[0][0], pair[0][1]] for pair in cntr]
                y_loc = [yy[pair[0][0], pair[0][1]] for pair in cntr]
                
                #get rid of polygons that touch the bondary of the calculated extent
                if xmin not in x_loc\
                    and xmax not in x_loc\
                    and ymin not in y_loc\
                    and ymax not in y_loc:
                        coords = list(zip(x_loc + [x_loc[0]], y_loc + [y_loc[0]]))
                        if len(coords) > 3:
                            #attempt some smoothing
                            coords = list(asPolygon(coords).buffer(1, join_style=1).buffer(-1, join_style=1).exterior.coords)
                            light_side.append(coords)
        self.light_side.append(light_side)
        
        
        # #plot
        # fig = plt.figure(figsize=(8,8))
        # ax = fig.gca()
        # ax.set_xlim(xmin, xmax)
        # ax.set_ylim(ymin, ymax)
        # plt.imshow(hs_array,cmap='Greys', extent=[xmin, xmax, ymin, ymax])
        # plt.show()


    def get_all_contours(self, topography_levels=20, shadow_levels=20):
        """Get all of the contours per class"""
        cntrs = {}
        for cluster in tqdm(np.unique(self.data['Cluster'].values)):
            points_df = self.data.loc[self.data['Cluster'] == cluster, ['Longitude', 'Latitude']]
            cntrs[cluster] = self.get_contours(points_df, topography_levels, shadow_levels)
        
        self.contours = cntrs
    
    
    def calculate_hillshade(self, array, azimuth, angle_altitude): 
        """Calculate a hillshade over the generated topography"""
        x, y = np.gradient(array)
        slope = np.pi/2. - np.arctan(np.sqrt(x*x + y*y))
        aspect = np.arctan2(-x, y)
        azimuthrad = azimuth*np.pi / 180.
        altituderad = angle_altitude*np.pi / 180.
        shaded = np.sin(altituderad) * np.sin(slope)\
        + np.cos(altituderad) * np.cos(slope)\
        * np.cos(azimuthrad - aspect)
        return 255*(shaded + 1)/2
    
        
    def fit(self, topography_levels=20, shadow_levels=20):
        """Generate data required for terraforming"""
        #transform 2d components into pseudo lat/longs
        self.rescale_coordinates()
        #generate contours per class
        self.get_all_contours(topography_levels, shadow_levels)
        
    
    def plot_surface(self):
        """This plots the surface layer which we need because we can't set it directly"""
        #globe
        self.fig.add_trace(
            go.Scattergeo(
                lon = [179.9,0,0,179.9],
                lat = [89.9,89.9,0,0],
                mode='lines',
                line=dict(width=0, color=self.ocean_colour),
                fill='toself',
                fillcolor = self.ocean_colour,
                hoverinfo='skip',
                opacity=1,
                showlegend=False
            ),
            row=1,
            col=1
            )

        #flat
        self.fig.add_trace(
            go.Scattergeo(
                lon = [179.9,0,0,179.9],
                lat = [89.9,89.9,0,0],
                mode='lines',
                line=dict(width=0, color=self.ocean_colour),
                fill='toself',
                fillcolor = self.ocean_colour,
                hoverinfo='skip',
                opacity=0.95,
                showlegend=False
           ),
           row=2,
           col=1
           )
        
  
    def plot_shadows(self):
        """Plot the hillshade-derived shadows"""
        #globe
        for cluster in tqdm(self.shadows):
            for ix, shadow in enumerate(cluster):
                if ix % 2:
                    shadow_array = np.array(shadow)
                    self.fig.add_trace(
                        go.Scattergeo(
                            lon = list(shadow_array[:, 0]),
                            lat = list(shadow_array[:, 1]),
                            hoverinfo='skip',
                            mode='lines',
                            line=dict(width=0,
                                    color='black'
                                    ),
                            fill='toself',
                            fillcolor = 'black',
                            opacity=0.1,
                            showlegend=False,
                            ),row=1,col=1)
                    
        #flat
        for cluster in tqdm(self.shadows):
            for ix, shadow in enumerate(cluster):
                if ix % 3:
                    shadow_array = np.array(shadow)
                    self.fig.add_trace(
                        go.Scattergeo(
                            lon = list(shadow_array[:, 0]),
                            lat = list(shadow_array[:, 1]),
                            hoverinfo='skip',
                            mode='lines',
                            line=dict(width=0,
                                    color='black'
                                    ),
                            fill='toself',
                            fillcolor = 'black',
                            opacity=0.1,
                            showlegend=False,
                            ),row=2,col=1)
                    
                    
    def plot_light_side(self):
        """Plot the hillshade-derived shadows"""
        #globe
        for cluster in tqdm(self.light_side):
            for ix, shadow in enumerate(cluster):
                if ix % 2:
                    shadow_array = np.array(shadow)
                    self.fig.add_trace(
                        go.Scattergeo(
                            lon = list(shadow_array[:, 0]),
                            lat = list(shadow_array[:, 1]),
                            hoverinfo='skip',
                            mode='lines',
                            line=dict(width=0,
                                    color='red'
                                    ),
                            fill='toself',
                            fillcolor = 'white',
                            opacity=0.1,
                            showlegend=False,
                            ),row=1,col=1)
                    
        #flat
        for cluster in tqdm(self.light_side):
            for ix, shadow in enumerate(cluster):
                if ix % 3:
                    shadow_array = np.array(shadow)
                    self.fig.add_trace(
                        go.Scattergeo(
                            lon = list(shadow_array[:, 0]),
                            lat = list(shadow_array[:, 1]),
                            hoverinfo='skip',
                            mode='lines',
                            line=dict(width=0,
                                    color='red'
                                    ),
                            fill='toself',
                            fillcolor = 'white',
                            opacity=0.1,
                            showlegend=False,
                            ),row=2,col=1)
  
    
    def plot_contours(self):
        """Plot the topography"""
        #globe
        for cluster, contour in tqdm(self.contours.items()):
            for ix, line in enumerate(contour):
                #need to update this to actually check for contours that form polygons
                if len(line) > 0 and ix > (self.max_contour-3)/len(contour) + 2:
                    if ix % 2 == 0:
                        for l in line:
                            self.fig.add_trace(
                                go.Scattergeo(
                                    lon = list(l[:, 0]),
                                    lat = list(l[:, 1]),
                                    hoverinfo='skip',
                                    mode='lines',
                                    line=dict(width=0,
                                            color='rgb' + str(self.cmap(ix/self.max_contour, bytes=True)[0:3])
                                            ),
                                    fill='toself',
                                    fillcolor = 'rgb' + str(self.cmap(ix/self.max_contour, bytes=True)[0:3]),
                                    opacity=0.95 - (ix * 0.01),
                                    showlegend=False,
                                    ),row=1,col=1)
                        # else:                    
                        #     self.fig.add_trace(
                        #         go.Scattergeo(
                        #             lon = list(l[:, 0]),
                        #             lat = list(l[:, 1]),
                        #             hoverinfo='skip',
                        #             mode='lines',
                        #             line=dict(width=0, #*np.power(np.exp(ix/max_contour),2),
                        #                     #dash='longdashdot',
                        #                     color='rgb' + str(self.cmap(ix/self.max_contour, bytes=True)[0:3])),
                        #             opacity=1,
                        #             showlegend=False
                        #             ),row=1,col=1)
                            
        #flat
        for cluster, contour in self.contours.items():
            for ix, line in enumerate(contour):
                #need to update this to actually check for contours that form polygons
                if len(line) > 0 and ix > (self.max_contour-3)/len(contour) + 2:
                    for l in line:
                        if ix % 3 == 0:
                            self.fig.add_trace(
                                go.Scattergeo(
                                    lon = list(l[:, 0]),
                                    lat = list(l[:, 1]),
                                    hoverinfo='skip',
                                    mode='lines',
                                    line=dict(width=0.1,
                                              color='rgb' + str(tuple(list(self.cmap(ix/len(contour), bytes=True)[0:3]) + [0.5]))
                                              ),
                                    fill='toself',
                                    showlegend=False,
                                    opacity=0.95 - (ix * 0.01)
                                    ),row=2,col=1)
                            
                            
    def plot_clustered_points(self):
        """Plot the provided point data"""
        self.fig.add_trace(go.Scattergeo(
            lon = self.data['Longitude'],
            lat = self.data['Latitude'],
            marker_color=self.data['Cluster'],
            hoverinfo='text',
            hovertext=self.data['Cluster'],
            marker_size=3,
            showlegend= False
        #     marker = dict(
        #         symbol='circle-open',
        #      )
            ),row=1,col=1)

        #plot the points - flat
        self.fig.add_trace(go.Scattergeo(
            lon = self.data['Longitude'],
            lat = self.data['Latitude'],
            marker_color=self.data['Cluster'],
            hoverinfo='text',
            hovertext=self.data['Cluster'],
            marker_size=1,
            showlegend=False,
        #     marker = dict(
        #         symbol='circle-open',
        #      )
            ),row=2,col=1)
    
    
    def update_geos(self):
        """Update config for maps"""
        # globe
        self.fig.update_geos(
            row=1,col=1,
            showland = False,
            showcountries = False,
            showocean = False,
            showcoastlines=False,
            showframe=False,
            showrivers=False,
            showlakes=False,
            showsubunits=False,
            #junk='',
            bgcolor = "black", #changes the whole graph background, but we only want the surface
            projection = dict(
                type = 'orthographic',
                rotation = dict(
                    lon = 0,
                    lat = 0,
                    roll = 0
                )
            ),
            lonaxis = dict(
                showgrid = False,
                gridcolor = 'rgb(102, 102, 102)',
                gridwidth = 1
            ),
            lataxis = dict(
                showgrid = False,
                gridcolor = 'rgb(102, 102, 102)',
                gridwidth = 1
            ))
        #flat
        self.fig.update_geos(
            row=2,col=1,
            showland = False,
            showcountries = False,
            showocean = False,
            showcoastlines=False,
            showframe=False,
            showrivers=False,
            showlakes=False,
            showsubunits=False,
            #junk='',
            bgcolor = "black", #changes the whole graph background, but we only want the surface
            projection = dict(
                type = 'natural earth',
                rotation = dict(
                    lon = 0,
                    lat = 0,
                    roll = 0
                )
            ),
            lonaxis = dict(
                showgrid = True,
                gridcolor = 'rgb(102, 102, 102)',
                gridwidth = 1
            ),
            lataxis = dict(
                showgrid = True,
                gridcolor = 'rgb(102, 102, 102)',
                gridwidth = 1
            ))
        
    
    def update_layout(self):
        """Update layout config"""
        self.fig.update_layout(
            autosize=True,
            width=int(1920/2),
            height=int(1281/2),
            title_text = 'Planetoids',
            showlegend = True,
            plot_bgcolor = "black",
            paper_bgcolor = "black",
            margin=dict(l=20, r=20, t=40, b=20)
        )
    
    
    def terraform(self, topography=True, points=True, shadows=True, render=True):
        """Construct a new world"""
        
        self.fig = make_subplots(
            rows=2, cols=2, 
            vertical_spacing=0.05,
            column_widths=[0.5, 0.5],
            row_heights=[0.66, 0.33],
            specs=[[{"type": "scattergeo", "colspan": 2}, None],
                   [{"type": "scattergeo", "colspan": 2}, None]]
            )
        
        #identify the maximum number of contours per continent
        self.max_contour = max([len(contour) for contour in self.contours.values()])
        self.cmap = cm.get_cmap(self.ecology, self.max_contour)
        
        self.ocean_colour = 'rgb' + str(self.cmap(3/self.max_contour, bytes=True)[0:3])
        
        self.plot_surface()

        if topography:
            self.plot_contours()
            
        if shadows:
            self.plot_shadows()
            self.plot_light_side()
            
        if points:
            self.plot_clustered_points()
                            
        self.update_geos()

        self.update_layout()

        if render:
            self.fig.show()
            
    
    def fit_terraform(self, topography_levels=20, shadow_levels=20, topography=True, points=True, shadows=True, render=True):
        """Fit and terraform in a single step, akin to fit_transform people are used to"""
        self.fit(topography_levels=topography_levels, shadow_levels=shadow_levels)
        self.terraform(topography, points, shadows, render)