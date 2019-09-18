import numpy as np
import pandas as pd
import umap
import pyproj  # pip install pyproj==2.2.1 --no-cache-dir
import plotly.graph_objects as go
from plotly import offline
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.stats as st
import cv2 as cv
from tqdm.autonotebook import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from PIL import Image
from shapely.geometry import asPoint
from shapely.geometry import asLineString
from shapely.geometry import asPolygon
from shapely.ops import unary_union
import random

# from shapely.geometry import MultiPoint
# from shapely.ops import transform
# from functools import partial
from functools import reduce
from plotly.subplots import make_subplots


class Planetoid(object):
    def __init__(
        self,
        data,
        y,
        x,
        cluster_field=None,
        ecology="gist_earth",
        random_state=None
    ):

        self.data = None
        self.y = None
        self.x = None
        self.cluster_field = None
        self.ecology = None
        self.random_state=None
        
        self.data_generated = False

        if isinstance(data, pd.DataFrame):
            self.data = data
        else:
            raise ValueError("Please provide a pandas DataFrame")
        if y in self.data.columns:
            self.y = y
        else:
            raise ValueError("X field not in provided DataFrame")
        if x in self.data.columns:
            self.x = x
        else:
            raise ValueError("Y field not in provided DataFrame")
        if cluster_field is not None or cluster_field in self.data.columns:
            self.cluster_field = cluster_field
        else:
            raise ValueError("Cluster field not in provided DataFrame")
        try:
            cm.get_cmap(ecology, 1)
            self.ecology = ecology
        except Exception as e:
            raise ValueError(e)
        if isinstance(random_state, int):
            self.random_state = random_state
            np.random.seed(self.random_state)
            random.seed(self.random_state)
            cv.setRNGSeed(self.random_state)
        elif random_state is None:
            pass
        else:
            raise ValueError("Please provide an integer value for your random seed")
            

        # only keep what we need
        self.data = self.data[[y, x, cluster_field]]

        # set the rest
        self.contours = dict()
        self.ocean_colour = None
        self.fig = None
        self.cmap = None
        self.max_contour = None
        self.shadows = list()
        self.highlight = list()
        self.topos = list()
        self.relief = list()

    def _rescale_coordinates(self):
        """Rescale provided components as pseudo latitudes and longitudes."""
        # trying to prevent issues at the extremes
        lat_scaler = MinMaxScaler(feature_range=(-80, 80))
        long_scaler = MinMaxScaler(feature_range=(-170, 170))

        self.data["Latitude"] = lat_scaler.fit_transform(
            self.data[self.y].values.reshape(-1, 1)
        ).reshape(-1)
        self.data["Longitude"] = long_scaler.fit_transform(
            self.data[self.x].values.reshape(-1, 1)
        ).reshape(-1)

        # self.data.plot(kind='scatter',
        #                 x='Longitude',
        #                 y='Latitude',
        #                 c=self.cluster_field,
        #                 cmap='Spectral')
        # plt.show()

    def _get_contours(self, cluster, subset, topography_levels, lighting_levels, relief_density):
        """Generate contour lines based on density of points per
        cluster/class."""

        # this is required since we need to throw some of them away later
        topography_levels += 5

        y = subset["Latitude"].values
        x = subset["Longitude"].values

        # Define the borders
        deltaX = (max(x) - min(x)) / 3
        deltaY = (max(y) - min(y)) / 3
        xmin = max(-180, min(x) - deltaX)
        xmax = min(180, max(x) + deltaX)
        ymin = max(-90, min(y) - deltaY)
        ymax = min(90, max(y) + deltaY)
        # print(xmin, xmax, ymin, ymax)
        # Create meshgrid
        # todo: let a user specify the grid density
        xx, yy = np.mgrid[
            xmin : xmax : (30 * 10 + 1j),#(30 * topography_levels + 1j),
            ymin : ymax : (30 * 10 + 1j),#(30 * topography_levels + 1j),
        ]

        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([x, y])
        kernel = st.gaussian_kde(values)
        # an attempt at adding slightly more detail to the relief
        kernel.set_bandwidth(bw_method=kernel.factor / 1.2)
        f = np.reshape(kernel(positions).T, xx.shape)
        self.topos.append(f)

        hillshade = self._calculate_hillshade(np.rot90(f), 315, 45)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.gca()
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        # cfset = ax.contourf(xx, yy, f, cmap='coolwarm')
        # ax.imshow(np.rot90(f), cmap='coolwarm', extent=[-180, 180, -90, 90])
        cset = ax.contour(xx, yy, f, colors="k", levels=topography_levels)
        plt.close(fig)

        cntrs = self._clean_contours(self._get_contour_verts(cset))

        self.contours[cluster] = cntrs

        self.generate_hillshade_polygons(
            hillshade, xx, yy, xmin, xmax, ymin, ymax, lighting_levels
        )
        self.generate_highlight_polygons(
            hillshade, xx, yy, xmin, xmax, ymin, ymax, lighting_levels
        )
        self.relief.append(self.generate_relief(f, xx, yy, cntrs, relief_density))

        return cntrs
    
    def _get_contour_verts(self, cn):
        """Get the vertices from the mpl plot to generate our own
        geometries."""
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

    def generate_hillshade_polygons(
        self, hillshade, xx, yy, xmin, xmax, ymin, ymax, lighting_levels
    ):

        # self.shadows = list()

        # we have to strech it for the opencv function to catch the edges properly
        hs_array = (
            (hillshade - hillshade.min()) / (hillshade.max() - hillshade.min()) * 255
        )
        hist, bin_edges = np.histogram(hs_array, bins=lighting_levels + 5)
        # bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])

        # still need to refine this, but this piece here should help catch only the shadows and not the 'light side'
        bin_edges = [x for x in bin_edges if x > 180]

        cluster_shadows = []
        for b in list(zip(bin_edges[:-1], bin_edges[1:])):
            hs_array_binary_slice = hs_array.copy()
            hs_array_binary_slice[
                (hs_array_binary_slice < b[0]) & (hs_array_binary_slice != 1)
            ] = 0
            hs_array_binary_slice[
                (hs_array_binary_slice >= b[0]) & (hs_array_binary_slice < b[1])
            ] = 1
            # hs_array_binary_slice[(hs_array_binary_slice>=b[1]) & (hs_array_binary_slice != 1)] = 0

            hs_array_binary_slice = np.flipud(hs_array_binary_slice)
            hs_array_binary_slice = hs_array_binary_slice.astype(np.uint8)

            # plt.imshow(hs_array_binary_slice,cmap='Greys', extent=[xmin, xmax, ymin, ymax])
            # plt.show()

            contours, hierarchy = cv.findContours(
                hs_array_binary_slice.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
            )
            for cntr in contours:
                x_loc = [xx[pair[0][0], pair[0][1]] for pair in cntr]
                y_loc = [yy[pair[0][0], pair[0][1]] for pair in cntr]

                # get rid of polygons that touch the bondary of the calculated extent
                if (
                    xmin not in x_loc
                    and xmax not in x_loc
                    and ymin not in y_loc
                    and ymax not in y_loc
                ):
                    coords = list(zip(x_loc + [x_loc[0]], y_loc + [y_loc[0]]))
                    if len(coords) > 3:
                        # attempt some smoothing and reorienting of generated polygons
                        coords = list(
                            asPolygon(coords)
                            .simplify(0.01)
                            .buffer(3, join_style=1)
                            .buffer(-3, join_style=1)
                            .exterior.coords
                        )
                        cluster_shadows.append(coords)
        self.shadows.append(cluster_shadows)

    def generate_highlight_polygons(
        self, hillshade, xx, yy, xmin, xmax, ymin, ymax, lighting_levels
    ):

        # self.shadows = list()

        # we have to strech it for the opencv function to catch the edges properly
        hs_array = (
            (hillshade - hillshade.min()) / (hillshade.max() - hillshade.min()) * 255
        )
        hist, bin_edges = np.histogram(hs_array, bins=lighting_levels + 5)
        # bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])

        # still need to refine this, but this piece here should help catch only the 'light side' highlights
        bin_edges = [x for x in bin_edges if x <= 70]

        highlight = []
        for b in list(zip(bin_edges[:-1], bin_edges[1:])):
            hs_array_binary_slice = hs_array.copy()
            hs_array_binary_slice[
                (hs_array_binary_slice < b[0]) & (hs_array_binary_slice != 1)
            ] = 0
            hs_array_binary_slice[
                (hs_array_binary_slice >= b[0]) & (hs_array_binary_slice < b[1])
            ] = 1
            # hs_array_binary_slice[(hs_array_binary_slice>=b[1]) & (hs_array_binary_slice != 1)] = 0

            hs_array_binary_slice = np.flipud(hs_array_binary_slice)
            hs_array_binary_slice = hs_array_binary_slice.astype(np.uint8)

            # plt.imshow(hs_array_binary_slice,cmap='Greys', extent=[xmin, xmax, ymin, ymax])
            # plt.show()

            contours, hierarchy = cv.findContours(
                hs_array_binary_slice.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
            )
            for cntr in contours:
                x_loc = [xx[pair[0][0], pair[0][1]] for pair in cntr]
                y_loc = [yy[pair[0][0], pair[0][1]] for pair in cntr]

                # get rid of polygons that touch the bondary of the calculated extent
                if (
                    xmin not in x_loc
                    and xmax not in x_loc
                    and ymin not in y_loc
                    and ymax not in y_loc
                ):
                    coords = list(zip(x_loc + [x_loc[0]], y_loc + [y_loc[0]]))
                    if len(coords) > 3:
                        # attempt some smoothing
                        coords = list(
                            asPolygon(coords)
                            .simplify(0.01)
                            .buffer(3, join_style=1)
                            .buffer(-3, join_style=1)
                            .exterior.coords
                        )
                        highlight.append(coords)
        self.highlight.append(highlight)

        # #plot
        # fig = plt.figure(figsize=(8,8))
        # ax = fig.gca()
        # ax.set_xlim(xmin, xmax)
        # ax.set_ylim(ymin, ymax)
        # plt.imshow(hs_array,cmap='Greys', extent=[xmin, xmax, ymin, ymax])
        # plt.show()

    def get_all_contours(self, topography_levels=20, lighting_levels=20, relief_density=3):
        """Get all of the contours per class."""
        for cluster in tqdm(
            np.unique(self.data["Cluster"].values), desc="Generating data"
        ):
            points_df = self.data.loc[
                self.data["Cluster"] == cluster, ["Longitude", "Latitude"]
            ]
            self._get_contours(cluster, points_df, topography_levels, lighting_levels, relief_density)

    def generate_relief(
        self, f, xx, yy, cntrs, density=3, min_length=0.005, max_length=0.2
    ):
        """Still need to have a proper rationale for this apart from it
        potentially looking good.

        since this effectively represents the gradient of the topography - is it good enough to use
        this as a proxy for either global winds and eventually repurpose for ocean currents now?
        Need to think about this carefully.
        Need to find a way to put a legitimate 'spin' on the primary axis of the planetoid and model this out
        for the currents
        """

        # create a matplotlib figure and adjust the width and heights
        fig = plt.figure()

        # create a single subplot, just takes over the whole figure if only one is specified
        ax = fig.add_subplot(111, frameon=False, xticks=[], yticks=[])

        # create the boundary
        aoe = unary_union(
            [
                asPolygon(x)
                for x in [item for sublist in cntrs for item in sublist]
                if len(x) > 0
            ]
        ).buffer(-3)

        # add a streamplot
        dy, dx = np.gradient(f)
        c = np.sqrt(dx * dx + dy * dy)
        stream_container = plt.streamplot(
            yy,
            xx,
            dx,
            dy,
            color="c",
            density=density,
            linewidth=1.0 * c / c.max(),
            arrowsize=0.1,
            minlength=min_length,
            maxlength=max_length,
        )

        # this is the data we're extracting from the relief
        widths = np.round(stream_container.lines.get_linewidth(), 1)
        segments = stream_container.lines.get_segments()

        segments_with_width = [
            [segments[i], widths[i]] for i in range(0, len(segments))
        ]

        cleaned = [
            [asLineString(p[0][:, [1, 0]]), p[1]]
            for p in segments_with_width
            if -90 < p[0][0].any() < 90 and -180 < p[0][1].any() < 180
        ]
        stream_container = [p for p in cleaned if p[0].intersects(aoe)]

        plt.close(fig)

        return stream_container

    def _clean_contours(self, cntrs):
        """Use Shapely to modify the contours to prevent the case where Plotly
        fills the inverted section instead."""
        cleaned = list()
        for ix, line in enumerate(cntrs):
            for il, l in enumerate(line):
                # expanding and contracting like this has a smoothing effect
                poly = (
                    asPolygon(l).buffer(0.01, join_style=1).buffer(-0.01, join_style=1)
                )
                if poly.geom_type == "MultiPolygon":
                    polys = [np.array(p.exterior.coords) for p in list(poly)]
                    coords = []
                    for co in coords:
                        if co.shape[0] >= 3:
                            coords.append(co)
                    cleaned.append(coords)

                else:
                    coords = np.array(poly.exterior.coords)
                    if coords.shape[0] >= 3:
                        cleaned.append([coords])
        return cleaned

    def _calculate_hillshade(self, array, azimuth, angle_altitude):
        """Calculate a hillshade over the generated topography."""

        # hacky fix for now - need to trace what's making the mirroring necessary
        azimuth += 180
        if azimuth >= 360:
            azimuth = azimuth - 360

        x, y = np.gradient(array)
        slope = np.pi / 2.0 - np.arctan(np.sqrt(x * x + y * y))
        aspect = np.arctan2(-x, y)
        azimuthrad = azimuth * np.pi / 180.0
        altituderad = angle_altitude * np.pi / 180.0
        shaded = np.sin(altituderad) * np.sin(slope) + np.cos(altituderad) * np.cos(
            slope
        ) * np.cos(azimuthrad - aspect)
        return 255 * (shaded + 1) / 2

    def plot_surface(self):
        """This plots the surface layer which we need because we can't set it
        directly."""
        # globe
        self.fig.add_trace(
            go.Scattergeo(
                lon=[-179.9, 179.9, 179.9, -179.9],
                lat=[89.9, 89.9, -89.9, -89.9],
                mode="lines",
                line=dict(width=1, color=self.ocean_colour),
                fill="toself",
                fillcolor=self.ocean_colour,
                hoverinfo="skip",
                opacity=1,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    def plot_shadows(self):
        """Plot the hillshade-derived shadows."""
        # globe
        for cluster in tqdm(self.shadows, desc="Plotting Shadows"):
            for ix, shadow in enumerate(cluster):
                if ix % 2 == 0:
                    shadow_array = np.array(shadow)
                    self.fig.add_trace(
                        go.Scattergeo(
                            lon=list(shadow_array[:, 0]),
                            lat=list(shadow_array[:, 1]),
                            hoverinfo="skip",
                            mode="lines",
                            line=dict(width=0, color="black"),
                            fill="toself",
                            fillcolor="black",
                            opacity=0.05 + (ix/len(cluster)*0.1),
                            showlegend=False,
                        ),
                        row=2,
                        col=1,
                    )

    def plot_highlight(self):
        """Plot the hillshade-derived lighting."""
        # globe
        for cluster in tqdm(self.highlight, desc="Plotting highlight"):
            for ix, lighting in enumerate(cluster):
                if ix % 2 == 0:
                    lighting_array = np.array(lighting)
                    self.fig.add_trace(
                        go.Scattergeo(
                            lon=list(lighting_array[:, 0]),
                            lat=list(lighting_array[:, 1]),
                            hoverinfo="skip",
                            mode="lines",
                            line=dict(width=0, color="white"),
                            fill="toself",
                            fillcolor="white",
                            opacity=0.01 + (ix/len(cluster)*0.1),
                            showlegend=False,
                        ),
                        row=2,
                        col=1,
                    )

    def plot_contours(self):
        """Plot the topography."""
        for cluster, cntrs in tqdm(self.contours.items(), desc="Plotting contours"):
            #introduce some randomness in the topography layering
            contours = cntrs.copy()
            dont_shuffle_start = contours[0:5]
            dont_shuffle_end = contours[-2:]
            do_shuffle = contours[5:-2]
            random.shuffle(do_shuffle)
            contours = dont_shuffle_start + do_shuffle + dont_shuffle_end
            
            for ix, line in enumerate(contours):
                if ix > (self.max_contour - 3) / len(contours) + 2:
                    if ix % 2 == 0:
                        for l in line:
                            self.fig.add_trace(
                                go.Scattergeo(
                                    lon=list(l[:, 0]),
                                    lat=list(l[:, 1]),
                                    hoverinfo="skip",
                                    mode="lines",
                                    line=dict(
                                        width=0,  # *np.power(np.exp(ix/max_contour),2),
                                        dash="longdashdot",
                                        color="rgb"
                                        + str(
                                            self.cmap(
                                                ix / self.max_contour, bytes=True
                                            )[0:3]
                                        ),
                                    ),
                                    fill="toself",
                                    fillcolor="rgb"
                                    + str(
                                        self.cmap(ix / self.max_contour, bytes=True)[
                                            0:3
                                        ]
                                    ),
                                    opacity=0.1 + ((ix / self.max_contour) * 0.5),
                                    showlegend=False,
                                ),
                                row=2,
                                col=1,
                            )
                    else:
                        for l in line:
                            self.fig.add_trace(
                                go.Scattergeo(
                                    lon=list(l[:, 0]),
                                    lat=list(l[:, 1]),
                                    hoverinfo="skip",
                                    mode="lines",
                                    line=dict(
                                        width=1,  # *np.power(np.exp(ix/max_contour),2),
                                        dash="longdashdot",
                                        color="rgb"
                                        + str(
                                            self.cmap(
                                                ix / self.max_contour, bytes=True
                                            )[0:3]
                                        ),
                                    ),
                                    opacity=0.1 + ((ix / self.max_contour) * 0.5),
                                    showlegend=False,
                                ),
                                row=2,
                                col=1,
                            )

    def plot_relief(self):
        """Plot the relief."""
        # globe
        for cluster in tqdm(self.relief, desc="Plotting relief"):

            for size in np.unique([x[1] for x in cluster]):

                # need to be smarter about segments that touch

                stream_array = np.array(
                    [stream[0].coords for stream in cluster if stream[1] == size]
                )
                stream_array = np.concatenate(
                    [
                        item
                        for sublist in [
                            [x, np.array([[None, None], [None, None]])]
                            for x in stream_array
                        ]
                        for item in sublist
                    ]
                )

                self.fig.add_trace(
                    go.Scattergeo(
                        connectgaps=False,
                        lon=list(stream_array[:, 0]),
                        lat=list(stream_array[:, 1]),
                        hoverinfo="skip",
                        mode="lines",
                        line=dict(
                            width=2 * size,
                            # dash='dot',
                            color='black'#"rgb"
                            #+ str(
                            #    self.cmap(int(stream_array.shape[0] / 3), bytes=True)[0:3]
                            #),
                        ),
                        opacity=0.1 + 0.15 * (1 / np.cos(size) - 1),
                        showlegend=False,
                    ),
                    row=2,
                    col=1,
                )

    def plot_clustered_points(self):
        """Plot the provided point data."""

        # globe
        self.fig.add_trace(
            go.Scattergeo(
                lon=self.data["Longitude"],
                lat=self.data["Latitude"],
                marker_color=self.data["Cluster"],
                hoverinfo="text",
                hovertext=self.data["Cluster"],
                marker_size=2,
                showlegend=False
                #     marker = dict(
                #         symbol='circle-open',
                #      )
            ),
            row=2,
            col=1,
        )

    def update_geos(self):
        """Update config for maps."""
        # globe
        self.fig.update_geos(
            row=2,
            col=1,
            showland=False,
            showcountries=False,
            showocean=False,
            showcoastlines=False,
            showframe=False,
            showrivers=False,
            showlakes=False,
            showsubunits=False,
            bgcolor="rgba(0,0,0,0)",
            projection=dict(type=self.projection, rotation=dict(lon=0, lat=0, roll=0)),
            lonaxis=dict(showgrid=True, gridcolor="rgb(102, 102, 102)", gridwidth=1),
            lataxis=dict(showgrid=True, gridcolor="rgb(102, 102, 102)", gridwidth=1),
        )

    def add_empty_trace(self):
        """Add invisible scatter trace.

        This trace is added to help the autoresize logic work.
        """

        width = int(1920 / 2)
        height = int(1280 / 2)
        self.fig.add_trace(
            go.Scatter(
                x=[0, width],
                y=[0, height],
                mode="markers",
                marker_opacity=0,
                showlegend=False,
            )
        )

        # Configure axes
        self.fig.update_xaxes(visible=False, fixedrange=True, range=[0, width])

        self.fig.update_yaxes(
            visible=False,
            fixedrange=True,
            range=[0, height],
            # the scaleanchor attribute ensures that the aspect ratio stays constant
            scaleanchor="x",
        )

    def update_layout(self, planet_name="Planetoids"):
        """Update layout config."""

        width = int(1920 / 2)
        height = int(1280 / 2)

        image_array = np.zeros((width, height))
        image_array = self._add_salt_and_pepper(image_array, 0.001).astype("uint8")
        image = Image.fromarray(image_array)

        self.fig.update_layout(
            autosize=True,
            width=width,
            height=height,
            title_text=planet_name,
            showlegend=True,
            dragmode="pan",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=0, b=0),
            images=[
                dict(
                    source=image,
                    xref="x",
                    yref="y",
                    x=0,
                    y=height,
                    sizex=width,
                    sizey=height,
                    sizing="stretch",
                    opacity=1,
                    layer="below",
                )
            ],
        )
        
    def fit(self, topography_levels=20, lighting_levels=20, relief_density=3):
        """Generate data required for terraforming."""
        # transform 2d components into pseudo lat/longs
        self._rescale_coordinates()
        # generate contours per class
        self.get_all_contours(topography_levels, lighting_levels, relief_density)
        self.data_generated = True

    def terraform(
        self,
        plot_topography=True,
        plot_points=True,
        plot_lighting=True,
        projection = 'orthographic',
        planet_name="Planetoids",
        render=True,
    ):
        """Construct a new world."""
        
        self.projection = projection
        
        if not self.data_generated:
            raise Exception("Please first run .fit() before attemption to terraform.")
        else:
            self.fig = make_subplots(
                rows=3,
                cols=2,
                vertical_spacing=0.05,
                # column_widths=[0.5, 0.5],
                row_heights=[0.05, 0.93, 0.02],
                specs=[
                    [None, None],
                    [{"type": "scattergeo", "colspan": 2}, None],
                    [None, None],
                ],
                subplot_titles=(planet_name, ""),
            )

            self.add_empty_trace()

            # identify the maximum number of contours per continent
            self.max_contour = max([len(contour) for contour in self.contours.values()])
            self.cmap = cm.get_cmap(self.ecology, self.max_contour + 1)

            self.ocean_colour = "rgb" + str(
                self.cmap(1 / self.max_contour, bytes=True)[0:3]
            )

            self.plot_surface()

            if plot_topography:
                self.plot_contours()

            self.plot_relief()

            if plot_lighting:
                self.plot_highlight()
                self.plot_shadows()

            if plot_points:
                self.plot_clustered_points()

            self.update_geos()

            self.update_layout(planet_name)

            if render:
                self.fig.show()

    def fit_terraform(
        self,
        topography_levels=20,
        lighting_levels=20,
        relief_density=3,
        plot_topography=True,
        plot_points=True,
        plot_lighting=True,
        projection='orthographic',
        planet_name="Planetoids",
        render=True,
    ):
        """Fit and terraform in a single step, akin to fit_transform people are
        used to."""
        self.fit(topography_levels=topography_levels, lighting_levels=lighting_levels, relief_density=relief_density)
        self.terraform(plot_topography, plot_points, plot_lighting, projection, planet_name, render)

    def save(self, filename="planetoid.html", output_type='file', include_plotlyjs=True, auto_open=False):
        offline.plot(self.fig, filename = filename, output_type=output_type, include_plotlyjs=include_plotlyjs, auto_open=auto_open)


    def _add_salt_and_pepper(self, gb, prob):
        """Adds "Salt & Pepper" noise to an image.

        gb: should be one-channel image with pixels in [0, 1] range
        prob: probability (threshold) that controls level of noise
        """

        rnd = np.random.rand(gb.shape[0], gb.shape[1])
        noisy = gb.copy()
        noisy[rnd < prob] = 0
        noisy[rnd > 1 - prob] = 255
        return noisy
