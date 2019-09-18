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
    """A procedurally generated world seeded from two dimensional data,
    optionally clustered.

    A `Planetoid` contains all the required material to generate a new world from a minimal set of input data.

    Apart from looking beautiful, the generated features can be interpreted analytically.

    # **Parameters**
    ----------
    `data` : `DataFrame`  
        Pandas `DataFrame` holding the seed data used to generate the `Planetoid`

    `y` : `string`  
        Column name for y-axis seed from data, this will be used to generate `latitudes`

    `x` : `string`  
        Column name for x-axis seed from data, this will be used to generate `longitudes`

    `cluster_field` : `string`, optional (default=`None`)  
        Optional column name for cluster attribute from data, this will be used to generate `land masses` independently

    `ecology` : `string` (default `gist_earth`)  
        Any one of the named `colormap` references from [**matplotlib**](https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html)

    `random_state` : `int`, optional (default=`None`)  
        Optional `integer` to seed the random number generators for reproducibility if required

    # **Examples**
    ----------
    See [**examples**]()
    """

    def __init__(
        self, data, y, x, cluster_field=None, ecology="gist_earth", random_state=None
    ):
        self._data = None
        self._y = None
        self._x = None
        self._cluster_field = None
        self._ecology = None
        self._random_state = None

        self._data_generated = False

        if isinstance(data, pd.DataFrame):
            self._data = data
        else:
            raise ValueError("Please provide a pandas DataFrame")
        if y in self._data.columns:
            self._y = y
        else:
            raise ValueError("X field not in provided DataFrame")
        if x in self._data.columns:
            self._x = x
        else:
            raise ValueError("Y field not in provided DataFrame")
        if cluster_field is not None or cluster_field in self._data.columns:
            self._cluster_field = cluster_field
        else:
            raise ValueError("Cluster field not in provided DataFrame")
        try:
            cm.get_cmap(ecology, 1)
            self._ecology = ecology
        except Exception as e:
            raise ValueError(e)
        if isinstance(random_state, int):
            self._random_state = random_state
            np.random.seed(self.random_state)
            random.seed(self.random_state)
            cv.setRNGSeed(self.random_state)
        elif random_state is None:
            pass
        else:
            raise ValueError("Please provide an integer value for your random seed")

        # only keep what we need
        self._data = self._data[[y, x, cluster_field]]

        # set the rest
        self._contours = dict()
        self._ocean_colour = None
        self._fig = None
        self._cmap = None
        self._max_contour = None
        self._shadows = list()
        self._highlight = list()
        self._topos = list()
        self._relief = list()

    @property
    def data(self):
        """
        Pandas `DataFrame` holding the seed data used to generate the `Planetoid`.
        """
        return self._data

    @property
    def y(self):
        """
        Column name for y-axis seed from data, this will be used to generate `latitudes`.
        """
        return self._y

    @property
    def x(self):
        """
        Column name for x-axis seed from data, this will be used to generate `longitudes`.
        """
        return self._x

    @property
    def cluster_field(self):
        """
        Optional column name for cluster attribute from data, this will be used
        to generate `land masses` independently.
        """
        return self._cluster_field

    @property
    def ecology(self):
        """
        Any one of the named `colormap` references from [**matplotlib**](https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html)
        """
        return self._ecology

    @property
    def random_state(self):
        """
        Optional `integer` to seed the random number generators for
        reproducibility if required.
        """
        return self._random_state
    
    @property
    def fig(self):
        """
        Plotly graph object of the terraformed `Planetoid`.
        """
        return self._fig

    def _rescale_coordinates(self):
        """
        Rescale provided components as pseudo latitudes and longitudes.
        """
        # trying to prevent issues at the extremes
        lat_scaler = MinMaxScaler(feature_range=(-80, 80))
        long_scaler = MinMaxScaler(feature_range=(-170, 170))

        self._data["Latitude"] = lat_scaler.fit_transform(
            self._data[self.y].values.reshape(-1, 1)
        ).reshape(-1)
        self._data["Longitude"] = long_scaler.fit_transform(
            self._data[self.x].values.reshape(-1, 1)
        ).reshape(-1)

        # self._data.plot(kind='scatter',
        #                 x='Longitude',
        #                 y='Latitude',
        #                 c=self.cluster_field,
        #                 cmap='Spectral')
        # plt.show()

    def _get_contours(
        self, cluster, subset, topography_levels, lighting_levels, relief_density
    ):
        """
        Generate contour lines based on density of points per cluster/class.
        """

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
            xmin : xmax : (30 * 10 + 1j),  # (30 * topography_levels + 1j),
            ymin : ymax : (30 * 10 + 1j),  # (30 * topography_levels + 1j),
        ]

        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([x, y])
        kernel = st.gaussian_kde(values)
        # an attempt at adding slightly more detail to the relief
        kernel.set_bandwidth(bw_method=kernel.factor / 1.2)
        f = np.reshape(kernel(positions).T, xx.shape)
        self._topos.append(f)

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

        self._contours[cluster] = cntrs

        self._generate_hillshade_polygons(
            hillshade, xx, yy, xmin, xmax, ymin, ymax, lighting_levels
        )
        self._generate_highlight_polygons(
            hillshade, xx, yy, xmin, xmax, ymin, ymax, lighting_levels
        )
        self._relief.append(self._generate_relief(f, xx, yy, cntrs, relief_density))

        return cntrs

    def _get_contour_verts(self, cn):
        """
        Get the vertices from the mpl plot to generate our own geometries.
        """
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

    def _generate_hillshade_polygons(
        self, hillshade, xx, yy, xmin, xmax, ymin, ymax, lighting_levels
    ):
        """Generate the hillshade (shadow) polygons"""

        # self._shadows = list()

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
        self._shadows.append(cluster_shadows)

    def _generate_highlight_polygons(
        self, hillshade, xx, yy, xmin, xmax, ymin, ymax, lighting_levels
    ):

        # self._shadows = list()

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
        self._highlight.append(highlight)

        # #plot
        # fig = plt.figure(figsize=(8,8))
        # ax = fig.gca()
        # ax.set_xlim(xmin, xmax)
        # ax.set_ylim(ymin, ymax)
        # plt.imshow(hs_array,cmap='Greys', extent=[xmin, xmax, ymin, ymax])
        # plt.show()

    def _get_all_contours(
        self, topography_levels=20, lighting_levels=20, relief_density=3
    ):
        """
        Get all of the contours per class.
        """
        for cluster in tqdm(
            np.unique(self._data["Cluster"].values), desc="Generating data"
        ):
            points_df = self._data.loc[
                self._data["Cluster"] == cluster, ["Longitude", "Latitude"]
            ]
            self._get_contours(
                cluster, points_df, topography_levels, lighting_levels, relief_density
            )

    def _generate_relief(
        self, f, xx, yy, cntrs, density=3, min_length=0.005, max_length=0.2
    ):
        """Generate the relief detail for the topography.
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
        """
        Use Shapely to modify the contours to prevent the case where Plotly
        fills the inverted section instead.
        """
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
        """
        Calculate a hillshade over the generated topography.
        """

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

    def _plot_surface(self):
        """
        This plots the surface layer which we need because we can't set it
        directly.
        """
        # globe
        self._fig.add_trace(
            go.Scattergeo(
                lon=[-179.9, 179.9, 179.9, -179.9],
                lat=[89.9, 89.9, -89.9, -89.9],
                mode="lines",
                line=dict(width=1, color=self._ocean_colour),
                fill="toself",
                fillcolor=self._ocean_colour,
                hoverinfo="skip",
                opacity=1,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    def _plot_shadows(self):
        """
        Plot the hillshade-derived shadows.
        """
        # globe
        for cluster in tqdm(self._shadows, desc="Plotting Shadows"):
            for ix, shadow in enumerate(cluster):
                if ix % 2 == 0:
                    shadow_array = np.array(shadow)
                    self._fig.add_trace(
                        go.Scattergeo(
                            lon=list(shadow_array[:, 0]),
                            lat=list(shadow_array[:, 1]),
                            hoverinfo="skip",
                            mode="lines",
                            line=dict(width=0, color="black"),
                            fill="toself",
                            fillcolor="black",
                            opacity=0.05 + (ix / len(cluster) * 0.1),
                            showlegend=False,
                        ),
                        row=2,
                        col=1,
                    )

    def _plot_highlight(self):
        """
        Plot the hillshade-derived lighting.
        """
        # globe
        for cluster in tqdm(self._highlight, desc="Plotting highlight"):
            for ix, lighting in enumerate(cluster):
                if ix % 2 == 0:
                    lighting_array = np.array(lighting)
                    self._fig.add_trace(
                        go.Scattergeo(
                            lon=list(lighting_array[:, 0]),
                            lat=list(lighting_array[:, 1]),
                            hoverinfo="skip",
                            mode="lines",
                            line=dict(width=0, color="white"),
                            fill="toself",
                            fillcolor="white",
                            opacity=0.01 + (ix / len(cluster) * 0.1),
                            showlegend=False,
                        ),
                        row=2,
                        col=1,
                    )

    def _plot_contours(self):
        """
        Plot the topography.
        """
        for cluster, cntrs in tqdm(self._contours.items(), desc="Plotting contours"):
            # introduce some randomness in the topography layering
            contours = cntrs.copy()
            dont_shuffle_start = contours[0:5]
            dont_shuffle_end = contours[-2:]
            do_shuffle = contours[5:-2]
            random.shuffle(do_shuffle)
            contours = dont_shuffle_start + do_shuffle + dont_shuffle_end

            for ix, line in enumerate(contours):
                if ix > (self._max_contour - 3) / len(contours) + 2:
                    if ix % 2 == 0:
                        for l in line:
                            self._fig.add_trace(
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
                                            self._cmap(
                                                ix / self._max_contour, bytes=True
                                            )[0:3]
                                        ),
                                    ),
                                    fill="toself",
                                    fillcolor="rgb"
                                    + str(
                                        self._cmap(ix / self._max_contour, bytes=True)[
                                            0:3
                                        ]
                                    ),
                                    opacity=0.1 + ((ix / self._max_contour) * 0.5),
                                    showlegend=False,
                                ),
                                row=2,
                                col=1,
                            )
                    else:
                        for l in line:
                            self._fig.add_trace(
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
                                            self._cmap(
                                                ix / self._max_contour, bytes=True
                                            )[0:3]
                                        ),
                                    ),
                                    opacity=0.1 + ((ix / self._max_contour) * 0.5),
                                    showlegend=False,
                                ),
                                row=2,
                                col=1,
                            )

    def _plot_relief(self):
        """
        Plot the relief.
        """
        # globe
        for cluster in tqdm(self._relief, desc="Plotting relief"):

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

                self._fig.add_trace(
                    go.Scattergeo(
                        connectgaps=False,
                        lon=list(stream_array[:, 0]),
                        lat=list(stream_array[:, 1]),
                        hoverinfo="skip",
                        mode="lines",
                        line=dict(
                            width=2 * size,
                            # dash='dot',
                            color="black"  # "rgb"
                            # + str(
                            #    self._cmap(int(stream_array.shape[0] / 3), bytes=True)[0:3]
                            # ),
                        ),
                        opacity=0.1 + 0.15 * (1 / np.cos(size) - 1),
                        showlegend=False,
                    ),
                    row=2,
                    col=1,
                )

    def _plot_clustered_points(self):
        """
        Plot the provided point data.
        """

        # globe
        self._fig.add_trace(
            go.Scattergeo(
                lon=self._data["Longitude"],
                lat=self._data["Latitude"],
                marker_color=self._data["Cluster"],
                hoverinfo="text",
                hovertext=self._data["Cluster"],
                marker_size=2,
                showlegend=False
                #     marker = dict(
                #         symbol='circle-open',
                #      )
            ),
            row=2,
            col=1,
        )

    def _update_geos(self):
        """
        Update config for maps.
        """
        # globe
        self._fig.update_geos(
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

    def _add_empty_trace(self):
        """Add invisible scatter trace.

        This trace is added to help the autoresize logic work.
        """

        width = 1920
        height = 1280
        self._fig.add_trace(
            go.Scatter(
                x=[0, width],
                y=[0, height],
                mode="markers",
                marker_opacity=0,
                showlegend=False,
            )
        )

        # Configure axes
        self._fig.update_xaxes(visible=False, fixedrange=True, range=[0, width])

        self._fig.update_yaxes(
            visible=False,
            fixedrange=True,
            range=[0, height],
            # the scaleanchor attribute ensures that the aspect ratio stays constant
            scaleanchor="x",
        )

    def _update_layout(self, planet_name="Planetoids"):
        """
        Update layout config.
        """

        width = 1920
        height = 1280

        image_array = np.zeros((width, height))
        image_array = self._add_salt_and_pepper(image_array, 0.001).astype("uint8")
        image = Image.fromarray(image_array)

        self._fig.update_layout(
            autosize=True,
            width=None,
            height=None,
            title_text=planet_name,
            showlegend=False,
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

    def fit(
        self,
        topography_levels=20,
        lighting_levels=20,
        relief_density=3,
        rescale_coordinates=True,
    ):
        """Use the seed data to generate data required to terraform the
        `Planetoid`.

        This function takes the seed data and constructs the base components required to terraform the planet.

        # **Parameters**
        ----------
        `topography_levels` : `int` (default=`20`)  
            Used to control the number of contours that are generated to represent topographic features.

        `lighting_levels` : `int` (default=`20`)  
            Used to control the number of contours that are generated to represent hillshade and highlight effects.

        `relief_density` : `int` (default=`3`)  
            Used to control the level of detail in the relief of the topography, this represents the gradient of topographic features.

        `rescale_coordinates` : `bool` (default=`True`)  
            Used to specify whether or not input seed `x` and `y` data should be rescaled to global geographic coordinates.
        """
        # transform 2d components into pseudo lat/longs
        if rescale_coordinates:
            self._rescale_coordinates()
        # generate contours per class
        self._get_all_contours(topography_levels, lighting_levels, relief_density)
        self._data_generated = True

    def terraform(
        self,
        plot_topography=True,
        plot_points=True,
        plot_lighting=True,
        projection="orthographic",
        planet_name="Planetoids",
        render=True,
    ):
        """Terraform the `Planetoid`.

        This function takes the fit data and generates an interactive plot.ly figure representing the terraformed `Planetoid`.
        The terraformed world is stored in the `fig` property of the `Planetoid`.


        # **Parameters**
        ----------
        `plot_topography` : `bool` (default=`True`)  
            Used to control whether or not the topography should be rendered.

        `plot_points` : `bool` (default=`True`)  
            Used to control whether or not the seed points should be rendered.

        `plot_lighting` : `bool` (default=`True`)  
            Used to control whether or not the lighting effects should be rendered.

        `projection` : `string` (default=`"orthographic"`)  
            Used to control the map projection of the output world.  
            The default `orthographic` projection produces a 'traditional' 3D globe,
            however more exotic `Planetoidal` views can be generated using other options.   
            Any of the available plotly [**ScatterGeo**](https://plot.ly/python/reference/#scattergeo) map projections can be used:
             - "equirectangular"
             - "mercator"
             - "orthographic"
             - "natural earth"
             - "kavrayskiy7"
             - "miller"
             - "robinson"
             - "eckert4"
             - "azimuthal equal area"
             - "azimuthal equidistant"
             - "conic equal area"
             - "conic conformal"
             - "conic equidistant"
             - "gnomonic"
             - "stereographic"
             - "mollweide"
             - "hammer"
             - "transverse mercator"
             - "albers usa"
             - "winkel tripel"
             - "aitoff"
             - "sinusoidal"

        `planet_name` : `string` (default=`"Planetoids"`)  
            This is a user-defined title that renders on the output figure.

        `render` : `bool` (default=`True`)  
            This controls whether or not the terraformed `Planetoid` should be rendered.
        """

        self.projection = projection

        if not self._data_generated:
            raise Exception("Please first run .fit() before attemption to terraform.")
        else:
            self._fig = make_subplots(
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

            self._add_empty_trace()

            # identify the maximum number of contours per continent
            self._max_contour = max(
                [len(contour) for contour in self._contours.values()]
            )
            self._cmap = cm.get_cmap(self.ecology, self._max_contour + 1)

            self._ocean_colour = "rgb" + str(
                self._cmap(1 / self._max_contour, bytes=True)[0:3]
            )

            self._plot_surface()

            if plot_topography:
                self._plot_contours()

            self._plot_relief()

            if plot_lighting:
                self._plot_highlight()
                self._plot_shadows()

            if plot_points:
                self._plot_clustered_points()

            self._update_geos()

            self._update_layout(planet_name)

            if render:
                self._fig.show()

    def fit_terraform(
        self,
        topography_levels=20,
        lighting_levels=20,
        relief_density=3,
        rescale_coordinates=True,
        plot_topography=True,
        plot_points=True,
        plot_lighting=True,
        projection="orthographic",
        planet_name="Planetoids",
        render=True,
    ):
        """Fit and terraform the `Planetoid` in a single step.

        This function takes the seed data and constructs the base components required to terraform the planet.  
        It then takes the fit data and generates an interactive plot.ly figure representing the terraformed `Planetoid`.  
        The terraformed world is stored in the `fig` property of the `Planetoid`.  


        # **Parameters**
        ----------

        `topography_levels` : `int` (default=`20`)  
            Used to control the number of contours that are generated to represent topographic features.

        `lighting_levels` : `int` (default=`20`)  
            Used to control the number of contours that are generated to represent hillshade and highlight effects.

        `relief_density` : `int` (default=`3`)  
            Used to control the level of detail in the relief of the topography, this represents the gradient of topographic features.

        `rescale_coordinates` : `bool` (default=`True`)  
            `Bool` used to specify whether or not input seed `x` and `y` data should be rescaled to global geographic coordinates.

        `plot_topography` : `bool` (default=`True`)  
            Used to control whether or not the topography should be rendered.

        `plot_points` : `bool` (default=`True`)  
            Used to control whether or not the seed points should be rendered.

        `plot_lighting` : `bool` (default=`True`)  
            Used to control whether or not the lighting effects should be rendered.

        `projection` : `string` (default=`"orthographic"`)  
            Used to control the map projection of the output world.  
            The default `orthographic` projection produces a 'traditional' 3D globe,
            however more exotic `Planetoidal` views can be generated using other options.  
            Any of the available plotly [**ScatterGeo**](https://plot.ly/python/reference/#scattergeo) map projections can be used:
             - "equirectangular"
             - "mercator"
             - "orthographic"
             - "natural earth"
             - "kavrayskiy7"
             - "miller"
             - "robinson"
             - "eckert4"
             - "azimuthal equal area"
             - "azimuthal equidistant"
             - "conic equal area"
             - "conic conformal"
             - "conic equidistant"
             - "gnomonic"
             - "stereographic"
             - "mollweide"
             - "hammer"
             - "transverse mercator"
             - "albers usa"
             - "winkel tripel"
             - "aitoff"
             - "sinusoidal"

        `planet_name` : `string` (default=`"Planetoids"`)  
            This is a user-defined title that renders on the output figure.

        `render` : `bool` (default=`True`)  
            This controls whether or not the terraformed `Planetoid` should be rendered.
        """
        self.fit(
            topography_levels=topography_levels,
            lighting_levels=lighting_levels,
            relief_density=relief_density,
            rescale_coordinates=rescale_coordinates,
        )
        self.terraform(
            plot_topography, plot_points, plot_lighting, projection, planet_name, render
        )

    def save(
        self,
        filename="planetoid.html",
        output_type="file",
        include_plotlyjs=True,
        auto_open=False
    ):
        """Save the `Planetoid` to a file.

        This function takes the graph object and provides a wrapper to save the terraformed `Planetoid`.  

        # **Parameters**
        ----------

        `filename` : `string` (default=`"planetoid.html"`)  
            Set this variable to name and save your output file. The default will create an HTML file in
            the current working directory called "planetoid.html".
            
        `output_type` : `string` (default=`"file"`)  
            Set this variable to the intended output type, either `"file"` or `"div"`.
            
        `include_plotlyjs` : `bool` (default=`True`)
            Allows a user to include or exclude the plotly.js library in the export.
            
        `auto_open` : `bool` (default=`False`)
            If True, the `Planetoid` will open in your web browser after saving.

        """
        offline.plot(       
            self._fig,
            filename=filename,
            output_type=output_type,
            include_plotlyjs=include_plotlyjs,
            auto_open=auto_open,
        )

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
