import sys

sys.path.append("..")

import pytest
import pandas as pd
import numpy as np
import pickle
from sklearn.datasets import make_blobs
from planetoids import planetoids as pt

np.random.seed(42)


def test_rescale_coordinates():
    print("Testing coord scaling")
    # np.save('rescaled', planet.data[['Latitude', 'Longitude']])
    planet._rescale_coordinates()
    np.testing.assert_array_almost_equal(
        planet.data[["Latitude", "Longitude"]].values, np.load("test/rescaled.npy")
    )


def test_get_all_contours():
    print("Testing get all contours")

    planet.get_all_contours(topography_levels=20, lighting_levels=20, relief_density=3)

    # with open("test/contours.pickle", "wb") as handle:
    #     pickle.dump(planet.contours, handle)

    contours = None
    with open("test/contours.pickle", "rb") as handle:
        contours = pickle.load(handle)

    for k, v in planet.contours.items():
        for ix, vv in enumerate(v):
            for ixx, vvv in enumerate(vv):
                np.testing.assert_almost_equal(vvv, contours[k][ix][ixx], verbose=True)


def test_get_contours():
    print("Testing get contours")
    cntrs = planet._get_contours(
        0,
        planet.data[planet.data["Cluster"] == 0],
        topography_levels=20,
        lighting_levels=20,
        relief_density=3,
    )
    # with open("test/cntrs_0.pickle", "wb") as handle:
    #     pickle.dump(cntrs, handle)

    contours = None
    with open("test/cntrs_0.pickle", "rb") as handle:
        contours = pickle.load(handle)
    for ix, v in enumerate(cntrs):
        for ixx, vv in enumerate(v):
            np.testing.assert_almost_equal(vv, contours[ix][ixx], verbose=True)


def test_get_contour_verts():
    subset = planet.data[planet.data["Cluster"] == 0]
    print("Testing get contour vertices")
    import scipy.stats as st
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()

    y = subset["Latitude"].values
    x = subset["Longitude"].values

    # Define the borders
    deltaX = (max(x) - min(x)) / 3
    deltaY = (max(y) - min(y)) / 3
    xmin = max(-180, min(x) - deltaX)
    xmax = min(180, max(x) + deltaX)
    ymin = max(-90, min(y) - deltaY)
    ymax = min(90, max(y) + deltaY)

    xx, yy = np.mgrid[xmin : xmax : (30 * 10 + 1j), ymin : ymax : (30 * 10 + 1j)]

    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    cset = ax.contour(xx, yy, f, colors="k", levels=20)
    plt.close(fig)

    # with open("test/verts_0.pickle", "wb") as handle:
    #     pickle.dump(planet._get_contour_verts(cset), handle)

    verts = None
    with open("test/verts_0.pickle", "rb") as handle:
        verts = pickle.load(handle)

    verts_calc = planet._get_contour_verts(cset)
    for ix, v in enumerate(verts):
        for ixx, vv in enumerate(v):
            np.testing.assert_almost_equal(vv, verts_calc[ix][ixx], verbose=True)


def test_clean_contours():
    print("Testing contour cleaning")
    verts = None
    clean_verts = None

    with open("test/verts_0.pickle", "rb") as handle:
        verts = pickle.load(handle)

    # with open("test/clean_verts_0.pickle", "wb") as handle:
    #     pickle.dump(planet._clean_contours(verts), handle)

    with open("test/clean_verts_0.pickle", "rb") as handle:
        clean_verts = pickle.load(handle)

    np.testing.assert_equal(clean_verts, planet._clean_contours(verts), verbose=True)


def test_calculate_hillshade():
    print("Testing hillshade generation")
    import scipy.stats as st
    import matplotlib.pyplot as plt

    subset = planet.data[planet.data["Cluster"] == 0]

    y = subset["Latitude"].values
    x = subset["Longitude"].values

    # Define the borders
    deltaX = (max(x) - min(x)) / 3
    deltaY = (max(y) - min(y)) / 3
    xmin = max(-180, min(x) - deltaX)
    xmax = min(180, max(x) + deltaX)
    ymin = max(-90, min(y) - deltaY)
    ymax = min(90, max(y) + deltaY)

    xx, yy = np.mgrid[xmin : xmax : (30 * 10 + 1j), ymin : ymax : (30 * 10 + 1j)]

    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    # np.save("test/hillshade.npy", planet._calculate_hillshade(np.rot90(f), 315, 45))

    np.testing.assert_array_equal(
        planet._calculate_hillshade(np.rot90(f), 315, 45),
        np.load("test/hillshade.npy"),
        verbose=True,
    )

    return xx, yy, xmin, xmax, ymin, ymax


def test_generate_hillshade_polygons():
    print("Testing shadow polygon generation")
    planet.generate_hillshade_polygons(
        np.load("test/hillshade.npy"), xx, yy, xmin, xmax, ymin, ymax, 20
    )
    hs_poly = planet.shadows
    # with open("test/hs_poly.pickle", "wb") as handle:
    #     pickle.dump(hs_poly, handle)

    hs_poly_check = None
    with open("test/hs_poly.pickle", "rb") as handle:
        hs_poly_check = pickle.load(handle)

    for ix, v in enumerate(hs_poly_check):
        for ixx, vv in enumerate(v):
            assert vv == hs_poly[ix][ixx]


def test_generate_highlight_polygons():
    print("Testing highlight polygon generation")
    planet.generate_highlight_polygons(
        np.load("test/hillshade.npy"), xx, yy, xmin, xmax, ymin, ymax, 20
    )
    highlight_poly = planet.highlight
    # with open("test/highlight_poly.pickle", "wb") as handle:
    #     pickle.dump(highlight_poly, handle)

    highlight_poly_check = None
    with open("test/highlight_poly.pickle", "rb") as handle:
        highlight_poly_check = pickle.load(handle)

    for ix, v in enumerate(highlight_poly_check):
        for ixx, vv in enumerate(v):
            assert vv == highlight_poly[ix][ixx]


##################################################
# Super basic testing for now to get the mvp out #
# This needs a lot of attention still.           #
##################################################

data = pd.read_csv("test/test_data.csv")
planet = pt.Planetoid(data, "0", "1", "Cluster", random_state=42)

test_rescale_coordinates()
test_get_all_contours()
test_get_contours()
test_get_contour_verts()
test_clean_contours()
xx, yy, xmin, xmax, ymin, ymax = test_calculate_hillshade()
test_generate_hillshade_polygons()
test_generate_highlight_polygons()

