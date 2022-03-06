import numpy as np

from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import time
from progress.bar import Bar
import os
import pdb


class Tile:
    def __init__(
        self,
        label: bool = 0,
        image: np.ndarray = None,
        resolution: int = 10,
        nodata: float = -999,
        location: np.ndarray = None,
        dem_path: str = None,
        area: str = None,
    ) -> None:

        self.label = label
        self.image = image

        self.resolution = resolution
        self.nodata = nodata
        self.location = location
        self.training_data = []
        self.area = area
        self.dem_path = dem_path

    def process(self):
        trend = self.detrend_raster()
        detrended = self.image - trend
        detrended[self.image == self.nodata] = self.nodata

        self.training_data.append(self.image)
        self.training_data.append(detrended)
       
        self.training_data.extend([
            self.radar_sim(image)
            for image in self.training_data
        ])
        
        self.training_data.extend([self.augment(img) for img in self.training_data])

    def detrend_raster(self, sigma: int = 1000):
        sigma = sigma / self.resolution
        V = self.image.copy()
        V[self.image == self.nodata] = np.nan
        V[np.isnan(V)] = 0
        W = 0 * self.image.copy() + 1
        W[np.isnan(W)] = 0
        trend = gaussian_filter(V, sigma=sigma) + 1e-300
        WW = gaussian_filter(W, sigma=sigma) + 1e-300
        trend /= WW

        return trend

    def radar_sim(self, image, swath=100, ratio=2):
        swath = int(swath / self.resolution)
        skip = swath * ratio
        
        m, n = image.shape[0],image.shape[1]
 
        tile = np.zeros(image.shape)
        slices = []
        for i in range(0, m, skip):
            for j in range(0, n, skip):
                if bool(np.random.binomial(n=1, p=0.5, size=(1, 1))):
                    slice = np.s_[:, j : (j + swath)]
                else:
                    slice = np.s_[i : (i + swath), :]
                slices.append(slice)
        for slice in slices:
            tile[slice] = image[slice]
        return tile

    def augment(self, image):
        augmented = []
        augmented.append(image[:, ::-1])
        augmented.append(image[::-1, :])
        transposed = [image.T for image in augmented]
        augmented.extend(transposed)
        return augmented


class DEM:
    def __init__(
        self,
        dem: np.ndarray,
        labels: np.ndarray = None,
        resolution: int = 10,
        grid=500,
        area: str = None,
    ) -> None:
        self.dem_path = dem
        self.area = area
        self.dem = np.load(dem, allow_pickle=True)
        self.height, self.width = self.dem.shape[0],self.dem.shape[1]

        if labels is not None:
            self.labels = np.load(labels, allow_pickle=True)

        self.resolution = resolution
        self.grid = grid
        self.area = area

    def iterate(self):
        tiles = []
        counter = 0
        overall = time.time()
        bar = Bar(
            "Extracting Features",
            max=int(self.height / self.grid) * int(self.width / self.grid),
        )
        for i in range(0, self.height - self.grid, self.grid):
            for j in range(0, self.width - self.grid, self.grid):
                indi = int(i / self.grid)
                indj = int(j / self.grid)

                image = self.dem[i : i + self.grid, j : j + self.grid]
                tileargs = {
                    "location": np.s_[i : i + self.grid, j : j + self.grid],
                    "image": image,
                    "label": self.labels[indi, indj],
                    "area": self.area,
                    "dem_path": self.dem_path,
                }
                tile = Tile(**tileargs)
                tile = tile.process()
                tiles.append(tile)

                counter += 1
                bar.next()
        bar.finish()
        overall = time.time() - overall
        print(
            "\nIteration Done, Elapse Time: {} for {} Quadrats\n".format(
                (np.round(overall)), counter
            )
        )
        return tiles

    def run(self, path, load=False):
        if not os.path.isfile(path):
            self.tiles = self.iterate()
            np.save(path, self.tiles)
        elif os.path.isfile(path) and load:
            self.tiles = np.load(path, allow_pickle=True)
