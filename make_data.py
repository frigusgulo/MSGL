import numpy as np
from yaml import load
from src.dem import DEM

dubawnt_dem = "./data/Dubawnt/MSGL_Large.npy"
dubawnt_labels = "./data/Dubawnt/MSGL_Large_Labels.npy"

brooks_dem = "./data/Brooks/brooks_dem.npy"
brooks_labels = "./data/Brooks/Brooks_labels.npy"

margold_dem = "./data/Margold/margold_dem.npy"

testset_dem = "./data/TestSet/MSGL_testdata.npy"
testset_labels = "./data/TestSet/Testset_Labels.npy"

thwaites_dem = "./data/Thwaites/thwaitesdem.npy"

dubawnt = DEM(dem=dubawnt_dem,labels=dubawnt_labels,area="Dubawnt")
dubawnt.run("./data/Training/dubawnt_traing.pkl",load=True)

brooks = DEM(dem=brooks_dem,labels=brooks_labels,area="Brooks")
brooks.run("./data/Training/brooks_training.pkl",load=True)

testset = DEM(dem=testset_dem,labels=testset_labels,area="Testset")
testset.run("./data/Training/testset_data.pkl",load=True)
