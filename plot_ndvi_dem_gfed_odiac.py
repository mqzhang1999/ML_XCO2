# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 12:39:27 2021

@author: mqzhang1999
"""
#This file is to plot four pic
#EVI ODIAC GFED DEM
#Read 2010 year
import numpy as np
from matplotlib import pyplot as plt
from netCDF4 import Dataset
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.io.img_tiles as cimgt
from cartopy.feature import ShapelyFeature
from cartopy.io.shapereader import Reader
from cartopy.mpl.patch import geos_to_path
import shapefile
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.ticker as mticker
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1 import make_axes_locatable
import glob

def add_china_shp(ax):
    fname=r"/data/mqzhang/china-shapefiles-master/china_nine_dotted_line.shp"
    shape_feature = ShapelyFeature(Reader(fname).geometries(),ccrs.PlateCarree(), edgecolor='black') 
    ax.add_feature(shape_feature,facecolor='none',edgecolor='black',linewidth=0.7)
    fname=r"/data/mqzhang/china-shapefiles-master/china.shp"
    shape_feature = ShapelyFeature(Reader(fname).geometries(),ccrs.PlateCarree(), edgecolor='black')
    ax.add_feature(shape_feature,facecolor='none',edgecolor='black',linewidth=0.2)
    fname=r"/data/mqzhang/china-shapefiles-master/china_country.shp"
    shape_feature = ShapelyFeature(Reader(fname).geometries(),ccrs.PlateCarree(), edgecolor='black')
    ax.add_feature(shape_feature,facecolor='none',edgecolor='black',linewidth=0.7)

def plot_sigle(filename,title,lat,lon,Value,cmap,MIN,MAX):
    proj = ccrs.AlbersEqualArea(central_latitude = 30, 
                             central_longitude = 105, 
                             standard_parallels = (25, 25))
    f,ax = plt.subplots(1, 1, subplot_kw={'projection': proj})
    gl = ax.gridlines(draw_labels=True)
    
    ax.set_extent([78, 130, 15, 55])
    gl.top_labels = gl.right_labels = False
    gl.xlabels_top = False  # 关闭顶端的经纬度标签
    gl.ylabels_right = False  # 关闭右侧的经纬度标签
    gl.xformatter = LONGITUDE_FORMATTER  # x轴设为经度的格式
    gl.yformatter = LATITUDE_FORMATTER  # y轴设为纬度的格式
    gl.xlocator = mticker.FixedLocator(np.arange(70, 130, 10))
    gl.ylocator = mticker.FixedLocator(np.arange(10,55, 10))
    gl.rotate_labels = False
    ax.coastlines(resolution='10m', color='black', linewidth=0.5)
    ac=ax.pcolormesh(lon,lat,Value,cmap=cmap,vmin=MIN,vmax=MAX,transform=ccrs.PlateCarree())
    add_china_shp(ax)
    divider = make_axes_locatable(ax)
    ax_cb = divider.new_vertical(size="5%", pad=0.2,pack_start=True, axes_class=plt.Axes)
    f.add_axes(ax_cb)
    cb=plt.colorbar(ac,orientation="horizontal",cax=ax_cb)
    #cb.set_label(clabel)
    ax.set_title(title)
    plt.savefig(filename+".png",dpi=400,bbox_inches='tight',pad_inches=0.0)
    

def read_evi():
    filelist=glob.glob("/home/mqzhang/data/aux_data/MYD13C2/NC/MYD13C2.A2010*.nc")
    d0=Dataset(filelist[0],'r')
    lon=d0["longitude"][:]
    lat=d0["latitude"][:]
    EVI=[]
    for f in filelist:
        dtemp=Dataset(f,"r")
        EVITemp=dtemp["CMG_0_05_Deg_Monthly_EVI"][:]
        EVITemp[np.where(EVITemp<-100)]=np.nan
        EVI.append(EVITemp)
    EVI=np.array(EVI)
    EVI=np.nanmean(EVI,axis=0)
    return lon,lat,EVI

def read_odiac():
    filelist=glob.glob("/home/mqzhang/DATA_STORE/odiac/NC/odiac2020b_1km_excl_intl_10*.nc")
    d0=Dataset(filelist[0],'r')
    lon=d0["lon"][:]
    lat=d0["lat"][:]
    Emis=[]
    for f in filelist:
        dtemp=Dataset(f,"r")
        EmisTemp=dtemp["Emission"][:]
        Emis.append(EmisTemp)
    Emis=np.array(Emis)
    Emis=np.nanmean(Emis,axis=0)
    return lon,lat,Emis

def read_dem():
    filename="srtm30plus_v11_land.nc"
    filepath=r'/home/mqzhang/DATA_STORE/AUX_DATA/'
    d=Dataset(filepath+filename,'r')
    Elev=d["elev"][:]
    lat=d["lat"][:]
    lon=d["lon"][:]
    return lon,lat,Elev
def read_gpw():
    filename=""
def read_gfed():
    filepath="/home/mqzhang/DATA_STORE/GFED4/GFED4/"
    filename="GFED4.1s_2010.hdf5"
    d=Dataset(filepath+filename,"r")
    lat=d["lat"][:,0]
    lon=d["lon"][0,:]
    GFED=[]
    for month in range(1,13):
        TempGFED=d['emissions/'+str(month).zfill(2)+'/C'][:]
        print(TempGFED.shape)
        GFED.append(TempGFED)
    GFED=np.array(GFED)
    print("Test GFED",GFED.shape)
    GFED=np.nanmean(GFED,axis=0)
    print(GFED.shape,"Test after mean")
    return lon,lat,GFED

from matplotlib.colors import LinearSegmentedColormap
white_jet = LinearSegmentedColormap.from_list('white_jet', [
    (0, '#ffffff'),
    (0.1, '#00007f'),
    (0.4, '#004dff'),
    (0.6, '#29ffce'),
    (0.8, '#ceff29'),
    (0.95, '#ff6800'),
    (1, '#7f0000'),
    ], N=256)



if __name__=="__main__":
    '''
    plt.close('all')
    lon,lat,EVI=read_evi()
    lon_grid,lat_grid = np.meshgrid(lon,lat)
    plot_sigle("EVI_2010","EVI",lat_grid,lon_grid,EVI,cmap="YlGn",MIN=0,MAX=0.7)
    '''
    plt.close("all")
    #lon,lat,Emis=read_odiac()
    d=Dataset("/home/mqzhang/DATA_STORE/odiac/ODIAC_DEMO.nc",'r')
    lon=np.array(d["lon"][:])
    lat=np.array(d["lat"][:])
    Emis=np.array(d["Emission"][:])
    lon_grid,lat_grid = np.meshgrid(lon,lat)
    #MAXV=np.nanmean(Emis)+3*np.nanstd(Emis)
    plot_sigle("Anth_Emis_2010","Fossil Fuel $CO_2$ Emission",lat_grid,lon_grid,Emis,cmap=white_jet,MIN=0,MAX=10)
    plt.close("all")
    
    #lon,lat,Elev=read_dem()
    #lon_grid,lat_grid=np.meshgrid(lon,lat)
    #plot_sigle("Dem_China_Region","Terrain Altitude",lat_grid,lon_grid,Elev,cmap='terrain',MIN=0,MAX=6000)
    #plt.close("all")
    #lon,lat,GFED=read_gfed()
    #lon_grid,lat_grid=np.meshgrid(lon,lat)
    #print(lon_grid.shape,"grid shape")
    #print(GFED.shape,"GFED shape")
    #plot_sigle("GFED_FireEmis_China_Region","GFED Fire Carbon Emission",lat_grid,lon_grid,GFED,cmap=white_jet,MIN=0,MAX=10,clabel="$gC/m^2/month$")
