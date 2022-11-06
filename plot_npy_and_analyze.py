import numpy as np
from matplotlib import pyplot as plt
from scipy.stats  import linregress
from mpl_toolkits.basemap import Basemap
resolution=0.25
LAT_GRID=np.linspace(0,60-resolution,int(60/resolution))
LON_GRID=np.linspace(70,140-resolution,int(70/resolution))
def plot_loc_trend(XCO2,filename,year_index):
    plt.close("all")
    print(year_index)
    print(XCO2[:,170,130])
    fig, axes = plt.subplots(figsize=(8,4))
    axes.plot(year_index,XCO2[:,170,130],label=str(LAT_GRID[170])+r"$N \degree$ "+str(LON_GRID[130])+r"$N \degree$" )
    axes.plot(year_index,XCO2[:,60,150],label=str(LAT_GRID[60])+r"$N \degree$ "+str(LON_GRID[150])+r'$N \degree$' )
    axes.plot(year_index,XCO2[:,100,140],label=str(LAT_GRID[100])+r"$N \degree$ "+str(LON_GRID[140])+r'$N \degree$' )
    from matplotlib.ticker import MaxNLocator
    axes.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes.set_xlabel("Year")
    axes.set_ylabel("$XCO_2/ppmv$")
    axes.legend()
    plt.savefig(filename,dpi=600,bbox_inches='tight',pad_inches=0.0)

def plot_basemap(value,geo_range,title,save_name,min_v=None,max_v=None):
    plt.close("all")
    GEO_RANGE=geo_range
    mp=Basemap(llcrnrlon=GEO_RANGE[0],llcrnrlat=GEO_RANGE[2],urcrnrlon=GEO_RANGE[1],urcrnrlat=GEO_RANGE[3],projection='cyl',resolution='h')
    mp.drawcoastlines(color='#AAAAAA',linewidth=0.5)
    mp.readshapefile("/home/mqzhang/data/china-shapefiles-master/china_country", 'china', drawbounds=True,color='#ff0000',linewidth=1)
    mp.readshapefile("/home/mqzhang/data/china-shapefiles-master/china",  'china', drawbounds=True,color='#000000',linewidth=0.2)
    mp.readshapefile("/home/mqzhang/data/china-shapefiles-master/china_nine_dotted_line",  'nine_dotted', drawbounds=True,color='#ff0000',linewidth=1)
    x0, y0 = mp(GEO_RANGE[0], GEO_RANGE[1])
    x1, y1 = mp(GEO_RANGE[2], GEO_RANGE[3])
    min_v=np.nanmean(value)-3*np.nanstd(value)
    max_v=np.nanmean(value)+3*np.nanstd(value)
    image=mp.imshow(value,extent=(x1,y1,x0,y0), interpolation='None',origin ='lower',cmap='jet',vmin=min_v,vmax=max_v)
    #image=mp.imshow(xco2_grid,extent=(x1,y1,x0,y0), interpolation='gaussian',origin ='upper',cmap='jet',vmin=390,vmax=400)
    #image.set_clip_path(clip)
    plt.title(title)
    mp.colorbar() 
    plt.savefig(save_name,bbox_inches='tight',dpi=600,pad_inches=0.0)
    plt.close('all')


lsm=np.load("Land_Sea_Mask.npy")
lsm=np.where(lsm<0.01)

filename="model_v52_predict_full_Test.npy"
XCO2=np.load(filename)
print(XCO2.shape)
XCO2[:,:4,:]=np.nan
XCO2[:,:,:4]=np.nan
#XCO2[:,lsm[0],lsm[1]]=np.nan
print(np.nanmin(XCO2),np.nanmax(XCO2))
plot_loc_trend(XCO2,"Test_loc_101.png",(np.arange(XCO2.shape[0])-1)/12+2003)
GEO_RANGE=[70,140-0.25,0,60-0.25]
plot_basemap(XCO2[190,:,:],GEO_RANGE,title="190_Index-XCO2-Map",save_name="190Index.png")
plot_basemap(np.nanmean(XCO2[0:12,:,:],axis=(0)),GEO_RANGE,title="Test_Part_AVE_F1-XCO2-Map",save_name="Test_Part_ave_F1.png")
plot_basemap(np.nanmean(XCO2[:,:,:],axis=(0)),GEO_RANGE,title="Test_Part_AVE-XCO2-Map",save_name="Test_Part_ave.png")
plot_basemap((XCO2[1,:,:]),GEO_RANGE,title="Test_Part_2013-XCO2-Map",save_name="Test_Part.png")
plot_basemap(XCO2[6,:,:],GEO_RANGE,title="2003-06-XCO2-Map",save_name="06.png")
plot_basemap(XCO2[11,:,:],GEO_RANGE,title="2003-12-XCO2-Map",save_name="12.png")
#plot_basemap(XCO2[169,:,:],GEO_RANGE,title="2017-02-XCO2-Map",save_name="201702.png")
XCO2[np.where(XCO2>500)]=np.nan
Trend=np.full((XCO2.shape[1],XCO2.shape[2]),np.nan)
year_index=np.arange(XCO2.shape[0])[:]/12
for i in range(5,Trend.shape[0]-5):
    for j in range(5,Trend.shape[1]-5):
        xco2_temp=XCO2[:,i,j]
        xco2_valid_index=np.where(~np.isnan(xco2_temp))
        if xco2_valid_index[0].size==0:
            Trend[i,j]=np.nan
        else:
            a=linregress(year_index[xco2_valid_index],xco2_temp[xco2_valid_index])
            Trend[i,j]=a.slope
print(np.nanmean(Trend))
print(np.nanstd(Trend))
plt.imshow(Trend,cmap='RdBu_r',origin='lower')
plt.colorbar()
plt.savefig("Test_Trend_M42.png",dpi=300)
