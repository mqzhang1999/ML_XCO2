#This File is Convert python  numpy npy file to netcdf
# @mqzhang
#python3 Test.npy BeginDate EndDate 
import numpy as np
from netCDF4 import Dataset
import glob
import datetime
import sys

#Format "200301"  "201504"  include the last
def creat_timesque_month(DateBegin,DateEnd):
    d1=datetime.datetime(2000,1,1)
    MonthNum=(int(DateEnd[0:4])-int(DateBegin[0:4]))*12+(int(DateEnd[4:6])-int(DateBegin[4:6]))+1
    BeginYear=int(DateBegin[0:4])
    BeginMonth=int(DateBegin[4:6])
    time_sque=[]
    for i in range(MonthNum):
        temp_year=BeginYear+int((i+BeginMonth-1)/12)
        temp_month=(BeginMonth+i-1)%12+1
        dtemp=datetime.datetime(temp_year,temp_month,15)
        interval=dtemp-d1
        time_sque.append(interval.days)
    return np.array(time_sque)


def creatnetcdf(lat,lon,var_list,var_name,var_unit,var_longname,time_sque,filename):
    ds = Dataset(filename, 'w', format='NETCDF4')
    time = ds.createDimension('time',time_sque.shape[0])
    LAT = ds.createDimension('latitude', lat.shape[0])
    LON = ds.createDimension('longitude', lon.shape[0])
    times = ds.createVariable('time', 'f4', ('time',))
    latitude = ds.createVariable('latitude', 'f4', ('latitude',))
    longitude = ds.createVariable('longitude', 'f4', ('longitude',))
    times.units="days since 2000-01-01 00:00:00 UTC"
    times.axis="T"
    times.long_name="time"
    times.cell_methods = "time: mean"

    latitude.units = "degrees_north"
    longitude.units ="degrees_east"
    latitude.standard_name="latitude"
    latitude.long_name="latitude"
    latitude.axis="Y"
    longitude.standard_name="longitude"
    longitude.long_name="longitude"
    longitude.axis="X"

    nc_var_list=[ds.createVariable(varname, 'f4', ('time', 'latitude', 'longitude',),zlib=True) for varname in var_name ]
    for index in range(len(nc_var_list)):
        print(var_longname[index])
        nc_var_list[index].units=var_unit[index]
        nc_var_list[index].long_name=var_longname[index]
        nc_var_list[index][:]=var_list[index]

    latitude[:]=lat
    longitude[:]=lon
    times[:]=time_sque    
    ds.close()

if __name__=="__main__":
    npyfilename="/home/mqzhang/DATA_STORE/model_file/W4_consider_distance/model_v6_predict_full_Test_delta_xco2.npy"
    data=np.load(npyfilename)
    DateBegin="200301"
    DateEnd="201912"
    #creatnetcdf(lat,lon,var_list,var_name,var_unit,var_longname,time_sque,filename)
    resolution=0.25
    LAT_GRID=np.linspace(0,60-resolution,int(60/resolution))
    LON_GRID=np.linspace(70,140-resolution,int(70/resolution))
    Time_Sque=creat_timesque_month(DateBegin,DateEnd)
    creatnetcdf(LAT_GRID,LON_GRID,[data],["xco2"],["ppmv"],[""],Time_Sque,"Model6_Final_XCO2.nc")
