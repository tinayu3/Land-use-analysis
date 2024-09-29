import os
from osgeo import gdal
 
#Input TIF file path
input_file =  r"J:\modis_china_2000_landcover.tif"
 
#Output binary file path
output_file = "output2000.bin"
 
#Open input file
dataset = gdal.Open(input_file)
 
if dataset is not None:
    #Get the width and height of the grid data
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    
    #Get data type
    data_type = dataset.GetRasterBand(1).DataType
 
    #Create output binary file
    with open(output_file, "wb") as f:
        #Go through each pixel from the bottom left to the top right
        for row in range(height - 1, -1, -1):
            for col in range(width):
                for band in range(dataset.RasterCount):
                    #Read the current pixel value
                    band_data = dataset.GetRasterBand(band + 1).ReadRaster(col, row, 1, 1)
                    
                    #Write pixel values ​​to a file
                    f.write(band_data)
                    
    #Close Dataset
    dataset = None
else:
    print("Failed to open input file.")