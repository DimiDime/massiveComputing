#!/usr/bin/env python
# coding: utf-8

# # My Functions Library
# 
# Due a bug in the multiprocessing module implemented for Windows Operating System, the functions which will be executed in the parallel threads MUST be implemented in a separated file, and import them in the main programs.
# 
# In order to be loaded in you own program, you have to write your own functions here, and export to a python ".py" file, to be imported in  the main script.
# 
# To export to a python file, select in the *File*  menu, the option *Download as* and save as *Python .py* file.

# ## Functions needed for FirstParallel notebook

import numpy as np

def init_second(my_matrix2):
    global matrix_2
    matrix_2=my_matrix2
    print(matrix_2.shape)


# In[ ]:


def init_globalimage(img,filt):
    global image
    global my_filter
    image=img
    my_filter=filt


# In[ ]:


def parallel_matmul(v):
    # v: is the input row
    # matrix_2: is the second matrix, shared by memory
    
    #here we calculate the shape of the second matrix, to generate the resultant row
    matrix_2 # we will uses the global matrix
    
    (rows,columns)=matrix_2.shape
    
    #we allocate the final vector of size the number of columns of matrix_2
    d=np.zeros(columns)
    
    #we calculate the dot product between vector v and each column of matrix_2
    for i in range(columns):
        d[i] = np.dot(v,matrix_2[:,i])
    
    #returns the final vector d
    return d


# In[ ]:


def parallel_filtering_image(r):
    # r: is the image row to filter
    # image is the global memory array
    # my_filter is the filter shape to apply to the image
    
    global image
    global my_filter
    
    #from the global variaable, gets the image shape
    (rows,cols,depth) = image.shape

    #fetch the r row from the original image
    srow=image[r,:,:]
    if ( r>0 ):
        prow=image[r-1,:,:]
    else:
        prow=image[r,:,:]
    
    if ( r == (rows-1)):
        nrow=image[r,:,:]
    else:
        nrow=image[r+1,:,:]
        
    #defines the result vector, and set the initial value to 0
    frow=np.zeros((cols,depth))
    
    for c in range (cols): # Create a for loop to iterate through all the columns of the original image  
        if ( c>0 ): # Check if the current index is bigger than zero
            pcol = c-1 # Set pcol to the previous index
        else:
            pcol= c # otherwise set the pcol to the current index
        if ( c == (cols-1)): # if we have reached the last index of the image
            ncol= c # then next column becommes that index
        else:
            ncol= c+1 # otherwise next column becommes the following index
        
        for d in range(depth): # for loop to go over the 3 levels of depth of the image
            # Here we apply the filter over each and every position of the image
            # from index [0,0] to [2,2]
            frow[c,d] = srow[c,d] * my_filter[1,1] + prow[pcol, d] * my_filter[0,0] + prow[c, d] * my_filter[0,1]
            frow[c,d] += prow[ncol, d] * my_filter[0,2] + srow[pcol, d] * my_filter[1,0] + srow[ncol, d] * my_filter[1,2]
            frow[c,d] += nrow[pcol, d] * my_filter[2,0] + nrow[c, d] * my_filter[2,1] + nrow[ncol,d] * my_filter[2,2]
    
    #return the filtered row
    return frow

# In[ ]:


#This cell should be the last one
#this avoid the execution of this script when is invoked directly.
if __name__ == "__main__":
    print("This is not an executable library")

