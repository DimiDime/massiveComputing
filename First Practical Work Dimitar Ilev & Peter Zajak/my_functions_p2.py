from re import I
import numpy as np
import multiprocessing as mp
import math
from multiprocessing import Process, Lock

# !!!
# Identical lines of code have already been commented in my_functions_p
# !!!

#This function initialize the global shared memory data

def pool_init(shared_array_,srcimg, imgfilter):
    #shared_array_: is the shared read/write data, with lock. It is a vector (because the shared memory should be allocated as a vector
    #srcimg: is the original image
    #imgfilter is the filter which will be applied to the image and stor the results in the shared memory array
    
    #We defines the local process memory reference for shared memory space
    global shared_space
    #Here we define the numpy matrix handler
    global shared_matrix
    
    #Here, we will define the readonly memory data as global (the scope of this global variables is the local module)
    global image
    global my_filter
    
    #here, we initialize the global read only memory data
    image=srcimg
    my_filter=imgfilter
    size = image.shape
    
    #Assign the shared memory  to the local reference
    shared_space = shared_array_
    #Defines the numpy matrix reference to handle data, which will uses the shared memory buffer
    shared_matrix = tonumpyarray(shared_space).reshape(size)


#This functions just create a numpy array structure of type unsigned int8, with the memory used by our global r/w shared memory
def tonumpyarray(mp_arr):
    #mp_array is a shared memory array with lock 
    return np.frombuffer(mp_arr.get_obj(),dtype=np.uint8)

# Int this function we apply the parallel filter execution on the image/s
def start_fs(image, filter_mask, filter_mask2, nump, fi1, fi2): 
    
    # Define the parallel process with the targeted filter mask and image
    p1 = mp.Process(target=image_filter, args=(image,filter_mask,nump,fi1)) 
    p2 = mp.Process(target=image_filter, args=(image,filter_mask2,nump,fi2)) 
  
    # Start the processes 
    p1.start() 
    p2.start() 
  
    # Wait until processes are finished 
    p1.join() 
    p2.join() 

def image_filter( image,#: numpy array,
filter_mask, #: numpy array 2D,
 numprocessors, #: int
 filtered_image ): #: multiprocessing.Array) 
    rows=range(image.shape[0])
    with mp.Pool(processes=numprocessors,initializer=pool_init,initargs=[filtered_image,image,filter_mask]) as p:
        e = p.map(f1,rows)
    return e

# Function for the application of the filter
def f1(row):
    global image
    global my_filter
    global shared_space
    
    with shared_space.get_lock():
        (rows,cols,depth) = image.shape
        try:
            (frows,fcols) = my_filter.shape
        except Exception as e:
            return e
        
        #fetch the r row from the original image
        rowss=[]
        #nrows=[]
        #srow=image[row,:,:]
        halo = math.floor(frows/2)

        for i in range(halo,0,-1):
            if (row-i>=0 ):
                rowss.append(image[row-i,:,:])
            else:
                rowss.append(image[row,:,:])
        rowss.append(image[row,:,:])
        for ii in range(halo,0,-1):
            if ( row+ii >= (rows) ):
                rowss.append(image[row,:,:])
            else:
                rowss.append(image[row+ii,:,:])
        
        #defines the result vector, and set the initial value to 0
        frow=np.zeros((cols,depth))
        halo2=math.floor(fcols/2)
        
        # iterate through the columns
        ko = 0
        for c in range(cols-1):
            
            colss=[]
            # in the middle of the matrix
            for j in range(halo2,0,-1):
                if ( c-j>=0 ):
                    colss.append( c-j ) # prev row
                else:
                    # at the left border of the matrix
                    colss.append(c)
        
            colss.append(c)
            for jj in range(halo2,0,-1):
                if ( c == (cols-jj)):
                    # at the right border of the matrix
                    colss.append(c) # next
                else:
                    # middle
                    colss.append(c+jj)
           
            for d in range(depth):
                # Define variables fr and fc
                fr=0
                fc=0
                for co in colss: # iterate through columns
                    for rr in rowss: # iterate through rows
                        frow[c,d] += rr[co,d] * my_filter[fr,fc]
                        fr+=1
               
                    fc+=1
                    fr=0
            
            #while we are in this code block no one, except this execution thread, can write in the shared memory
            shared_matrix[row,:,:]=frow

    return 0