from re import I
import numpy as np
import multiprocessing as mp




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
def to_numpy_array(mp_array):
    #mp_array is a shared memory array with lock
    return np.frombuffer(mp_array.get_obj(),dtype=np.uint8)

def image_filter( image: numpy array,
filter_mask: numpy array 2D,
 numprocessors: int
 filtered_image: multiprocessing.Array)

    # Copy in the shared space the original image, in parallel.
    rows=range(image.shape[0])
    
    with mp.Pool(processes=numprocessors,initializer=pool_init,initargs=[filtered_image,image,filter_mask]) as p:
        p.map(f1,rows) # We are using the Pool object for parallel execution across multiple inputs
    
def f1(row):
    global image
    global my_filter
    global shared_space

    with shared_space.get_lock():
        (rows,cols,depth) = image.shape
        #fetch the r row from the original image
        
        srow=image[row,:,:]
        if ( row>0 ):
            prow=image[row-1,:,:]
        else:
            prow=image[row,:,:]
    
        if ( row == (rows-1)):
            nrow=image[row,:,:]
        else:
            nrow=image[row+1,:,:]

        #defines the result vector, and set the initial value to 0
        frow=np.zeros((cols,depth))

        # iterate through the columns
        for c in range(cols):
            # in the middle of the matrix
            if ( c>0 ):
                pcol= c-1 # prev row
            else:
                # at the left border of the matrix
                pcol=c
            
            if ( c == (cols-1)):
                # at the right border of the matrix
                ncol=c # next
            else:
                # middle
                ncol=c+1

            for d in range(depth):
                # apply the formula from slides for each depth
                frow[c, d] = srow[c,d] * my_filter[1,1] + srow[pcol,d] * my_filter[0,1] + srow[ncol,d] * my_filter[2,1] + (prow[c,d] * my_filter[0,1] +prow[pcol,d] * my_filter[0,0] +prow[ncol,d] * my_filter[0,2] ) + (nrow[c,d] * my_filter[2,1] + nrow[pcol,d] * my_filter[2,0] + nrow[ncol,d] * my_filter[2,2] )
    
            #while we are in this code block no one, except this execution thread, can write in the shared memory
            shared_matrix[row,:,:]=frow
    return