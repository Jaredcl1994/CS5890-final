# FINAL PROJECT - The Giant Insta Filter
## Jared Lambert

## Description
Given a scientific dataset, compute a stencil computation with:
* Shared memory version
* MPI version using blocked domain decomposition and ghost zones sharing
* GPU implementation
* Scaling study

## My Implementations

I did not complete the distributed code. It took me a long time to get the serial code and gpu code working, and then I only had time for the shared code besides that. Included in this folder are the serial code, the shared memory code, and the cuda code.
I used OMP for the shared memory version in omp_final.c. The gpu code is in cuda_final.cu. My start on the distributed code is in mpi_final.c.

The serial code is contained in each of the files and was used for comparison.  

### OMP Shared version
To implement the shared version I put an OMP parallel for loop declaration on all of the loops of the serial version.
All of the buffers where shared, while all of the counting variables, etc. were private.   

to run the omp code (computes using serial version and omp version, outputs to finalImage.raw), use:  

`gcc -o final omp_final.c -lm -fopenmp
./final`

### CUDA version
I didn't implement tiling in this code as I was strapped for time. I use a default block size of 16 for the kernel functions.
Besides removing the for loops and recalculating the index for the thread I was on, I also had to copy memory to and from the device.
I ran into a problem when trying to record the maximum value of any pixel in the image for one of the Canny processing steps.
In order to find a correct maximum, I used a cuda function that takes a device variable address and a value, then stores the maximum in the
device variable address.  

For some reason, my outputs were never exactly the same as the serial version. For many of the pixels, the values were different from the original by on one of the steps. Most of the pixels were correct, but many of them had a value off by 1 or 2, never more. I was unable to find the reason why, despite thorough investigation and experimentation.

To run the cuda code (computes using serial version and cuda version, outputs to finalImage.raw), use:

`srun nvcc ./cuda_final.cu -o final
srun ./final`

These programs each compute the serial code alongside their parallel code.

### Distributed version

I planned to use MPI in the distributed versions of my code. In my implementation I would have used MPI_SEND to distributed data to all of the threads. I think this would be easier than MPI_SCATTER because you need access to blocks of data sent to the threads around you as well. Using this data distribution technique, along with MPI_RECEIVE for the main thread, of course, most of the rest of the code would have stayed the same.

I regret that I didn't have more time to finish this section.

## Visualizations

My processed images are found in serial.raw, omp.raw, and cuda.raw.

## Scaling Study

The image associated with my scaling study can be found in timing.png. This contains the execution times for
processing our original image 1, 2, 3, 4, and 5 times. I was not able to go to any more iterations because I
was using too much of the hpc resources.

I attempted to time the cuda code using several different methods. The first method I tried was using time.h
to record start and stop times. I kept getting 0 for the execution time with this method. I then tried to use
cuda specific functions like cudaDeviceSynchronize to force the cpu to wait for the device to finish.
Still I got 0 for the execution times. Then I implemented a cuda utility library cutil.h, which had some timing functions
included in it. Despite many attempts, I coudn't get this to work either. I tried one other method that escapes
me at the moment.

So, I timed it using my phone. It is not very accurate, but it's the solution I came up with after many frustrated attempts. This by hand timing method includes the time it takes to read the buffer in and allocate the memory, while the others do not include these actions.

### Results

OMP was much slower than the serial version. The serial version and the shared version both increased linearly from 1-5
iterations, but OMP was increasing by 30s per iteration while serial was increasing by about 5. At first I thought that
this was because of memory allocation, so I made sure that memory was allocated outside of the timing. I found that
this did not increase performance. My hypothesis as to why OMP was much slower than the serial is because
I separated the algorithm into several different functions. Each function had a #pragma omp parallel' in it.
I think that the separate calls to allocate threads mixed with the serial operations like calling the
next function caused the slowdown.  

The gpu code, on the other hand, ran much faster than the serial code. It was the fastest of my implementations. That makes sense because my calculations were small but largely parallel; perfect for gpus. The first 4 experiments were around 4 seconds, which makes me think that the overhead was around 4 seconds. The acutal algorithm was much faster than that.
