# FINAL PROJECT - The Giant Insta Filter
## Jared Lambert

# Description
Given a scientific dataset, compute a stencil computation with:
* Shared memory version
* MPI version using blocked domain decomposition and ghost zones sharing
* GPU implementation
* Scaling study

## shared memory version
should be simple. find parts that can be parallelized with threads. computation is trivial.
## distributed memory implementation
* domain decomposition (MPI file I/O)
* Ghost regions exchange (with and without halos regions preloaded
* computation is identical to shared memory
* generate output image (MPI file I/O)
## single gpu implementation using a convolution kernel



