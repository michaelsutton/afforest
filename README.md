Afforest Connected Components Algorithm
===================

Overview
----------------
This repository contains the implementation of a new parallel shared-memory Connected-Components algorithm, named Afforest. The algorithm uses *subgraph sampling* for a fast approximation phase, after which a *component skip* reduces the number of edges processed to be sub-linear.    

The implementation relies on the GAP benchmark [code](https://github.com/sbeamer/gapbs) for graph-related infrastructure. The actual implementation is under src/cc_afforest.cc. A CUDA-based GPU implementation can be found under the device/ directory.

Build & Run
----------------
Build:

    $ make cc_afforest

Run Afforest on a Kronecker graph with 2^10 vertices and average degree 4 for 1 iteration and verify the results:

    $ ./cc_afforest -g10 -k4 -n1 -v

Additional command line flags can be found with `-h`

To run the GPU version of Afforest, use:

    $ make cuda
    $ ./cc_cuda -g10 -k4 -n1 -v
Note that the cub submodule is required for this version.

For a full guideline and to execute full benchmark runs, follow the instructions under [GAP](https://github.com/sbeamer/gapbs).  

The code can be compiled on Windows as well -- see under win/apps for a Visual Studio solution file. 

Publication
----------------
Please cite this implementation by our publication:
Michael Sutton, Tal Ben-Nun, Amnon Barak: "Optimizing Parallel Graph Connectivity Computation via Subgraph Sampling", Symposium on Parallel and Distributed Processing, IPDPS 2018.

