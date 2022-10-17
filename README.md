# UCSB-ECE273
This is a repo for the homework of the course [ECE273](https://www.ccdc.ucsb.edu/course/ECE273) - *Tensor Computation for Machine Learning and Big Data*, Fall 2022 in UCSB.

As the course is continuing, more code will be pushed.
- [x] Problem 1: CP of a FC layer - see the `codehwp1.ipynb`
- [ ] Problem 2: Trucker of a conv filter
- [ ] Problem 3: Tensor-train of an embedding table

## Requriements
The code has been run under Ubuntu 18.04. It has been tested with the following dependencies
- Python 3.9
- numpy

## Code Architecture
```
ll
├── llBase.py
├── llCPTest.ipynb
├── llDecomposition
├── llTenalg
└── llTensor
```
- llBase.py - some basic tensor operations
- llDecomposition - whole pipelines doing several decomposition
- llTenalg - various tensor-specific algorithm
- llTensor - self-implemented data structure `Tensor` to replace np.array

## Credits
This code's architecture is based on [TensorLy](http://tensorly.org/stable/index.html)
