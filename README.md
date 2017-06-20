# HYPPO: A Hybrid, Piecewise Polynomial Modeling Technique for Non-Smooth Surfaces

## Using the source code

**Note** The code uses Python 2.7

The modeling code, `build_model.py` requires as input a .csv file.
The format of the csv file must be:

> x\_(1,1), x\_(1,2), x\_(1,3), ..., x\_(1,n), z\_1
>
> x\_(2,1), x\_(2,2), x\_(2,3), ..., x\_(2,n), z\_2
>
> ...
>
> x\_(m,1), x\_(m,2), x\_(m,3), ..., x\_(m,n), z\_m

x\_(i,j) is the value of the j'th variable in the i'th observation.
z\_i is the corresponding dependent variable.  A sample data file is included called `WordCount_times.txt`
This file contains the run time of Word Count (using Spark) varying two framework parameters
(the x and y values in the file).

The `build_model.py` code accepts the .csv file of data and creates an n-dimensional grid of predictions.
The usage is:

`./build_model.py FileName --model HYPPO --K --ranges xmin xmax xstep ymin ymax ystep ...`

For each independent variable, the code requires 3 parameters: min, max, and step.
In the code, ranges is optional and contains default ranges suitable for the provided data.
However, if the model you construct has a number of independent variables other than 2 you MUST
set this to fit your data.  The model predicts the dependent variable for each point in the
cartesian product of the lists `range(min, max, step).`  K is the number of nearest neighbors
used to build the model--either KNN or HYPPO.
To run the code on the sample dataset:

`./build_model.py WordCount_times.txt --model HYPPO --K 7 --ranges 1 60 1 100 1500 100 >> model.txt`

This stores the output of `build_model.py` in a text file called `model.txt`
The file `model.txt` is in the correct format for use with the other piece of provided code: `plot_3d_model.py`

The usage for this code is simply:

`./plot_3D_model.py model.txt`

Typing:

`./build_model.py --help`

Will display additional help information for the code.

**Note** This code can be used to compute KNN regression (fixing polynomial degree to be 0) by
using the `--model KNN` flag.  It can also be used to compute traditional Surrogate Based Models
(SBM) where the degree of the polynomial is flexible, but K=N-1 (1 less than the number of data
points available).  This can be accomplished with the `--model SBM` flag.
