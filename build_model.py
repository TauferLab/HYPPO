#!/usr/bin/python

import argparse
import itertools
import numpy as np
import random

### This function expects a list of coefficients for the polynomial in order: 
### This function expects the degree of the polynomial (integer)
### This function expects a point (list of floats) of where to evaluate the polynomial.
### This function returns the value of the polynomial evaluated at the point provided.
def evaluate_polynomial(coefficients, degree, point):
	if degree == 0:
		return coefficients[0]
	
	monomials = [reduce(lambda a, b: a*b, x) for x in itertools.combinations_with_replacement([1.0] + point, degree)]
	return sum( map(lambda a: a[0]*a[1], zip(coefficients, monomials)) )


### independent_variable_points is a list of settings for the independent variables that were observed.
### dependent_variable_values is a list of observed values of the dependent variable.
### It is important that for each i the result of independent_variable_points[i] is stored as dependent_variable_values[i].
### degree is the degree of the polynomial to build.
### This function returns the list of coefficients of the best fit polynomial surface of degree "degree"
def determine_coefficients( independent_variable_points, dependent_variable_values, degree):
	X = []
	if degree > 0:
		for iv in independent_variable_points:
			X.append( [reduce(lambda a, b: a*b, x) for x in itertools.combinations_with_replacement([1.0] + iv, degree)] )
	else:
		X = [ [1] for iv in independent_variable_points ]
	X = np.array(X)
	Xt = np.transpose(X)
	Z = np.array(dependent_variable_values)
	coef = np.linalg.solve( np.dot(Xt, X), np.dot(Xt, Z) )
	return list( coef )


### data_points is a list of the observed independent variable settings
### specific_points is one chosen setting of the independent variables
### k is the number of nearest neighbors to find.
### scale indicates how the coordinates can/should be re-scaled before the distance metric is computed.
### For example, if the points are of the form (x, y) and x's are measured in 1's and y's are measured by 100's.
### Then, it may be desirable to multiply the x-values by 100 to bring them onto the same scale as the y-values.
### To do this, set scale=[100, 1].
### This function returns a list of indices (in data_points) of the k nearest neighbors.
### If specific_point is among the data sampled in data points (distance 0) it is NOT returned.
def indices_of_kNN( data_points, specific_point, k, scale):
	
	scale = np.array(scale)
	scaled_data = [ np.array(x)*scale for x in data_points ]
	specific_point = np.array(specific_point)*scale

	distances = [ sum( (x-specific_point)**2 ) for x in scaled_data ]
	indices = np.argsort( distances, kind='mergesort' )[:k+1]

	if distances[indices[0]] < .001:
		return indices[1:]
	else:
		return indices[:k]



def binom(n, r):
	if n < r:
		return 0
	retVal = 1
	for i in xrange(r):
		retVal = (retVal*(n-i))/(i+1)
	return retVal


### independent_data_points is a list of the observed independent variables to build models from.
### dependent_data_points is a list of the observed dependent variables (in the same order).
### k is the number of folds or partitions to divide the data into.
### num_random_partitions is the number of times the data is randomly partitioned (for averaging over many runs)
def kfold_crossvalidation( independent_data_points, dependent_data_points, k, num_random_partitions):
	
	n = len(independent_data_points)					### Number of data points
	dim = len(independent_data_points[0])				### Dimension of the data
	size_of_smallest_learning_set = int(n*(1.0-1.0/k))	### Used to constrain degree of polynomial
	possible_degrees = [d for d in xrange(9) if binom(d+dim, dim)<size_of_smallest_learning_set]
	
	fold_sizes = [ n/k ]	### Integer division rounds down.
	first_index = [ 0 ]		### Index of first element of the fold in the indices list (below)
	for i in xrange(1,k):
		fold_sizes.append( (n - sum(fold_sizes))/(k-i) )
		first_index.append( first_index[i-1] + fold_sizes[i-1] )
	
	Total_SSE = [ 0 for x in possible_degrees ]		### A list of 0's of same length as possible degrees
	
	
	for iteration in xrange(num_random_partitions):
		### Randomly partition the data into k sets as equally sized as possible
		indices = range(n)
		random.shuffle( indices )	### Get a new random shuffling of the indices
		Folds = [ indices[first_index[fold]:first_index[fold]+fold_sizes[fold]] for fold in xrange(k) ]
		
		for d in possible_degrees:
			### Build k models of degree d (each model reserves one set as testing set)
			for testing_fold in xrange(k):
				model_independent_data = []
				model_dependent_data = []
				testing_independent_data = [ independent_data_points[i] for i in Folds[ testing_fold ] ]
				testing_dependent_data = [ dependent_data_points[i] for i in Folds[testing_fold] ]
				for fold in xrange(k):
					if fold != testing_fold:
						model_independent_data = model_independent_data + [ independent_data_points[i] for i in Folds[fold] ]
						model_dependent_data = model_dependent_data + [ dependent_data_points[i] for i in Folds[fold] ]
				
				### Get the polynomial built from the model data of degree d
				try:
					coefficients = determine_coefficients( model_independent_data, model_dependent_data, d)
				
					### Predict the testing points and add the error to the Total_SSE[d]
					for x, z in zip(testing_independent_data, testing_dependent_data):
						Total_SSE[d] += (evaluate_polynomial(coefficients, d, x) - z)**2	### The square of the difference between the polynomial prediction at x and the observed value (z) at x.
				except:
					Total_SSE[d] += 99999999999		### Basically, this d was too big.

	### return index of minimum Total_SSE
	### Note: Total_SSE[i] corresponds to polynomial of degree i.
	return Total_SSE.index( min(Total_SSE) )



### Ideal for small sample sizes
def leave_one_out_crossvalidation(independent_data_points, dependent_data_points):
	return kfold_crossvalidation(independent_data_points, dependent_data_points, len(independent_data_points), 1)




def main():

	parser = argparse.ArgumentParser()
	parser.add_argument( "fileName", help="The path to the csv file containing the training data")
	
	parser.add_argument( "--model", help="The type of model to build.",
				choices=["HYPPO", "KNN", "SBM"], default="HYPPO")
	parser.add_argument( "--K", help="The number of nearest neighbors to use for either the KNN or HYPPO model",
				type=int, default=6)
	
	parser.add_argument("--ranges", help="One or more triples of integers in the form min, max, step",
				nargs='+', type=int, default=[1, 60, 1, 100, 1500, 100])
	args=parser.parse_args()

	fileName=args.fileName 	### Name of the file containing the data from which to build the model.
							### It is expected that the file be comma separated.
							### It is also expected that the format be x1, x2, ..., xm, z
							### Where x1,...,xm are the independent variables and z is the observed dependent variable.

	K = args.K				### The number of nearest neighbors to use when building the model.

	# xmin  	### The next values (in groups of 3) give min max and step (as would be inputs to xrange) to describe the region that will be modeled.
	# xstep		### E.g. if your space has 2 dependent variables, x and y, which each range from 1 to 10 (increments of 1) then the next 6 values
	# xmax	])	### would be 1 10 1 1 10 1.		
				### A variable number of these are allowed as long as they come in groups of 3.
	
	Common_Multiple = 1												
	ranges = []
	step = []
	for i in xrange(0, len(args.ranges), 3):
		ranges.append( xrange(args.ranges[i], args.ranges[i+1], args.ranges[i+2]) )
		Common_Multiple = Common_Multiple * args.ranges[i+2]
		step.append( args.ranges[i+2] )

	values_to_model = [list(x) for x in itertools.product( *ranges )]

	SCALE = [Common_Multiple/s for s in step]

	Independent_Data = []
	Dependent_Data = []

	with open(fileName, 'r') as inFile:
		for line in inFile:
			numbers=[float(x) for x in line.strip().split(",")]
			Independent_Data.append( numbers[:-1] )
			Dependent_Data.append( numbers[-1] )
	
	if args.model=="SBM":
		K = len(Dependent_Data) - 1

	degree = -1				### Not yet set.
	for x in values_to_model:
		### Find Nearest neighbors
		indices_of_nearest_neighbors = indices_of_kNN(Independent_Data, x, K, SCALE)

		### Select the data associated with the nearest neighbors for use with modeling
		selected_independent_data = [Independent_Data[i] for i in indices_of_nearest_neighbors]
		selected_dependent_data = [Dependent_Data[i] for i in indices_of_nearest_neighbors]

		if args.model=="HYPPO":
			### Determine the best polynomial degree
			degree = leave_one_out_crossvalidation(selected_independent_data, selected_dependent_data)
		
		elif args.model=="KNN":
			### Setting the degree to 0 forces us to just average the nearest neighbors.
			### This is exactly kNN (a degree 0 polynomial).
			degree = 0
	
		else:
			assert args.model=="SBM"
			if degree == -1:
				degree = kfold_crossvalidation(selected_independent_data, selected_dependent_data, 10, 10)

		### Compute the coefficients of the "best" polynomial of degree degree
		coefficients = determine_coefficients(selected_independent_data, selected_dependent_data, degree)

		### Using the surface, predict the value of the point.
		z  = evaluate_polynomial(coefficients, degree, x)
	
		line_output = ""
		for i in x:
			line_output = line_output + "%d,"%i
		line_output = line_output + "%f"%z
		print line_output
	

main()
