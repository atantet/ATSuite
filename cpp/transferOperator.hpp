#ifndef TRANSFEROPERATOR_HPP
#define TRANSFEROPERATOR_HPP

#include <iostream>
#include <vector>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_vector_uint.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_matrix_uint.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <omp.h>
#include "atmath.hpp"

typedef Eigen::Triplet<double> triplet;
typedef std::vector<triplet> tripletVector;
typedef Eigen::Triplet<size_t> tripletUInt;
typedef std::vector<tripletUInt> tripletUIntVector;
typedef Eigen::SparseMatrix<double, Eigen::ColMajor> SpMatCSC;
typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SpMatCSR;


// Decalrations

void getTransitionMatrix(const gsl_matrix_uint *, const size_t,
			 SpMatCSR *, SpMatCSR *, gsl_vector *, gsl_vector *);
void getTransitionMatrix(const gsl_matrix *, const gsl_matrix *,
			 const std::vector<gsl_vector *> *,
			 SpMatCSR *, SpMatCSR *, gsl_vector *, gsl_vector *);
void getTransitionMatrix(const gsl_matrix *, const std::vector<gsl_vector *> *,
			 const size_t tauStep,
			 SpMatCSR *, SpMatCSR *, gsl_vector *, gsl_vector *);
gsl_matrix_uint *getGridMembership(const gsl_matrix *, const gsl_matrix *,
				  const std::vector<gsl_vector *> *);
gsl_matrix_uint *getGridMembership(const gsl_matrix *,
				  const std::vector<gsl_vector *> *,
				  const size_t);
gsl_vector_uint *getGridMembership(const gsl_matrix *,
				  const std::vector<gsl_vector *> *);
gsl_matrix_uint *getGridMembership(gsl_vector_uint *, const size_t);
int getBoxMembership(gsl_vector *, const std::vector<gsl_vector *> *);
void filterTransitionMatrix(SpMatCSR *, gsl_vector *, gsl_vector *, double);
std::vector<gsl_vector *> *getGridRect(size_t, size_t, double, double);
std::vector<gsl_vector *> *getGridRect(gsl_vector_uint *,
				       gsl_vector *, gsl_vector *);
void writeGridRect(FILE *, std::vector<gsl_vector *> *, bool);


// Definitions

// Get the forward and backward transition matrices from the membership matrix
void getTransitionMatrix(const gsl_matrix_uint *gridMem, const size_t N,
			 SpMatCSR *P, SpMatCSR *Q,
			 gsl_vector *initDist, gsl_vector *finalDist)
{
  size_t nOut = 0;
  const size_t nTraj = gridMem->size1;
  size_t box0, boxf;
  tripletUIntVector T;
  T.reserve(nTraj);

  // Get transition count triplets
  for (size_t traj = 0; traj < nTraj; traj++) {
    box0 = gsl_matrix_uint_get(gridMem, traj, 0);
    boxf = gsl_matrix_uint_get(gridMem, traj, 1);
    
    // Add transition triplet
    if ((box0 < N) && (boxf < N))
      T.push_back(tripletUInt(box0, boxf, 1));
    else
      nOut++;
  }
  std::cout <<  nOut * 100. / nTraj
	    << "% of the trajectories ended up out of the domain." << std::endl;
  
  // Get correlation matrix
  P->setFromTriplets(T.begin(), T.end());
  // Get initial and final distribution
  getRowSum(P, initDist);
  getColSum(P, finalDist);
  // Get forward and backward transition matrices
  *Q = SpMatCSR(P->transpose());
  toLeftStochastic(P);
  toLeftStochastic(Q);
  normalizeVector(initDist);
  normalizeVector(finalDist);

  return;
}


// Get transition matrix from initial and final states of trajectories
void getTransitionMatrix(const gsl_matrix *initStates,
			 const gsl_matrix *finalStates,
			 const std::vector<gsl_vector *> *gridBounds,
			 SpMatCSR *P, SpMatCSR *Q,
			 gsl_vector *initDist, gsl_vector *finalDist)
{
  const size_t nTraj = initStates->size1;
  const size_t dim = initStates->size2;
  size_t N = 1;
  size_t nOut = 0;
  size_t box0, boxf;
  gsl_vector *bounds;
  gsl_matrix_uint *gridMem;
  tripletUIntVector T;

  // Get grid dimensions
  for (size_t dir = 0; dir < dim; dir++){
    bounds = gridBounds->at(dir);
    N *= bounds->size - 1;
  }
  T.reserve(nTraj);
  P = new SpMatCSR(N, N);
  Q = new SpMatCSR(N, N);

  // Get grid membership
  gridMem = getGridMembership(initStates, finalStates, gridBounds);
  
  for (size_t traj = 0; traj < nTraj; traj++) {
    box0 = gsl_matrix_uint_get(gridMem, traj, 0);
    boxf = gsl_matrix_uint_get(gridMem, traj, 1);
    
    // Add transition to matrix
    if ((box0 < N) && (boxf < N))
      T.push_back(tripletUInt(box0, boxf, 1));
    else
      nOut++;
  }
  std::cout <<  nOut * 100. / nTraj
	    << " of the trajectories ended up out of the domain." << std::endl;
  
  // Get correlation matrix
  P->setFromTriplets(T.begin(), T.end());
  // Get initial and final distribution
  initDist = getRowSum(P);
  finalDist = getColSum(P);
  // Get forward and backward transition matrices
  *Q = SpMatCSR(P->transpose());
  toLeftStochastic(P);
  toLeftStochastic(Q);
  normalizeVector(initDist);
  normalizeVector(finalDist);
   
  return;
}


// Get membership matrix for initial and final states of trajectories
gsl_matrix_uint *getGridMembership(const gsl_matrix *initStates,
				  const gsl_matrix *finalStates,
				  const std::vector<gsl_vector *> *gridBounds)
{
  const size_t nTraj = initStates->size1;
  const size_t dim = initStates->size2;
  size_t N = 1;
  gsl_vector *bounds;
  gsl_matrix_uint *gridMem;

  // Get grid size
  for (size_t dir = 0; dir < dim; dir++){
    bounds = gridBounds->at(dir);
    N *= bounds->size - 1;
  }
  gridMem = gsl_matrix_uint_alloc(N, 2);
  
  // Assign a pair of source and destination boxes to each trajectory
#pragma omp parallel
  {
    gsl_vector *X = gsl_vector_alloc(dim);
    
#pragma omp for
    for (size_t traj = 0; traj < nTraj; traj++) {
      // Find initial box
      gsl_matrix_get_row(X, initStates, traj);
      gsl_matrix_uint_set(gridMem, traj, 0, getBoxMembership(X, gridBounds));
      
      // Find final box
      gsl_matrix_get_row(X, finalStates, traj);
      gsl_matrix_uint_set(gridMem, traj, 1, getBoxMembership(X, gridBounds));
    }
    gsl_vector_free(X);
  }
  
  return gridMem;
}


// Get transition matrix from single trajectory 
void getTransitionMatrix(const gsl_matrix *states,
			 const std::vector<gsl_vector *> *gridBounds,
			 const size_t tauStep,
			 SpMatCSR *P, SpMatCSR *Q,
			 gsl_vector *initDist, gsl_vector *finalDist)
{
  size_t nTraj;
  const size_t dim = states->size2;
  size_t N = 1;
  size_t nOut = 0;
  size_t box0, boxf;
  gsl_vector *bounds;
  gsl_matrix_uint *gridMem;
  tripletUIntVector T;
  P = new SpMatCSR(N, N);
  Q = new SpMatCSR(N, N);

  // Get grid dimensions
  for (size_t dir = 0; dir < dim; dir++){
    bounds = gridBounds->at(dir);
    N *= bounds->size - 1;
  }

  // Get grid membership
  gridMem = getGridMembership(states, gridBounds, tauStep);
  nTraj = gridMem->size1;
  T.reserve(nTraj);
  
  for (size_t traj = 0; traj < nTraj; traj++) {
    box0 = gsl_matrix_uint_get(gridMem, traj, 0);
    boxf = gsl_matrix_uint_get(gridMem, traj, 1);
    
    // Add transition to matrix
    if ((box0 < N) && (boxf < N))
      T.push_back(tripletUInt(box0, boxf, 1));
    else
      nOut++;
  }
  std::cout <<  nOut * 100. / nTraj
	    << " of the trajectories ended up out of the domain." << std::endl;
  
  // Get correlation matrix
  P->setFromTriplets(T.begin(), T.end());
  // Get initial and final distribution
  initDist = getRowSum(P);
  finalDist = getColSum(P);
  // Get forward and backward transition matrices
  *Q = SpMatCSR(P->transpose());
  toLeftStochastic(P);
  toLeftStochastic(Q);
  normalizeVector(initDist);
  normalizeVector(finalDist);
   
  return;
}


// Get grid membership from a single long trajectory
gsl_matrix_uint *getGridMembership(const gsl_matrix *states,
				  const std::vector<gsl_vector *> *gridBounds,
				  const size_t tauStep)
{
  const size_t nStates = states->size1;
  gsl_vector_uint *gridMemVect;
  gsl_matrix_uint *gridMem = gsl_matrix_uint_alloc(nStates - tauStep, 2);

  // Get membership vector
  gridMemVect = getGridMembership(states, gridBounds);

  // Get membership matrix from vector
  for (size_t traj = 0; traj < (nStates - tauStep); traj++) {
    gsl_matrix_uint_set(gridMem, traj, 0,
		       gsl_vector_uint_get(gridMemVect, traj));
    gsl_matrix_uint_set(gridMem, traj, 1,
		       gsl_vector_uint_get(gridMemVect, traj + tauStep));
  }

  // Free
  gsl_vector_uint_free(gridMemVect);
  
  return gridMem;
}


// Get grid membership from a single long trajectory
gsl_vector_uint *getGridMembership(const gsl_matrix *states,
				  const std::vector<gsl_vector *> *gridBounds)
{
  const size_t nStates = states->size1;
  const size_t dim = states->size2;
  gsl_vector_uint *gridMem = gsl_vector_uint_alloc(nStates);

  // Assign a pair of source and destination boxes to each trajectory
#pragma omp parallel
  {
    gsl_vector *X = gsl_vector_alloc(dim);
    
#pragma omp for
    for (size_t traj = 0; traj < nStates; traj++) {
      // Find initial box
      gsl_matrix_get_row(X, states, traj);
#pragma omp critical
      {
	gsl_vector_uint_set(gridMem, traj, getBoxMembership(X, gridBounds));
      }
    }
    gsl_vector_free(X);
  }
  
  return gridMem;
}


// Get grid membership matrix from membership vector for a given lag
gsl_matrix_uint *getGridMembership(gsl_vector_uint *gridMemVect,
				  const size_t tauStep)
{
  const size_t nStates = gridMemVect->size;
  gsl_matrix_uint *gridMem = gsl_matrix_uint_alloc(nStates - tauStep, 2);

  // Get membership matrix from vector
  for (size_t traj = 0; traj < (nStates - tauStep); traj++) {
    gsl_matrix_uint_set(gridMem, traj, 0,
		       gsl_vector_uint_get(gridMemVect, traj));
    gsl_matrix_uint_set(gridMem, traj, 1,
		       gsl_vector_uint_get(gridMemVect, traj + tauStep));
  }

  return gridMem;
}


// Get membership of one realization
int getBoxMembership(gsl_vector *X, const std::vector<gsl_vector *> *gridBounds)
{
  const size_t dim = X->size;
  size_t inBox, nBoxDir;
  size_t foundBox;
  size_t subbp, subbn, ids;
  gsl_vector *bounds;
  size_t N = 1;

  // Get grid dimensions
  for (size_t d = 0; d < dim; d++){
    bounds = gridBounds->at(d);
    N *= bounds->size - 1;
  }

  // Get box
  foundBox = N;
  for (size_t box = 0; box < N; box++){
    inBox = 0;
    subbp = box;
    for (size_t d = 0; d < dim; d++){
      bounds = gridBounds->at(d);
      nBoxDir = bounds->size - 1;
      subbn = (size_t) (subbp / nBoxDir);
      ids = subbp - subbn * nBoxDir;
      inBox += (size_t) ((gsl_vector_get(X, d)
			  >= gsl_vector_get(bounds, ids))
			 & (gsl_vector_get(X, d)
			    < gsl_vector_get(bounds, ids+1)));
      subbp = subbn;
    }
    if (inBox == dim){
      foundBox = box;
      break;
    }
  }
  
  return foundBox;
}


// Remove weak nodes
// norm = 0: normalize over all elements
// norm = 1: to right stochastic
// norm = 2: to left stochastic
// No normalization for any other choice
void filterTransitionMatrix(SpMatCSR *T,
			    gsl_vector *rowCut, gsl_vector *colCut,
			    double alpha, int norm)
{
  double valRow, valCol;
  
  for (int outerIdx = 0; outerIdx < T->outerSize(); outerIdx++){
    valRow = gsl_vector_get(rowCut, outerIdx);
    for (SpMatCSR::InnerIterator it(*T, outerIdx); it; ++it){
      valCol = gsl_vector_get(colCut, it.col());
      // Remove elements of states to be removed
      if ((valRow < alpha) || (valCol < alpha))
	it.valueRef() = 0.;
    }
  }
  //T->prune();
    
  // Normalize
  switch (norm){
  case 0:
    toAndStochastic(T);
    break;
  case 1:
    toRightStochastic(T);
    break;
  case 2:
    toLeftStochastic(T);
    break;
  default:
    break;
  }
  normalizeVector(rowCut);
  normalizeVector(colCut);

  return;
}


// Get rectangular grid
std::vector<gsl_vector *> *getGridRect(size_t dim, size_t nx,
				       double xmin, double xmax)
{
  double delta;
  std::vector<gsl_vector *> *gridBounds = new std::vector<gsl_vector *>(dim);

  for (size_t d = 0; d < dim; d++) {
    // Alloc one dimensional box boundaries vector
    (*gridBounds)[d] = gsl_vector_alloc(nx + 1);
    // Get spatial step
    delta = (xmax - xmin) / nx;
    gsl_vector_set((*gridBounds)[d], 0, xmin);
    
    for (size_t i = 1; i < nx + 1; i++)
      gsl_vector_set((*gridBounds)[d], i,
		     gsl_vector_get((*gridBounds)[d], i-1) + delta);
  }

  return gridBounds;
}


// Get rectangular grid
std::vector<gsl_vector *> *getGridRect(gsl_vector_uint *nx,
				       gsl_vector *xmin, gsl_vector *xmax)
{
  const size_t dim = nx->size;
  double delta;
  std::vector<gsl_vector *> *gridBounds = new std::vector<gsl_vector *>(dim);

  for (size_t d = 0; d < dim; d++) {
    // Alloc one dimensional box boundaries vector
    (*gridBounds)[d] = gsl_vector_alloc(gsl_vector_uint_get(nx, d) + 1);
    // Get spatial step
    delta = (gsl_vector_get(xmax, d) - gsl_vector_get(xmin, d))
      / gsl_vector_uint_get(nx, d);
    gsl_vector_set((*gridBounds)[d], 0, gsl_vector_get(xmin, d));
    
    for (size_t i = 1; i < gsl_vector_uint_get(nx, d) + 1; i++)
      gsl_vector_set((*gridBounds)[d], i,
		     gsl_vector_get((*gridBounds)[d], i-1) + delta);
  }

  return gridBounds;
}


// Write rectangular grid to file
void writeGridRect(FILE *gridFile, std::vector<gsl_vector *> *gridBounds,
		   bool verbose=false)
{
  gsl_vector *bounds;
  size_t dim = gridBounds->size();
  
  if (verbose)
    std::cout << "Domain grid (min, max, n):" << std::endl;
  
  for (size_t d = 0; d < dim; d++) {
    bounds = (*gridBounds)[d];
    if (verbose) {
      std::cout << "dim " << d+1 << ": ("
		<< gsl_vector_get(bounds, 0) << ", "
		<< gsl_vector_get(bounds, bounds->size - 1) << ", "
		<< (bounds->size - 1) << ")" << std::endl;
    }
    
    for (size_t i = 0; i < bounds->size; i++)
      fprintf(gridFile, "%lf ", gsl_vector_get((*gridBounds)[d], i));
    fprintf(gridFile, "\n");
  }

  return;
}


#endif
