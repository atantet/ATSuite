#ifndef ATMATH_HPP
#define ATMATH_HPP

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <list>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_vector_int.h>
#include <gsl/gsl_matrix.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#define plog2p( x ) ( (x) > 0.0 ? (x) * log(x) / log(2) : 0.0 )

using namespace std;
using namespace Eigen;

typedef SparseMatrix<double, RowMajor> SpMatCSR;
typedef SparseMatrix<double, ColMajor> SpMatCSC;
typedef SparseMatrix<bool, ColMajor> SpMatCSCBool;
typedef SparseMatrix<int, RowMajor> SpMatIntCSR;

// Declarations
double entropy(VectorXd *);
double entropyRate(SpMatCSC *, VectorXd *);
double entropyRate(MatrixXd *, VectorXd *);
void toRightStochastic(SpMatCSC *);
void toRightStochastic(SpMatCSR *);
void toLeftStochastic(SpMatCSR *);
void toAndStochastic(SpMatCSR *);
void toAndStochastic(SpMatCSC *);
gsl_vector* getRowSum(SpMatCSR *);
void getRowSum(SpMatCSR *, gsl_vector *);
gsl_vector_int* getRowSum(SpMatIntCSR *);
gsl_vector* getColSum(SpMatCSR *);
void getColSum(SpMatCSR *, gsl_vector *);
gsl_vector* getColSum(SpMatCSC *);
double getSum(SpMatCSR *);
double sumVectorElements(gsl_vector *);
void normalizeVector(gsl_vector *);
void normalizeRows(SpMatCSR *, gsl_vector *);
void condition4Entropy(SpMatCSC *);
SpMatCSCBool * cwiseGT(SpMatCSC *, double);
SpMatCSCBool * cwiseLT(SpMatCSC *, double);
bool any(SpMatCSCBool *);
double max(SpMatCSC *);
double min(SpMatCSC *);
vector<int> * argmax(SpMatCSC *);
void lowlevelTransition(SpMatCSC *, VectorXd *, VectorXi *,
			MatrixXd *, VectorXd *);


// Definitions
double entropy(VectorXd *dist)
{
  double s = 0.;
  for (int k = 0; k < dist->size(); k++)
    s -= plog2p((*dist)(k));
  return s;
}

double entropyRate(SpMatCSC *T, VectorXd *dist)
{
  int i, j;
  double s = 0.;
  for (j = 0; j < T->outerSize(); j++){
    for (SpMatCSC::InnerIterator it(*T, j); it; ++it){
      i = it.row();
      // Increment low-level transition matrix
      s -= (*dist)(i) * plog2p(it.value());
    }
  }
  return s;
}

double entropyRate(MatrixXd *T, VectorXd *dist)
{
  int i, j;
  double s = 0.;
  for(j = 0; j < T->cols(); j++){
    for (i = 0; i < T->rows(); i++){
      // Increment low-level transition matrix
      s -= (*dist)(i) * plog2p((*T)(i, j));
    }
  }
  return s;
}  


void toRightStochastic(SpMatCSC *T)
{
  int j;
  VectorXd rowSum = VectorXd::Zero(T->rows());
  // Calculate row sums
  for (j = 0; j < T->cols(); j++)
    for (SpMatCSC::InnerIterator it(*T, j); it; ++it)
      rowSum(it.row()) += it.value();

  // Normalize rows
  for (j = 0; j < T->cols(); j++)
    for (SpMatCSC::InnerIterator it(*T, j); it; ++it)
      if (rowSum(it.row()) > 0.)
	it.valueRef() /= rowSum(it.row());

  return;
}
  

void toRightStochastic(SpMatCSR *T)
{
  // Get row sum vector
  gsl_vector *rowSum = getRowSum(T);

  // Normalize rows
  normalizeRows(T, rowSum);

  // Free row sum vector
  gsl_vector_free(rowSum);
  
  return;
}
  

void toLeftStochastic(SpMatCSR *T)
{
  int j;
  VectorXd colSum = VectorXd::Zero(T->cols());
  // Calculate row sums
  for (j = 0; j < T->rows(); j++)
    for (SpMatCSR::InnerIterator it(*T, j); it; ++it)
      colSum(it.col()) += it.value();

  // Normalize rows
  for (j = 0; j < T->rows(); j++)
    for (SpMatCSR::InnerIterator it(*T, j); it; ++it)
      if (colSum(it.col()) > 0.)
	it.valueRef() /= colSum(it.col());

  return;
}

// Normalize by sum of elements
void toAndStochastic(SpMatCSR *T)
{
  double norm = getSum(T);
  // Normalize
  for (int outerIdx = 0; outerIdx < T->outerSize(); outerIdx++)
    for (SpMatCSR::InnerIterator it(*T, outerIdx); it; ++it)
      it.valueRef() /= norm;
  return;
}

// Normalize by sum of elements
void toAndStochastic(SpMatCSC *T)
{
  double norm = 0.;
  // Get Norm
  for (int outerIdx = 0; outerIdx < T->outerSize(); outerIdx++)
    for (SpMatCSC::InnerIterator it(*T, outerIdx); it; ++it)
      norm += it.value();
  // Normalize
  for (int outerIdx = 0; outerIdx < T->outerSize(); outerIdx++)
    for (SpMatCSC::InnerIterator it(*T, outerIdx); it; ++it)
      it.valueRef() /= norm;
  return;
}

gsl_vector* getRowSum(SpMatCSR *T)
{
  int N = T->rows();
  gsl_vector *rowSum = gsl_vector_calloc(N);

  // Calculate row sums
  for (int i = 0; i < N; i++)
    for (SpMatCSR::InnerIterator it(*T, i); it; ++it)
      gsl_vector_set(rowSum, i, gsl_vector_get(rowSum, i)+it.value()); 
  
  return rowSum;
}

void getRowSum(SpMatCSR *T, gsl_vector *rowSum)
{
  int N = T->rows();

  // Calculate row sums
  gsl_vector_set_all(rowSum, 0.);
  for (int i = 0; i < N; i++)
    for (SpMatCSR::InnerIterator it(*T, i); it; ++it)
      gsl_vector_set(rowSum, i, gsl_vector_get(rowSum, i)+it.value()); 
  
  return;
}

gsl_vector* getColSum(SpMatCSR *T)
{
  int N = T->rows();
  gsl_vector *colSum = gsl_vector_calloc(N);

  // Calculate col sums
  for (int irow = 0; irow < N; irow++)
    for (SpMatCSR::InnerIterator it(*T, irow); it; ++it)
      gsl_vector_set(colSum, it.col(),
		     gsl_vector_get(colSum, it.col()) + it.value()); 
  
  return colSum;
}

void getColSum(SpMatCSR *T, gsl_vector *colSum)
{
  int N = T->rows();

  // Calculate col sums
  gsl_vector_set_all(colSum, 0.);
  for (int irow = 0; irow < N; irow++)
    for (SpMatCSR::InnerIterator it(*T, irow); it; ++it)
      gsl_vector_set(colSum, it.col(),
		     gsl_vector_get(colSum, it.col()) + it.value()); 
  
  return;
}

gsl_vector* getColSum(SpMatCSC *T)
{
  int N = T->rows();
  gsl_vector *colSum = gsl_vector_calloc(N);
  // Calculate col sums
  for (int icol = 0; icol < N; icol++)
    for (SpMatCSC::InnerIterator it(*T, icol); it; ++it)
      gsl_vector_set(colSum, icol, gsl_vector_get(colSum, icol) + it.value()); 
  
  return colSum;
}


double getSum(SpMatCSR *T)
{
  int N = T->rows();
  double sum = 0.;
  // Calculate col sums
  for (int irow = 0; irow < N; irow++)
    for (SpMatCSR::InnerIterator it(*T, irow); it; ++it)
      sum += it.value();
  
  return sum;
}


// Get the sum of the elements of a vector
double sumVectorElements(gsl_vector *v)
{
  double sum = 0.;
  
  for (size_t i = 0; i < v->size; i++)
    sum += gsl_vector_get(v, i);
  
  return sum;
}


// Normalize the elements of a vector
void normalizeVector(gsl_vector *v)
{
  double sum;

  sum = sumVectorElements(v);
  gsl_vector_scale(v, 1. / sum);
  
  return;
}


gsl_vector_int* getRowSum(SpMatIntCSR *T)
{
  int N = T->rows();
  gsl_vector_int *rowSum = gsl_vector_int_calloc(N);
  // Calculate row sums
  for (int i = 0; i < N; i++)
    for (SpMatIntCSR::InnerIterator it(*T, i); it; ++it)
      gsl_vector_int_set(rowSum, i, gsl_vector_int_get(rowSum, i)+it.value()); 
  
  return rowSum;
}


void normalizeRows(SpMatCSR *T, gsl_vector *rowSum)
{
  double rowSumi;
  for (int i = 0; i < T->rows(); i++){
    rowSumi = gsl_vector_get(rowSum, i);
    for (SpMatCSR::InnerIterator it(*T, i); it; ++it)
      if (rowSumi > 0.)
	it.valueRef() /= rowSumi;
  }
  return ;
}


void condition4Entropy(SpMatCSC *T)
{
  int j;
  VectorXd rowSum = VectorXd::Zero(T->rows());
  // Calculate row sums
  for (j = 0; j < T->cols(); j++){
    for (SpMatCSC::InnerIterator it(*T, j); it; ++it){
      if (it.value() > 1.)
	it.valueRef() = 1.;
      if (it.value() < 0.)
	it.valueRef() = 0.;
    }
  }

  return;
}


SpMatCSCBool * cwiseGT(SpMatCSC *T, double ref)
{
  int j;
  SpMatCSCBool *cmpT = new SpMatCSCBool(T->rows(), T->cols());
  for (j = 0; j < T->cols(); j++)
    for (SpMatCSC::InnerIterator it(*T, j); it; ++it)
      cmpT->insert(it.row(), j) = it.value() > ref;

  return cmpT;
}
  

SpMatCSCBool * cwiseLT(SpMatCSC *T, double ref)
{
  int j;
  SpMatCSCBool *cmpT = new SpMatCSCBool(T->rows(), T->cols());
  for (j = 0; j < T->cols(); j++)
    for (SpMatCSC::InnerIterator it(*T, j); it; ++it)
      cmpT->insert(it.row(), j) = it.value() < ref;

  return cmpT;
}


bool any(SpMatCSCBool *T)
{
  int j;
  for (j = 0; j < T->cols(); j++)
    for (SpMatCSCBool::InnerIterator it(*T, j); it; ++it)
      if (it.value())
	return true;

  return false;
}


double max(SpMatCSC *T)
{
  int j;
  SpMatCSC::InnerIterator it(*T, 0);
  double maxValue = it.value();
  for (j = 0; j < T->cols(); j++)
    for (SpMatCSC::InnerIterator it(*T, j); it; ++it)
      if (it.value() > maxValue)
	maxValue = it.value();

  return maxValue;
}


double min(SpMatCSC *T)
{
  int j;
  SpMatCSC::InnerIterator it(*T, 0);
  double minValue = it.value();
  for (j = 0; j < T->cols(); j++)
    for (SpMatCSC::InnerIterator it(*T, j); it; ++it)
      if (it.value() < minValue)
	minValue = it.value();

  return minValue;
}


vector<int> * argmax(SpMatCSC *T)
{
  int j;
  vector<int> *argmax = new vector<int>(2);
  SpMatCSC::InnerIterator it(*T, 0);
  double maxValue = it.value();
  for (j = 0; j < T->cols(); j++){
    for (SpMatCSC::InnerIterator it(*T, j); it; ++it){
      if (it.value() > maxValue){
	argmax->at(0) = it.row();
	argmax->at(1) = it.col();
	maxValue = it.value();
      }
    }
  }

  return argmax;
}

void lowlevelTransition(SpMatCSC *highT, VectorXd *highDist, VectorXi *member,
			MatrixXd *lowT, VectorXd *lowDist)
{
  const int N = highT->rows();
  int ncom;
  int i, j;

  // Get set of communities from membership vector
  list<int>  coms(member->data(), member->data() + N);
  coms.sort();
  coms.unique();
  ncom = coms.size();

  // Allocate and initialize low-level transition matrix
  *lowT = MatrixXd::Zero(ncom, ncom);
  *lowDist = VectorXd::Zero(ncom);

  // Get community map
  map<int, int> comMap;
  list<int>::iterator it = coms.begin();
  for (int k = 0; k < ncom; k++){
    comMap[*it] = k;
    advance(it, 1);
  }

  // Increment low-level stationary distribution
  for (int i = 0; i < N; i++)
    (*lowDist)(comMap[(*member)(i)]) += (*highDist)(i);

  for (j = 0; j < N; j++){
    for (SpMatCSC::InnerIterator it(*highT, j); it; ++it){
      i = it.row();
      // Increment low-level transition matrix
      (*lowT)(comMap[(*member)(i)], comMap[(*member)(j)]) += (*highDist)(i) * it.value();
    }
  }
  // Normalize
  for (i = 0; i < ncom; i++)
    for (j = 0; j < ncom; j++)
      (*lowT)(i, j) = (*lowT)(i, j) / (*lowDist)(i);

  return;
}


#endif
