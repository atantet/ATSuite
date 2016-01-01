#ifndef TRANSFEROPERATORTEST_HPP
#define TRANSFEROPERATORTEST_HPP

#include <vector>
#include <cmath>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_vector_int.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "atmath.hpp"
#include "atio.hpp"
#include "atspectrum.hpp"
#include "transferOperator.hpp"

typedef Eigen::Triplet<double> triplet;
typedef std::vector<triplet> tripletVector;
typedef Eigen::SparseMatrix<double, Eigen::ColMajor> SpMatCSC;
typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SpMatCSR;


// Declarations
void  getSurrogateSpectrum(gsl_vector_int *, size_t, size_t, gsl_matrix *, gsl_matrix *,
			   const char *, double, int, int, bool, double *);
void getSurrogateSpectrumFromCount(SpMatCSR *, gsl_rng *, gsl_matrix *, gsl_matrix*,
				   const char *, double, int, int, bool, double *);
SpMatCSR *getShuffledCountMatrix(SpMatCSR *, gsl_vector *, gsl_rng *);
void getShuffledRow(SpMatCSR *, size_t, size_t, tripletVector *, gsl_rng *);


// Definitions
void getSurrogateSpectrum(gsl_vector_int * gridMem, size_t N, size_t tauStep,
			  gsl_matrix *EigValRealDist, gsl_matrix *EigValImagDist,
			  const char *which="LM", double tol=0., int maxit=0, int ncv=0,
			  bool AutoShift=true, double *resid=NULL)
{
  SpMatCSR *C = new SpMatCSR(N, N);
  tripletVector *T;
  std::vector< std::vector <int > > boxDest;
  std::vector<int> rowTmp;
  const gsl_rng_type *rngType = gsl_rng_ranlxs1;
  int seed=0.;

  // Get transition count triplets
  T = getTransitionCountTriplets(gridMem, tauStep);
  
  // Get transition count matrix
  C->setFromTriplets(T->begin(), T->end());

  // Set random generator and its seed
  gsl_rng *r = gsl_rng_alloc(rngType);
  gsl_rng_set(r, seed);
  
  // Get surrogate spectrum
  getSurrogateSpectrumFromCount(C, r, EigValRealDist, EigValImagDist,
				which, tol, maxit, ncv, AutoShift, resid);

  gsl_rng_free(r);
  delete C;
  delete T;

  return;
}


void getSurrogateSpectrumFromCount(SpMatCSR *C, gsl_rng *r,
				   gsl_matrix *EigValRealDist, gsl_matrix *EigValImagDist,
				   const char *which="LM", double tol=0., int maxit=0,
				   int ncv=0, bool AutoShift=true, double *resid=NULL)
{
  int nev = EigValRealDist->size1;
  int Ns = EigValRealDist->size2;
  int N = C->rows();
  gsl_vector *nTransPerRow;

  // Get row sums
  nTransPerRow = getRowSum(C);

#pragma omp parallel
  {
    double *EigValReal;
    double *EigValImag;
    SpMatCSR *Ps;
    SpMatCSR CCopy(N, N);
    ARluNonSymMatrix<double, double> *PsAR;
#pragma omp critical
    {
      CCopy = *C;
    }
    
    // Get Ns surrogate spectra
#pragma omp for
    for (int s = 0; s < Ns; s++){
      // Get shuffled count matrix
      std::cout << "Getting surrogate eigenvalues " << s << " out of " << (Ns-1)
		<< std::endl;
      Ps = getShuffledCountMatrix(&CCopy, nTransPerRow, r);
      // Get transition matrix
      toStochastic(Ps);
      // Convert to Arpack matrix format
      PsAR = Eigen2AR(Ps);
      
#pragma omp critical
      {
	// Get eigenvalues
	EigValReal = new double[nev + 2];
	EigValImag = new double[nev + 2];
	getEigValNonSym(PsAR, EigValReal, EigValImag, nev, which, tol, maxit, ncv,
			AutoShift, resid);
      
	// Store
	for (int ev = 0; ev < nev; ev++){
	  gsl_matrix_set(EigValRealDist, ev, s, EigValReal[ev]);
	  gsl_matrix_set(EigValImagDist, ev, s, EigValImag[ev]);
	}
      }
      
      // Clean
      delete[] EigValReal;
      delete[] EigValImag;
      delete Ps;
      delete PsAR;
    }
  }
  
  gsl_vector_free(nTransPerRow);

  return;
}


SpMatCSR *getShuffledCountMatrix(SpMatCSR *C, gsl_vector *nTransPerRow, gsl_rng *r)
{
  size_t N = C->rows();
  SpMatCSR *Cs = new SpMatCSR(N, N);
  tripletVector *Ts = new tripletVector;
  size_t nTrans;
  
  // Reserve shuffled transition count triplet
  Ts->reserve(C->nonZeros());

  // Shuffle each row
  for (size_t i = 0; i < N; i++){
    nTrans = (size_t) round(gsl_vector_get(nTransPerRow, i));
    if (nTrans > 0)
      getShuffledRow(C, i, nTrans, Ts, r);
  }

  // Get shuffled matrix
  Cs->setFromTriplets(Ts->begin(), Ts->end());
  delete Ts;
  
  return Cs;
}

void getShuffledRow(SpMatCSR *C, size_t iRow, size_t nTrans,
		    tripletVector *Ts,
		    gsl_rng *r)
{
  size_t k;
  gsl_vector_int *dstRepeat = gsl_vector_int_alloc(nTrans);
  unsigned long int sampleIdx;
  int sample;

  // Get repeated list of destinations
  k = 0;
  for (SpMatCSR::InnerIterator it(*C, iRow); it; ++it){
    for (int r = 0; r < it.value(); r++){
      gsl_vector_int_set(dstRepeat, k, it.col());
      k++;
    }
  }

  // Shuffle row
  for (k = 0; k < nTrans; k++){
    sampleIdx = gsl_rng_uniform_int(r, nTrans);
    sample = gsl_vector_int_get(dstRepeat, sampleIdx);
    Ts->push_back(triplet(iRow, sample, 1));
  }

  gsl_vector_int_free(dstRepeat);
  return;
}


#endif
