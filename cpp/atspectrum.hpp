#ifndef ATSPECTRUM_HPP
#define ATSPECTRUM_HPP

#include "arlnsmat.h"
#include "arlsnsym.h"

// Declarations
void getEigValNonSym(ARluNonSymMatrix<double, double> *, double *, double *,
		     int, const char *, double, int, int, bool, double *);

// Definitions
void getEigValNonSym(ARluNonSymMatrix<double, double> *P,
		     double *EigValReal, double *EigValImag,
		     int nev, const char *which, double tol=0., int maxit=0,
		     int ncv=0, bool AutoShift=true, double *resid=NULL)
{
  ARluNonSymStdEig<double> EigProb;

  // Define eigenvalue problem
  EigProb = ARluNonSymStdEig<double>(nev, *P, which, ncv, tol, maxit, resid,
				     AutoShift);

  // Get eigenvalues
  EigProb.Eigenvalues(EigValReal, EigValImag);

  return;
}


#endif
