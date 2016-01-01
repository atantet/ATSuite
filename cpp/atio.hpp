#ifndef ATIO_HPP
#define ATIO_HPP

#include <cstdlib>
#include <cstdio>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "arlnsmat.h"
#include "arlssym.h"
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

using namespace std;

typedef Eigen::SparseMatrix<double, Eigen::ColMajor> SpMatCSC;
typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SpMatCSR;
typedef Eigen::Triplet<double> Tri;


// Declarations

void Eigen2Pajek(FILE *, SpMatCSR *);

void Eigen2Compressed(FILE *, SpMatCSC *);

void Eigen2Compressed(FILE *, SpMatCSR *);

SpMatCSC * pajek2Eigen(FILE *);

ARluNonSymMatrix<double, double> * pajek2AR(FILE *);

ARluNonSymMatrix<double, double> * CSC2AR(FILE *);

ARluNonSymMatrix<double, double> * Eigen2AR(SpMatCSC *);

ARluNonSymMatrix<double, double> * Eigen2AR(SpMatCSR *);

ARluSymMatrix<double> * Eigen2ARSym(SpMatCSC *);

ARluSymMatrix<double> * Eigen2ARSym(SpMatCSR *);

ARluNonSymMatrix<double, double> * Compressed2AR(FILE *);

SpMatCSR * Compressed2Eigen(FILE *);

gsl_matrix * Compressed2EdgeList(FILE *);

void Compressed2EdgeList(FILE *, FILE *);

SpMatCSR * CSC2CSR(SpMatCSC *T);

SpMatCSC * CSR2CSC(SpMatCSR *T);

vector<Tri> Eigen2Triplet(SpMatCSC *);

vector<Tri> Eigen2Triplet(SpMatCSR *);

void fprintfEigen(FILE *, SpMatCSR *);
  
size_t lineCount(FILE *);

// Definitions

void Eigen2Pajek(FILE *f, SpMatCSR *P){
  int N = P->rows();
  int E = P->nonZeros();

  // Write vertices
  fprintf(f, "*Vertices %d\n", N);
  for (int k = 0; k < N; k++)
    fprintf(f, "%d \"%d\"\n", k, k);

  // Write Edges
  fprintf(f, "Edges %d\n", E);
  for (int i = 0; i < P->rows(); i++)
    for (SpMatCSR::InnerIterator it(*P, i); it; ++it)
      fprintf(f, "%d %d %lf\n", i, it.col(), it.value());

  return;
}


void Eigen2Compressed(FILE *f, SpMatCSC *P){
  char sparseType[] = "CSC";

  // Write type, inner size, outer size and number of non-zeros
  fprintf(f, "%s %d %d %d\n", sparseType, P->innerSize(), P->outerSize(), P->nonZeros());

  // Write values
  for (int nz = 0; nz < P->nonZeros(); nz++)
    fprintf(f, "%lf ", (P->valuePtr())[nz]);
  fprintf(f, "\n");

  // Write row indices
  for (int nz = 0; nz < P->nonZeros(); nz++)
    fprintf(f, "%d ", (P->innerIndexPtr())[nz]);
  fprintf(f, "\n");

  // Write first element of column pointer
  for (int outer = 0; outer < P->outerSize()+1; outer++)
    fprintf(f, "%d ", (P->outerIndexPtr())[outer]);
  fprintf(f, "\n");
  
  return;
}


void Eigen2Compressed(FILE *f, SpMatCSR *P){
  char sparseType[] = "CSR";

  // Write type, inner size, outer size and number of non-zeros
  fprintf(f, "%s %d %d %d\n", sparseType, P->innerSize(), P->outerSize(), P->nonZeros());

  // Write values
  for (int nz = 0; nz < P->nonZeros(); nz++)
    fprintf(f, "%lf ", (P->valuePtr())[nz]);
  fprintf(f, "\n");

  // Write column indices
  for (int nz = 0; nz < P->nonZeros(); nz++)
    fprintf(f, "%d ", (P->innerIndexPtr())[nz]);
  fprintf(f, "\n");

  // Write first element of row pointer
  for (int outer = 0; outer < P->outerSize()+1; outer++)
    fprintf(f, "%d ", (P->outerIndexPtr())[outer]);
  fprintf(f, "\n");
  
  return;
}


SpMatCSC * pajek2Eigen(FILE *f){
  char label[20];
  int N, E;
  int i, j, i0;
  double x;

  vector<Tri> tripletList;

  // Read vertices
  fscanf(f, "%s %d", label, &N);

  // Define sparse matrix
  SpMatCSC *T = new SpMatCSC(N, N);

  // Read first (assume monotonous)
  fscanf(f, "%d %s", &i0, label);
  for (int k = 1; k < N; k++){
    fscanf(f, "%d %s", &i, label);
  }

  // Read Edges
  fscanf(f, "%s %d", label, &E);

  // Reserve triplet capacity
  tripletList.reserve(E);

  for (int k = 0; k < E; k++){
    fscanf(f, "%d %d %lf", &i, &j, &x);
    tripletList.push_back(Tri(i - i0, j - i0, x));
  }
  
  // Assign matrix elements
  T->setFromTriplets(tripletList.begin(), tripletList.end());

  return T;
}


ARluNonSymMatrix<double, double> *pajek2AR(FILE *f)
{
  // The edge entries must be ordered by row and then by col.
  char label[20];
  int N, E, row, col;
  double val;
  int *irow, *pcol;
  double *nzval;
  ARluNonSymMatrix<double, double> *T = new ARluNonSymMatrix<double, double>;

  // Read vertices
  fscanf(f, "%s %d", label, &N);
  pcol = new int [N+1];

  // Read first (assume monotonous)
  for (int k = 0; k < N; k++)
    fscanf(f, "%d %s", &row, label);

  // Read number of edges
  fscanf(f, "%s %d", label, &E);
  irow = new int [E];
  nzval = new double [E];
  for (int k = 0; k < N+1; k++)
    pcol[k] = E;

  // Read edges
  for (int k = 0; k < E; k++){
    fscanf(f, "%d %d %lf", &row, &col, &val);
    irow[k] = row;
    printf("irow[%d] = %d\n", k, row);
    nzval[k] = val;
    if (k < pcol[col])
      pcol[col] = k;
  }

  // Define matrix, order=2 for degree ordering of A.T + A (good for transition mat)
  T->DefineMatrix(N, E, nzval, irow, pcol, 0.1, 2, true);

  return T;
}


ARluNonSymMatrix<double, double> * CSC2AR(FILE *f)
{
  int innerSize, outerSize, nnz;
  char sparseType[4];
  double *nzval;
  int *irow, *pcol;
  ARluNonSymMatrix<double, double> *T = new ARluNonSymMatrix<double, double>;

  // Read type, inner size, outer size and number of non-zeros and allocate
  fscanf(f, "%s %d %d %d", sparseType, &innerSize, &outerSize, &nnz);
  nzval = new double [nnz];
  irow = new int [nnz];
  pcol = new int[outerSize+1];

  // Read values
  for (int nz = 0; nz < nnz; nz++)
    fscanf(f, "%lf ", &nzval[nz]);

  // Read row indices
  for (int nz = 0; nz < nnz; nz++)
    fscanf(f, "%d ", &irow[nz]);

  // Read first element of column pointer
  for (int outer = 0; outer < outerSize+1; outer++)
    fscanf(f, "%d ", &pcol[outer]);
  
  // Define matrix, order=2 for degree ordering of A.T + A (good for transition mat)
  T->DefineMatrix(outerSize, nnz, nzval, irow, pcol, 0.1, 2, true);
  
  return T;
}


SpMatCSR *Compressed2Eigen(FILE *f)
{
  int innerSize, outerSize, nnz;
  char sparseType[4];
  double *nzval;
  int *innerIndexPtr, *outerIndexPtr;
  SpMatCSR *T;
  vector<Tri> tripletList;

  // Read type, inner size, outer size and number of non-zeros and allocate
  fscanf(f, "%s %d %d %d", sparseType, &outerSize, &innerSize, &nnz);
  nzval = new double [nnz];
  innerIndexPtr = new int [nnz];
  outerIndexPtr = new int[outerSize+1];
  T = new SpMatCSR(outerSize, innerSize);
  Eigen::VectorXf innerNNZ(outerSize);

  // Read values
  for (int nz = 0; nz < nnz; nz++)
    fscanf(f, "%lf ", &nzval[nz]);

  // Read inner indices (column)
  for (int nz = 0; nz < nnz; nz++)
    fscanf(f, "%d ", &innerIndexPtr[nz]);

  // Read first element of column pointer
  fscanf(f, "%d ", &outerIndexPtr[0]);
  for (int outer = 1; outer < outerSize+1; outer++){
    fscanf(f, "%d ", &outerIndexPtr[outer]);
    innerNNZ(outer-1) = outerIndexPtr[outer] - outerIndexPtr[outer-1];
  }
  T->reserve(innerNNZ);

  // Insert elements
  for (int outer = 0; outer < outerSize; outer++)
     for (int nzInner = outerIndexPtr[outer]; nzInner < outerIndexPtr[outer+1]; nzInner++)
       T->insertBackUncompressed(outer, innerIndexPtr[nzInner]) =  nzval[nzInner];
  //T.sumupDuplicates();

  delete nzval;
  delete innerIndexPtr;
  delete outerIndexPtr;
  
  return T;
}


ARluNonSymMatrix<double, double>* Eigen2AR(SpMatCSC *TEigen)
{
  int outerSize, nnz;
  double *nzval;
  int *irow, *pcol;
  ARluNonSymMatrix<double, double> *T = new ARluNonSymMatrix<double, double>;

  outerSize = TEigen->outerSize();
  nnz = TEigen->nonZeros();
  nzval = new double [nnz];
  irow = new int [nnz];
  pcol = new int [outerSize+1];

  // Set values
  nzval = TEigen->valuePtr();

  // Set inner indices
  irow = TEigen->innerIndexPtr();

  // Set first element of column pointer
  pcol = TEigen->outerIndexPtr();

  // Define matrix, order=2 for degree ordering of A.T + A (good for transition mat)
  T->DefineMatrix(outerSize, nnz, nzval, irow, pcol, 0.1, 2, true);
  
  return T;
}


ARluNonSymMatrix<double, double>* Eigen2AR(SpMatCSR *TEigenCSR)
{
  SpMatCSC *TEigen;
  ARluNonSymMatrix<double, double> *T;

  // Convert from Eigen CSR to CSC
  TEigen = CSR2CSC(TEigenCSR);

  // Convert from Eigen CSC to AR
  T = Eigen2AR(TEigen);

  return T;
}


ARluSymMatrix<double>* Eigen2ARSym(SpMatCSC *TEigen)
{
  int outerSize, nnz;
  double *nzval, *nzvalSym;
  int *irow, *pcol, *irowSym, *pcolSym;
  ARluSymMatrix<double> *T = new ARluSymMatrix<double>;
  int nzIdx, nzSymIdx, innerIdx;
  bool isNewCol;

  outerSize = TEigen->outerSize();
  nnz = TEigen->nonZeros();
  nzval = new double [nnz];
  irow = new int [nnz];
  pcol = new int [outerSize+1];
  nzvalSym = new double [(int) ((nnz-outerSize)/2)+outerSize+1];
  irowSym = new int [(int) ((nnz-outerSize)/2)+outerSize+1];
  pcolSym = new int [outerSize+1];

  // Set values
  nzval = TEigen->valuePtr();
  // Set inner indices
  irow = TEigen->innerIndexPtr();
  // Set first element of column pointer
  pcol = TEigen->outerIndexPtr();

  // Discard lower triangle
  nzIdx = 0;
  nzSymIdx = 0;
  for (int outerIdx = 0; outerIdx < outerSize; outerIdx++){
    isNewCol = true;
    while (nzIdx < pcol[outerIdx+1]){
      innerIdx = irow[nzIdx];
      if (outerIdx >= innerIdx){
	nzvalSym[nzSymIdx] = nzval[nzIdx];
	irowSym[nzSymIdx] = irow[nzIdx];
	if (isNewCol){
	  pcolSym[outerIdx] = nzSymIdx;
	  isNewCol = false;
	}
	nzSymIdx++;
      }
      nzIdx++;
    }
    // Check for empty column
    if (isNewCol) 
      pcolSym[outerIdx] = nzSymIdx;
  }
  pcolSym[outerSize] = nzSymIdx;

  // Define matrix, order=2 for degree ordering of A.T + A (good for transition mat)
  T->DefineMatrix(outerSize, nzSymIdx, nzvalSym, irowSym, pcolSym,
		  'U', 0.1, 2, true);
  
  return T;
}


ARluSymMatrix<double>* Eigen2ARSym(SpMatCSR *TEigenCSR)
{
  SpMatCSC *TEigen;
  ARluSymMatrix<double> *T;

  // Convert from Eigen CSR to CSC
  TEigen = CSR2CSC(TEigenCSR);

  // Convert from Eigen CSC to AR
  T = Eigen2ARSym(TEigen);

  return T;
}


ARluNonSymMatrix<double, double> * Compressed2AR(FILE *f)
{
  int innerSize, outerSize, nnz;
  char sparseType[4];
  double *nzval;
  int *irow, *pcol;
  ARluNonSymMatrix<double, double> *T = new ARluNonSymMatrix<double, double>;

  // Read type, inner size, outer size and number of non-zeros and allocate
  fscanf(f, "%s %d %d %d", sparseType, &innerSize, &outerSize, &nnz);
  nzval = new double [nnz];
  irow = new int [nnz];
  pcol = new int[outerSize+1];

  // Read values
  for (int nz = 0; nz < nnz; nz++)
    fscanf(f, "%lf ", &nzval[nz]);

  // Read row indices
  for (int nz = 0; nz < nnz; nz++)
    fscanf(f, "%d ", &irow[nz]);

  // Read first element of column pointer
  for (int outer = 0; outer < outerSize+1; outer++)
    fscanf(f, "%d ", &pcol[outer]);
  
  // Define matrix, order=2 for degree ordering of A.T + A (good for transition mat)
  T->DefineMatrix(outerSize, nnz, nzval, irow, pcol, 0.1, 2, true);
  
  return T;
}


gsl_matrix * Compressed2EdgeList(FILE *f)
{
  int innerSize, outerSize, nnz;
  char sparseType[4];
  double *nzval;
  int *idx, *ptr;
  gsl_matrix *EdgeList;
  int iOuter, iInner;
  
  // Read type, inner size, outer size and number of non-zeros and allocate
  fscanf(f, "%s %d %d %d", sparseType, &innerSize, &outerSize, &nnz);
  nzval = new double [nnz];
  idx = new int [nnz];
  ptr = new int[outerSize+1];
  EdgeList = gsl_matrix_alloc(nnz, 3);
  if (strcmp(sparseType, "CSR") == 0)
    iOuter = 0;
  else if (strcmp(sparseType, "CSR") == 0)
    iOuter = 1;
  else{
    std::cerr << "Invalid sparse matrix type." << std::endl;
    exit(EXIT_FAILURE);
  }
  iInner = (iOuter + 1)%2;

  // Read values
  for (int nz = 0; nz < nnz; nz++)
    fscanf(f, "%lf ", &nzval[nz]);

  // Read row indices
  for (int nz = 0; nz < nnz; nz++)
    fscanf(f, "%d ", &idx[nz]);

  // Read first element of column pointer
  for (int outer = 0; outer < outerSize+1; outer++)
    fscanf(f, "%d ", &ptr[outer]);

  int nz = 0;
  for (int outer = 0; outer < outerSize; outer++){
    for (int inner = ptr[outer]; inner < ptr[outer+1]; inner++){
	gsl_matrix_set(EdgeList, nz, iOuter, outer);
	gsl_matrix_set(EdgeList, nz, iInner, idx[inner]);
	gsl_matrix_set(EdgeList, nz, 2, nzval[inner]);
	nz++;
    }
  }

  delete(nzval);
  delete(idx);
  delete(ptr);
  return EdgeList;
}

void Compressed2EdgeList(FILE *src, FILE *dst)
{
  gsl_matrix *EdgeList = Compressed2EdgeList(src);
  size_t nnz = EdgeList->size1;

  for (size_t nz = 0; nz < nnz; nz++)
    fprintf(dst, "%d\t%d\t%lf\n",
	    (int) gsl_matrix_get(EdgeList, nz, 0),
	    (int) gsl_matrix_get(EdgeList, nz, 1),
	    gsl_matrix_get(EdgeList, nz, 2));

  gsl_matrix_free(EdgeList);
  return;
}

SpMatCSR * CSC2CSR(SpMatCSC *T){
  int N = T->rows();

  // Define sparse matrix
  SpMatCSR *TCSR = new SpMatCSR(N, N);

  // Get triplet list
  vector<Tri> tripletList = Eigen2Triplet(T);

  // Assign matrix elements
  TCSR->setFromTriplets(tripletList.begin(), tripletList.end());

  return TCSR;
}


SpMatCSC * CSR2CSC(SpMatCSR *T){
  int N = T->rows();

  // Define sparse matrix
  SpMatCSC *TCSC = new SpMatCSC(N, N);

  // Get triplet list
  vector<Tri> tripletList = Eigen2Triplet(T);

  // Assign matrix elements
  TCSC->setFromTriplets(tripletList.begin(), tripletList.end());

  return TCSC;
}


vector<Tri> Eigen2Triplet(SpMatCSC *T)
{
  // Works for column major only
  int nnz = T->nonZeros();
  vector<Tri> tripletList;
  tripletList.reserve(nnz);

  for (int beta = 0; beta < T->outerSize(); ++beta)
    for (SpMatCSC::InnerIterator it(*T, beta); it; ++it)
      tripletList.push_back(Tri(it.row(), it.col(), it.value()));

  return tripletList;
}


vector<Tri> Eigen2Triplet(SpMatCSR *T)
{
  // Works for column major only
  int nnz = T->nonZeros();
  vector<Tri> tripletList;
  tripletList.reserve(nnz);

  for (int beta = 0; beta < T->outerSize(); ++beta)
    for (SpMatCSR::InnerIterator it(*T, beta); it; ++it)
      tripletList.push_back(Tri(it.row(), it.col(), it.value()));

  return tripletList;
}


void fprintfEigen(FILE *fp, SpMatCSR *T, const char *format)
{
  int count;
  for (int irow = 0; irow < T->outerSize(); ++irow){
    count = 0;
    for (SpMatCSR::InnerIterator it(*T, irow); it; ++it){
      while (count < it.col()){
	fprintf(fp, "---\t");
	count++;
      }
      fprintf(fp, format, it.value());
      fprintf(fp, "\t");
      count++;
    }
    while (count < T->innerSize()){
      fprintf(fp, "---\t");
      count++;
    }
    fprintf(fp, "\n");
  }
  return;
}


// Count number of lines in a text file
size_t lineCount(FILE *fp)
{
  size_t count = 0;
  int ch;

  // Count lines
  do {
    ch = fgetc(fp);
    if (ch == '\n')
      count++;
  } while (ch != EOF);

  return count;
}

#endif
