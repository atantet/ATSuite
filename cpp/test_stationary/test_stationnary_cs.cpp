#include <cstdlib>
#include <cstdio>
#include <vector>
#include <atgraph.hpp>
#include <cs.h>
#include <iostream>
#include <cmath>

using namespace std;

int main(void)
{
  const float tau = 0.;
  const double tol = 1.e-10;

  int N;
  double eps;
  int k;
  char src_path[] = "/Users/atantet/PhD/dev/bve_t21/transfer/pc01/graph/graph_phase_nt73000_N2226_dt09.net";
  FILE *f;
  cs *T

  // Open source file
  if ((f = fopen(src_path, "r")) == NULL){
    fprintf(stderr, "Could not open Pajek file for reading.\n");
    return 1;
  }

  // Read to COO matrix
  printf("Reading triplet from file...\n");
  T = pajek2triplet(f);
  printf("Matrix of shape (%d, %d) with %d non-zeros.\n", T->n, T->m, T->nzmax);
  N = T->n;

  // Get Dense
  printf("Getting dense matrix...\n");
  Td = MatrixXd(*T);
  
  // Add teletransportation
  Ttele = Td * (1. - tau) + tau * MatrixXd::Ones(N, N) / N;

  // First matrix multiplication
  u = RowVectorXd::Ones(N) / N;
  uNew = u * Ttele;
  epsVect = uNew - u;
  eps = epsVect.dot(epsVect);
  
  while (eps > pow(tol, 2)) {
    u = uNew;
    uNew = u * Ttele;
    epsVect = uNew - u;
    eps = epsVect.dot(epsVect);
    k++;
  }

  // Normalize
  uNew = uNew / uNew.sum();

  cout << "Dominant left eigen-vector is:\n" << uNew << endl;
  cout << "Error " << eps << " reached in " << k << " steps.\n" << endl;

  FILE *dst_f = fopen("rank.txt", "w");
  for (int k = 0; k < N; k++)
    fprintf(dst_f, "%e\n", uNew(k));
  
  fclose(f);
  fclose(dst_f);

  return 0;
}
