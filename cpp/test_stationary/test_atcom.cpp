#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <atcom.hpp>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <atgraph.hpp>

using namespace std;
using namespace Eigen;

int main(void)
{
  char src_dir[] = "/Users/atantet/PhD/dev/bve_t21/transfer/pc01/";
  char src_path[128];
  FILE *f, *f_mem;
  SparseMatrix<double> *highT = new SparseMatrix<double>;
  VectorXd *codelength = new VectorXd;
  int N;
  VectorXi member;
  
  for (int dt = 3; dt <16; dt++){
    cout << "Time-step " << dt << endl;
    sprintf(src_path, "%s/graph/graph_phase_nt73000_N2226_dt%02d.net", src_dir, dt);

    // Open source files
    if ((f = fopen(src_path, "r")) == NULL){
      fprintf(stderr, "Could not open Pajek file for reading.\n");
      return 1;
    }
    
    // Read to COO matrix
    highT = pajek2EigenSparse(f);
    N = highT->rows();

    greedyCodelength(highT, codelength);
    cout << (*codelength)(0) << endl;

    fclose(f);
  }
  return 0;
}