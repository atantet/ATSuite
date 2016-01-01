#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <atcom.hpp>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <atgraph.hpp>

using namespace std;
using namespace Eigen;

int main(void)
{
  char src_dir[] = "/Users/atantet/PhD/dev/bve_t21/transfer/pc01/";
  int N = 980;
  char src_path[128];
  FILE *f;
  ofstream fCode;
  SparseMatrix<double> *highT = new SparseMatrix<double>;
  VectorXd *codelength = new VectorXd;
  VectorXi member;
  
  for (int dt = 8; dt <= 8; dt++){
    cout << "Time-step " << dt << endl;
    sprintf(src_path, "%s/graph/graph_phase_nt73000_N%d_dt%02d.net", src_dir, N, dt);
    
    // Open source files
    if ((f = fopen(src_path, "r")) == NULL){
      fprintf(stderr, "Could not open Pajek file for reading.\n");
      return 1;
    }
    
    // Read to COO matrix
    highT = pajek2EigenSparse(f);

    greedyCodelengthUpdate(highT, codelength);

    // Open source files
    fCode.open("codelength.txt");
    fCode << *codelength;
    fCode.close();      

    fclose(f);
  }
  return 0;
}
