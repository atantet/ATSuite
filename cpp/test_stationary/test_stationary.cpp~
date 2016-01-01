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
  char src_mem[128];
  FILE *f, *f_mem;
  SparseMatrix<double> *T = new SparseMatrix<double>;
  int N;
  int crap;
  VectorXi member;
  double codelength;
  
  for (int dt = 3; dt <16; dt++){
    cout << "Time-step " << dt << endl;
    sprintf(src_path, "%s/graph/graph_phase_nt73000_N2226_dt%02d.net", src_dir, dt);
    sprintf(src_mem, "%s/com/com_multi_phase_nt73000_N2226_dt%02d.txt", src_dir, dt);

    // Open source files
    if ((f = fopen(src_path, "r")) == NULL){
      fprintf(stderr, "Could not open Pajek file for reading.\n");
      return 1;
    }
    
    if ((f_mem = fopen(src_mem, "r")) == NULL){
      fprintf(stderr, "Could not open membership file.\n");
      return 1;
    }

    // Read to COO matrix
    T = pajek2EigenSparse(f);
    N = T->rows();
    member = VectorXi(N);

    // Read Membership vector
    for (int i = 0; i < N; i++)
      fscanf(f_mem, "%d %d", &(member(i)), &crap);
  
    codelength = twolevelCodelength(T, &member);

    fclose(f);
    fclose(f_mem);
  }
  return 0;
}
