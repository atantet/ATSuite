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
  //  FILE *f_mem;
  SparseMatrix<double> *highT = new SparseMatrix<double>(6, 6);
  VectorXd *codelength = new VectorXd;
  VectorXd *modularity = new VectorXd;
  VectorXd *sLowDist = new VectorXd;
  VectorXd *sLowT = new VectorXd;
  double *sHighDist = new double;
  char dstMember[128];
  ofstream fMember;
  int N = 6;
  int nStore = N;
  MatrixXi *memberStore = new MatrixXi(N, nStore);
  MatrixXi *sizeStore = new MatrixXi(N, nStore);
  MatrixXd *exitStore = new MatrixXd(N, nStore);
  MatrixXd *lowDistStore = new MatrixXd(N, nStore);

  highT->coeffRef(0, 1) = 0.5;
  highT->coeffRef(0, 2) = 0.5;
  highT->coeffRef(1, 0) = 0.5;
  highT->coeffRef(1, 2) = 0.5;
  highT->coeffRef(2, 0) = 1./3;
  highT->coeffRef(2, 1) = 1./3;
  highT->coeffRef(2, 3) = 1./3;
  highT->coeffRef(3, 2) = 1./3;
  highT->coeffRef(3, 4) = 1./3;
  highT->coeffRef(3, 5) = 1./3;
  highT->coeffRef(4, 3) = 0.5;
  highT->coeffRef(4, 5) = 0.5;
  highT->coeffRef(5, 3) = 0.5;
  highT->coeffRef(5, 4) = 0.5;

  greedyCodelengthChange(highT, codelength, sHighDist, sLowDist, sLowT, modularity, memberStore, lowDistStore, sizeStore, exitStore, 0.);
  
  for (int k = 0; k < nStore; k++){
    sprintf(dstMember, "membership_ncom%d.txt", N - k);
    fMember.open(dstMember);
    fMember << memberStore->col(k);
    fMember.close();
  }
  
  delete highT;
  delete codelength;
  delete modularity;
  delete memberStore;
  delete lowDistStore;
  delete sizeStore;
  
  return 0;
}