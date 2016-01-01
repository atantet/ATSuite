#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <atcom_alglib.hpp>
#include <atgraph.hpp>

using namespace std;
using namespace alglib;
typedef sparsematrix spAlg;

int main(void)
{
  //  FILE *f_mem;
  spAlg *highT = new spAlg;
  real_1d_array *codelength = new real_1d_array;
  real_1d_array *sLowDist = new real_1d_array;
  real_1d_array *sLowT = new real_1d_array;
  integer_2d_array *memberStore = new integer_2d_array;
  integer_2d_array *sizeStore = new integer_2d_array;
  real_2d_array *lowDistStore = new real_2d_array;

  sparsecreate(6, 6, 14, *highT);
  sparseset(*highT, 0, 1, 0.5);
  sparseset(*highT, 0, 2, 0.5);
  sparseset(*highT, 1, 0, 0.5);
  sparseset(*highT, 1, 2, 0.5);
  sparseset(*highT, 2, 0, 1. / 3);
  sparseset(*highT, 2, 1, 1. / 3);
  sparseset(*highT, 2, 3, 1. / 3);
  sparseset(*highT, 3, 2, 1. / 3);
  sparseset(*highT, 3, 4, 1. / 3);
  sparseset(*highT, 3, 5, 1. / 3);
  sparseset(*highT, 4, 3, 0.5);
  sparseset(*highT, 4, 5, 0.5);
  sparseset(*highT, 5, 3, 0.5);
  sparseset(*highT, 5, 4, 0.5);

  memberStore->setlength(6, 6);
  sizeStore->setlength(6, 6);
  lowDistStore->setlength(6, 6);

  greedyCodelength(highT, codelength, sLowDist, sLowT, memberStore, lowDistStore, sizeStore, 0.);
  return 0;
}
