#ifndef ATGRAPH_ALGLIB_HPP
#define ATGRAPH_ALGLIB_HPP

#include <cstdio>
#include <cstdlib>
#include <stack>
#include <vector>
#include <iostream>
#include <igraph/igraph.h>
#include "linalg.h"

#define plog2p( x ) ( (x) > 0.0 ? (x) * log(x) / log(2) : 0.0 )

using namespace std;
using namespace alglib;

typedef sparsematrix spAlg;

spAlg * pajek2HashTable(FILE *f){
  char label[20];
  int N, E;
  int i, j, i0;
  double x;
  spAlg *hashTable = new spAlg;

  // Read vertices
  fscanf(f, "%s %d", label, &N);

  // Read first (assume monotonous)
  fscanf(f, "%d %s", &i0, label);
  for (int k = 1; k < N; k++){
    fscanf(f, "%d %s", &i, label);
  }

  // Read Edges
  fscanf(f, "%s %d", label, &E);

  // Create Hash Table
  sparsecreate(N, N, E, *hashTable);

  for (int k = 0; k < E; k++){
    fscanf(f, "%d %d %lf", &i, &j, &x);
    sparseset(*hashTable, i, j, x);
  }
  
  return hashTable;
}


spAlg * igraph2HashTable(igraph_t *srcGraph)
{
  igraph_integer_t N, E;
  igraph_vector_t edges, weights;
  spAlg *hashTable = new spAlg;

  // Get edge list and weights
  N = igraph_vcount(srcGraph);
  E = igraph_ecount(srcGraph);
  igraph_vector_init(&edges, E * 2);
  igraph_vector_init(&weights, E);
  igraph_get_edgelist(srcGraph, &edges, true);
  if ((bool) igraph_cattribute_has_attr(srcGraph, IGRAPH_ATTRIBUTE_EDGE, "weight"))
    EANV(srcGraph, "weight", &weights);
  else
    igraph_vector_fill(&weights, 1.);

  // Create Hash Table
  sparsecreate(N, N, E, *hashTable);

  // Get triplet list from igraph graph
  for (int k = 0; k < E; k++)
    sparseset(*hashTable, VECTOR(edges)[k], VECTOR(edges)[E+k], VECTOR(weights)[k]);

return hashTable;
}

void setConstant(real_1d_array *v, double constant)
{
  for (int j = 0; j < v->length(); j++)
    (*v)(j) = constant;
  return;
}

void setConstant(integer_1d_array *v, int constant)
{
  for (int j = 0; j < v->length(); j++)
    (*v)(j) = constant;
  return;
}

void setLinSpaced(real_1d_array *v, double low, double high)
{
  int length = v->length();
  double stride = (high - low) / length;
  (*v)(0) = low;
  for (int j = 1; j < length; j++)
    (*v)(j) = (*v)(j-1) + stride;
  return;
}

void setLinSpaced(integer_1d_array *v, int low, int high)
{
  int length = v->length();
  int stride = (high - low) / length;
  (*v)(0) = low;
  for (int j = 1; j < length; j++)
    (*v)(j) = (*v)(j-1) + stride;
  return;
}

void setRow(real_2d_array *a, real_1d_array *row, int i)
{
  for (int j = 0; j < row->length(); j++)
    (*a)(i, j) = (*row)(j);
  return;
}

void setRow(integer_2d_array *a, integer_1d_array *row, int i)
{
  for (int j = 0; j < row->length(); j++)
    (*a)(i, j) = (*row)(j);
  return;
}

void setCol(real_2d_array *a, real_1d_array *col, int j)
{
  for (int i = 0; i < col->length(); i++)
    (*a)(i, j) = (*col)(i);
  return;
}

void setCol(integer_2d_array *a, integer_1d_array *col, int j)
{
  for (int i = 0; i < col->length(); i++)
    (*a)(i, j) = (*col)(i);
  return;
}

double getMin(real_1d_array *v, int *arg)
{
  double mn = (*v)(0);
  *arg = 0;
  for (int j = 0; j < v->length(); j++){
    if ((*v)(j) < mn){
      mn = (*v)(j);
      *arg = j;
    }
  } 
  return mn;
}

int getMin(integer_1d_array *v, int *arg)
{
  int mn = (*v)(0);
  *arg = 0;
  for (int j = 0; j < v->length(); j++){
    if ((*v)(j) < mn){
      mn = (*v)(j);
      *arg = j;
    }
  } 
  return mn;
}

int getNNZ(spAlg *s)
{
  int nnz = 0;
  ae_int_t t0 = 0;
  ae_int_t t1 = 0;
  ae_int_t i, j;
  double v;

  while (sparseenumerate(*s, t0, t1, i, j, v))
    nnz++;

  return nnz;
}
  
double entropy(real_1d_array *dist)
{
  double s = 0.;
  for (int k = 0; k < dist->length(); k++)
    s -= plog2p((*dist)(k));
  return s;
}

double entropyRate(spAlg *T, real_1d_array *dist)
{
  double s = 0.;
  ae_int_t t0 = 0;
  ae_int_t t1 = 0;
  ae_int_t i, j;
  double v;

  while (sparseenumerate(*T, t0, t1, i, j, v))
    s -= (*dist)(i) * plog2p(v);
  
  return s;
}
#endif
