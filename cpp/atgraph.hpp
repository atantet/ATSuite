#ifndef ATGRAPH_HPP
#define ATGRAPH_HPP

#include <cstdio>
#include <cstdlib>
#include <stack>
#include <vector>
#include <iostream>
#include <igraph/igraph.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <atio.hpp>

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::SparseMatrix<double, RowMajor> SpMatCSR;
typedef Eigen::Triplet<double> Tri;


int pajek2igraph(FILE *f, igraph_t * dst_graph){
  char label[20];
  int N, E;
  int vertex_id, vertex_id2, vertex_id0;
  double edge_weight;

  igraph_vector_t vertices, edges, weights;


  // Read vertices
  fscanf(f, "%s %d", label, &N);
  igraph_vector_init(&vertices, N);
  // Read first (assume monotonous)
  fscanf(f, "%d %s", &vertex_id0, label);
  igraph_vector_set(&vertices, 0, 0.);
  for (long int i = 1; i < N; i++){
    fscanf(f, "%d %s", &vertex_id, label);
    igraph_vector_set(&vertices, i, vertex_id - vertex_id0);
  }

  // Initialize empty directed graph
  igraph_empty(dst_graph, N, IGRAPH_DIRECTED);

  // Read Edges
  fscanf(f, "%s %d", label, &E);
  igraph_vector_init(&edges, E * 2);
  igraph_vector_init(&weights, E);

  for (long int i = 0; i < E; i++){
    fscanf(f, "%d %d %lf", &vertex_id, &vertex_id2, &edge_weight);
    igraph_vector_set(&edges, i*2, vertex_id - vertex_id0);
    igraph_vector_set(&edges, i*2+1, vertex_id2 - vertex_id0);
    igraph_vector_set(&weights, i, edge_weight);
  }
  
  // Add edges
  igraph_add_edges(dst_graph, &edges, 0);

  // Add weight attribute
  igraph_cattribute_EAN_setv(dst_graph, "weight", &weights);
  
  // Clean-up
  fclose(f);  
  igraph_vector_destroy(&edges);
  igraph_vector_destroy(&vertices);
    
  return 0;
}


int pajek2igraphNoVertices(FILE *f, igraph_t * dst_graph, int vertex_id0){
  char label[20];
  int N, E;
  int vertex_id, vertex_id2;
  double edge_weight;

  igraph_vector_t vertices, edges, weights;


  // Read vertices
  fscanf(f, "%s %d", label, &N);
  igraph_vector_init(&vertices, N);
  // Read first (assume monotonous)
  for (long int i = 0; i < N; i++){
    igraph_vector_set(&vertices, i, i);
  }

  // Initialize empty directed graph
  igraph_empty(dst_graph, N, IGRAPH_DIRECTED);

  // Read Edges
  fscanf(f, "%s %d", label, &E);
  igraph_vector_init(&edges, E * 2);
  igraph_vector_init(&weights, E);

  for (long int i = 0; i < E; i++){
    fscanf(f, "%d %d %lf", &vertex_id, &vertex_id2, &edge_weight);
    igraph_vector_set(&edges, i*2, vertex_id - vertex_id0);
    igraph_vector_set(&edges, i*2+1, vertex_id2 - vertex_id0);
    igraph_vector_set(&weights, i, edge_weight);
  }
  
  // Add edges
  igraph_add_edges(dst_graph, &edges, 0);

  // Add weight attribute
  igraph_cattribute_EAN_setv(dst_graph, "weight", &weights);
  
  // Clean-up
  fclose(f);  
  igraph_vector_destroy(&edges);
  igraph_vector_destroy(&vertices);
    
  return 0;
}


int pajek2igraphSym(FILE *f, igraph_t * dst_graph){
  char label[20];
  int N, E;
  int vertex_id, vertex_id2, vertex_id0;
  double edge_weight;

  igraph_vector_t vertices, edges, weights;


  // Read vertices
  fscanf(f, "%s %d", label, &N);
  igraph_vector_init(&vertices, N);
  // Read first (assume monotonous)
  fscanf(f, "%d %s", &vertex_id0, label);
  igraph_vector_set(&vertices, 0, 0.);
  for (long int i = 1; i < N; i++){
    fscanf(f, "%d %s", &vertex_id, label);
    igraph_vector_set(&vertices, i, vertex_id - vertex_id0);
  }

  // Initialize empty directed graph
  igraph_empty(dst_graph, N, IGRAPH_UNDIRECTED);

  // Read Edges
  fscanf(f, "%s %d", label, &E);
  igraph_vector_init(&edges, E * 2);
  igraph_vector_init(&weights, E);

  for (long int i = 0; i < E; i++){
    fscanf(f, "%d %d %lf", &vertex_id, &vertex_id2, &edge_weight);
    igraph_vector_set(&edges, i*2, vertex_id - vertex_id0);
    igraph_vector_set(&edges, i*2+1, vertex_id2 - vertex_id0);
    igraph_vector_set(&weights, i, edge_weight);
  }
  
  // Add edges
  igraph_add_edges(dst_graph, &edges, 0);

  // Add weight attribute
  igraph_cattribute_EAN_setv(dst_graph, "weight", &weights);
  
  // Clean-up
  fclose(f);  
  igraph_vector_destroy(&edges);
  igraph_vector_destroy(&vertices);
    
  return 0;
}


SpMat * pajek2EigenSparse(FILE *f){
  char label[20];
  int N, E;
  int i, j, i0;
  double x;

  vector<Tri> tripletList;

  // Read vertices
  fscanf(f, "%s %d", label, &N);

  // Define sparse matrix
  SpMat *T = new SpMat(N, N);

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


void EigenSparse2Pajek(FILE *f, SpMatCSR *P){
  int N = P->rows();
  int E = P->nonZeros();

  // Write vertices
  fprintf(f, "*Vertices %d\n", N);
  for (int k = 0; k < N; k++)
    fprintf(f, "%d ""%d""\n", k, k);

  // Write Edges
  fprintf(f, "Edges %d\n", E);
  for (int i = 0; i < P->rows(); i++)
    for (SpMatCSR::InnerIterator it(*P, i); it; ++it)
      fprintf(f, "%d %d %lf\n", i, it.col(), it.value());

  return;
}


SpMat * igraph2EigenSparse(igraph_t *srcGraph)
{
  igraph_integer_t N, E;
  igraph_vector_t edges, weights;

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

  // Get triplet list from igraph graph
  vector<Tri> tripletList;
  for (int k = 0; k < E; k++)
    tripletList.push_back(Tri(VECTOR(edges)[k], VECTOR(edges)[E+k],
			      VECTOR(weights)[k]));

  // Define sparse matrix
  SpMat *T = new SpMat(N, N);
  // Assign matrix elements
  T->setFromTriplets(tripletList.begin(), tripletList.end());

  igraph_vector_destroy(&edges);

  return T;
}


int array2igraph(FILE *f, int N, igraph_t * dst_graph){
  double edge_weight;
  int i, j;

  igraph_vector_t vertices, edges, weights;

  // Initialize empty directed graph
  igraph_vector_init(&vertices, N);
  igraph_vector_init(&edges, N*N*2);
  igraph_vector_init(&weights, N*N);
  igraph_empty(dst_graph, N, IGRAPH_DIRECTED);

  for (i = 0; i < N; i++){
    igraph_vector_set(&vertices, i, i);
    for (j = 0; j < N; j++){
      fscanf(f, "%lf", &edge_weight);
      igraph_vector_set(&edges, (i*N+j)*2, i);
      igraph_vector_set(&edges, (i*N+j)*2+1, j);
      igraph_vector_set(&weights, i*N+j, edge_weight);
    }
  }
  
  // Add edges
  igraph_add_edges(dst_graph, &edges, 0);
  // Add weight attribute
  igraph_cattribute_EAN_setv(dst_graph, "weight", &weights);

  // Clean-up
  fclose(f);  
  igraph_vector_destroy(&edges);
  igraph_vector_destroy(&vertices);
    
  return 0;
}

void addCol2Col(SpMat *T, int j_src, int j_dst)
{
  stack<double> insertValue;
  stack<int> insertRow;

  // Iterate over elements of column j_src
  for (SpMat::InnerIterator it_src(*T, j_src); it_src; ++it_src){
    // Try to find element (i_src = it_src.row(), j_dst)
    SpMat::InnerIterator it_dst(*T, j_dst);
    for (; it_dst && (it_dst.row() < it_src.row()); ++it_dst);
    if (it_dst && (it_dst.row() == it_src.row()))
      it_dst.valueRef() = it_dst.value() + it_src.value();
    else{
      insertValue.push(it_src.value());
      insertRow.push(it_src.row());
    }
  }
  while (insertValue.size() > 0){
    T->insert(insertRow.top(), j_dst) = insertValue.top();
    insertValue.pop();
    insertRow.pop();
  }

  return;
}
	  
    
void addCol2ColTriplet(SpMat *T, int jSrc, int jDst)
{
  // Number of columns
  int N = T->cols();
  // Triplets
  vector<Tri> tripletT, tripletInsert;

  // Reserve space for a maximum number of insertions
  tripletInsert.reserve(N);

  // Iterate over elements of column jSrc
  for (SpMat::InnerIterator itSrc(*T, jSrc); itSrc; ++itSrc){
    // Try to find element (i_src = itSrc.row(), jDst)
    SpMat::InnerIterator itDst(*T, jDst);
    for (; itDst && (itDst.row() < itSrc.row()); ++itDst);
    if (itDst && (itDst.row() == itSrc.row()))
      itDst.valueRef() += itSrc.value();
    else
      tripletInsert.push_back(Tri(itSrc.row(), jDst, itSrc.value()));
  }

  // Get original triplet list with modifications of existing elements
  tripletT = EigenSparse2Triplet(T);

  // Reserve space for the elements to insert to triplet list
  tripletT.reserve(tripletT.size() + tripletInsert.size());

  // Insert new elements to triplet list
  for (unsigned int alpha = 0; alpha < tripletInsert.size(); alpha++)
    tripletT.push_back(tripletInsert.at(alpha));

  // Update matrix
  T->setFromTriplets(tripletT.begin(), tripletT.end());

  return;
}
	  
    
void addRow2Row(SpMat *T, int i_src, int i_dst)
{
  double val;
  bool flag;
  stack<double> insertValue;
  stack<int> insertCol;
  for (int j = 0; j < T->outerSize(); j++){
    flag = false;
    // Try to find element (i_src, j)
    SpMat::InnerIterator it(*T, j);
    for(; it && (it.row() < i_src); ++it);
    if (it && (it.row() == i_src)){
      val = it.value();
      flag = true;
    }

    // If (i_src, j) exists
    if (flag){
      // Try to find element (i_dst, j)
      SpMat::InnerIterator it(*T, j);
      for (; it && (it.row() < i_dst); ++it);
      // If element exists, just add
      if (it && (it.row() == i_dst)){
	it.valueRef() += val;
      }
      // Otherwise insert it
      else{
	insertValue.push(val);
	insertCol.push(it.col());
      }
    }
  } 
  while (insertValue.size() > 0){
    T->insert(i_dst, insertCol.top()) = insertValue.top();
    insertValue.pop();
    insertCol.pop();
  }

  return;
}
	  

void addRow2RowTriplet(SpMat *T, int iSrc, int iDst)
{
  // Number of rows
  int N = T->rows();
  // Triplets
  vector<Tri> tripletT, tripletInsert;

  // Reserve space for a maximum number of insertions
  tripletInsert.reserve(N);

  for (int j = 0; j < T->outerSize(); j++){
    // Try to find element (iSrc, j)
    SpMat::InnerIterator itSrc(*T, j);
    for(; itSrc && (itSrc.row() < iSrc); ++itSrc);
    if (itSrc && (itSrc.row() == iSrc)){
      // (iSrc, j) found, now try to find element (iDst, j)
      SpMat::InnerIterator itDst(*T, j);
      for (; itDst && (itDst.row() < iDst); ++itDst);
      // If element exists, just add
      if (itDst && (itDst.row() == iDst))
	itDst.valueRef() += itSrc.value();
      // Otherwise insert it
      else
	tripletInsert.push_back(Tri(iDst, j, itSrc.value()));
    }
  } 

  // Get original triplet list with modifications of existing elements
  tripletT = EigenSparse2Triplet(T);

  // Reserve space for the elements to insert to triplet list
  tripletT.reserve(tripletT.size() + tripletInsert.size());

  // Insert new elements to triplet list
  for (unsigned int alpha = 0; alpha < tripletInsert.size(); alpha++)
    tripletT.push_back(tripletInsert.at(alpha));

  // Update matrix
  T->setFromTriplets(tripletT.begin(), tripletT.end());

  return;
}
	  

void addRow2RowTriplet(SpMat *T, int iSrc, int iDst, SpMatCSR *TCSR)
{
  // Number of rows
  int N = T->rows();
  // Triplets
  vector<Tri> tripletT, tripletInsert;

  // Reserve space for a maximum number of insertions
  tripletInsert.reserve(N);

  // Iterate the elevents of row jSrc
  for(SpMat::InnerIterator itSrc(*TCSR, iSrc); itSrc; ++itSrc){
    //Try to find element (iDst, iSrc == itSrc.col())
    SpMat::InnerIterator itDst(*TCSR, iDst);
    for (; itDst && (itDst.col() < itSrc.col()); ++itDst);
    if (itDst && (itDst.col() == itSrc.col()))
      itDst.valueRef() += itSrc.value();
    // Otherwise insert it
    else
      tripletInsert.push_back(Tri(iDst, itSrc.col(), itSrc.value()));
  } 

  // Get original triplet list with modifications of existing elements
  tripletT = EigenSparse2Triplet(T);

  // Reserve space for the elements to insert to triplet list
  tripletT.reserve(tripletT.size() + tripletInsert.size());

  // Insert new elements to triplet list
  for (unsigned int alpha = 0; alpha < tripletInsert.size(); alpha++)
    tripletT.push_back(tripletInsert.at(alpha));

  // Update matrix
  T->setFromTriplets(tripletT.begin(), tripletT.end());

  return;
}
	  

void scalProdInner(SpMat *T, int outer, double coef)
{
  for (SpMat::InnerIterator it(*T, outer); it; ++it)
    it.valueRef() *= coef;

  return;
}
     

void scalProdOuter(SpMat *T, int inner, double coef)
{
  for (int j = 0; j < T->outerSize(); j++)
    for (SpMat::InnerIterator it(*T, j); it; ++it)
      if (it.index() == inner)
	it.valueRef() *= coef;

  return;
}


double getModularity(SpMat *T)
{
  int N = T->innerSize();
  double modularity;
  VectorXd self = VectorXd::Zero(T->innerSize());

  // Gets strength and self-loops
  for (int j = 0; j < T->outerSize(); j++){
    for (SpMat::InnerIterator it(*T, j); it; ++it){
      if (it.index() == j)
	self(j) = it.value();
    }
  }

  // Compute modularity
  modularity = (self.sum() - 1) / N;

  return modularity;
}


#endif
