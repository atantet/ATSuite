#include <stdlib.h>
#include <stdio.h>
#include <boost/algorithm/string.hpp>

int write_pajek(const char *path, int *data[], int N){
  char buffer[80];

  FILE *f;
  if ((f = fopen(path, "w")) == NULL){
    fprintf(stderr, "Could not open Pajek file for writting.");
    return 1;
  }

  // Write vertices
  sprintf(buffer, "*Vertices %d\n", N);
  fputs(buffer, f);
  for (int k = 0; k < N; k++){
    sprintf(buffer, "%d \"%d\"\n", k+1, k+1);
    fputs(buffer, f);
  }

  // Write list of edges
  sprintf(buffer, "*Edges \n");
  fputs(buffer, f);
  for (int i = 0; i < N; i++){
    for (int j = 0; j < N; j++){
      if (data[i][j] > 0){
	// Write raw net
	sprintf(buffer, "%d\t%d\t%d\n", i+1, j+1, data[i][j]);
	if (fputs(buffer, f) == EOF){
	  fprintf(stderr, "Error: EOF while writing pajek file.");
	  return 1;
	}
      }
    }
  }
  fclose(f);  
  return 0;
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
