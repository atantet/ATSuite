#ifndef ODESOLVERS_HPP
#define ODESOLVERS_HPP

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

// Declarations
gsl_matrix * generateCuspEuler(gsl_vector *, double, double, double, double,
			       int, double);
gsl_vector * cuspEuler(gsl_vector *, double, double, double);
gsl_vector * cuspField(gsl_vector *, double, double);
gsl_matrix * generateLorenzRK4(gsl_vector *, double, double, double, double,
			       double, int, double);
gsl_vector * lorenzRK4(gsl_vector *, double, double, double, double);
gsl_vector * lorenzField(gsl_vector *, double, double, double);


// Definitions

gsl_matrix * generateCuspEuler(gsl_vector *state, double r, double h,
			       double length, double dt, int sampling,
			       double spinup){
  size_t nt = length / dt;
  size_t ntSpinup = spinup / dt;
  size_t dim = state->size;
  gsl_matrix *data = gsl_matrix_alloc((size_t) (nt/sampling), dim);
  gsl_vector *res, *init;

  init = gsl_vector_alloc(dim);
  gsl_vector_memcpy(init, state);
  // Get spinup
  for (size_t i = 1; i <= ntSpinup; i++){
    res = cuspEuler(init, r, h, dt);
    gsl_vector_memcpy(init, res);
    gsl_vector_free(res);
  }
  // Get record
  for (size_t i = ntSpinup+1; i <= nt; i++){
    res = cuspEuler(init, r, h, dt);
    if (i%sampling == 0)
      gsl_matrix_set_row(data, (i-ntSpinup)/sampling-1, res);
    gsl_vector_memcpy(init, res);
    gsl_vector_free(res);
  }
  gsl_vector_free(init);
  
  return data;
}

gsl_vector * cuspEuler(gsl_vector *state,
		       double r, double h,
		       double dt)
{
  size_t dim = state->size;
  gsl_vector *newState = gsl_vector_calloc(dim);

  newState = cuspField(state, r, h);
  gsl_vector_scale(newState, dt);
  gsl_vector_add(newState, state);

  return newState;
}


gsl_vector * cuspField(gsl_vector *state,
		       double r, double h)
{
  size_t dim = state->size;
  gsl_vector *field = gsl_vector_alloc(dim);

  // Fx = sigma * (y - x)
  gsl_vector_set(field, 0, h + r * gsl_vector_get(state, 0) 
		 - pow(gsl_vector_get(state, 0), 3));
  
  return field;
}


gsl_matrix * generateLorenzRK4(gsl_vector *state,
			       double rho, double sigma, double beta,
			       double length, double dt, int sampling,
			       double spinup)
{
  size_t nt = length / dt;
  size_t ntSpinup = spinup / dt;
  size_t dim = state->size;
  gsl_matrix *data = gsl_matrix_alloc((size_t) (nt/sampling), dim);
  gsl_vector *res, *init;

  init = gsl_vector_alloc(dim);
  gsl_vector_memcpy(init, state);
  // Get spinup
  for (size_t i = 1; i <= ntSpinup; i++){
    res = lorenzRK4(init, rho, sigma, beta, dt);
    gsl_vector_memcpy(init, res);
    gsl_vector_free(res);
  }
  // Get record
  for (size_t i = ntSpinup+1; i <= nt; i++){
    res = lorenzRK4(init, rho, sigma, beta, dt);
    if (i%sampling == 0)
      gsl_matrix_set_row(data, (i-ntSpinup)/sampling-1, res);
    gsl_vector_memcpy(init, res);
    gsl_vector_free(res);
  }
  gsl_vector_free(init);
  
  return data;
}


gsl_vector * lorenzRK4(gsl_vector *state,
		       double rho, double sigma, double beta,
		       double dt)
{
  size_t dim = state->size;
  gsl_vector *k1, *k2, *k3, *k4;
  gsl_vector *tmp = gsl_vector_calloc(dim);

  k1 = lorenzField(state, rho, sigma, beta);
  gsl_vector_scale(k1, dt);
  
  gsl_vector_memcpy(tmp, k1);
  gsl_vector_scale(tmp, 0.5);
  gsl_vector_add(tmp, state);
  k2 = lorenzField(tmp, rho, sigma, beta);
  gsl_vector_scale(k2, dt);

  gsl_vector_memcpy(tmp, k2);
  gsl_vector_scale(tmp, 0.5);
  gsl_vector_add(tmp, state);
  k3 = lorenzField(tmp, rho, sigma, beta);
  gsl_vector_scale(k3, dt);

  gsl_vector_memcpy(tmp, k3);
  gsl_vector_add(tmp, state);
  k4 = lorenzField(tmp, rho, sigma, beta);
  gsl_vector_scale(k4, dt);

  gsl_vector_scale(k2, 2);
  gsl_vector_scale(k3, 2);
  gsl_vector_memcpy(tmp, k1);
  gsl_vector_add(tmp, k2);
  gsl_vector_add(tmp, k3);
  gsl_vector_add(tmp, k4);
  gsl_vector_scale(tmp, 1. / 6);
  gsl_vector_add(tmp, state);

  gsl_vector_free(k1);
  gsl_vector_free(k2);
  gsl_vector_free(k3);
  gsl_vector_free(k4);

  return tmp;
}


gsl_vector * lorenzField(gsl_vector *state,
			 double rho, double sigma, double beta)
{
  size_t dim = state->size;
  gsl_vector *field = gsl_vector_alloc(dim);

  // Fx = sigma * (y - x)
  gsl_vector_set(field, 0, sigma * (gsl_vector_get(state, 1) - gsl_vector_get(state, 0)));
  // Fy = x * (rho - z) - y
  gsl_vector_set(field, 1, gsl_vector_get(state, 0)
		 * (rho - gsl_vector_get(state, 2)) - gsl_vector_get(state, 1));
  // Fz = x*y - beta*z
  gsl_vector_set(field, 2, gsl_vector_get(state, 0) * gsl_vector_get(state, 1)
		 - beta * gsl_vector_get(state, 2));
  
  return field;
}

#endif
