#ifndef SDESOLVERS_HPP
#define SDESOLVERS_HPP

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ODESolvers.hpp>


// Declarations

gsl_matrix * generateCuspAdditiveWienerEM(gsl_vector *, double, double,
					  gsl_matrix *, double,
					  double, double, int, double);
gsl_matrix * generateCuspAdditiveWienerEM(gsl_vector *, double,
					  gsl_vector *,
					  gsl_matrix *, double,
					  double, double, int, double);
gsl_vector * cuspAdditiveWienerEM(gsl_vector *, double, double,
				  gsl_vector *, double, double);
gsl_matrix * generateLorenzLinearWienerEM(gsl_vector *, double, double, double,
					  gsl_matrix *, double,
					  double, double, int, double);
gsl_vector * lorenzLinearWienerEM(gsl_vector *, double, double, double,
				  gsl_vector *, double, double);
gsl_vector * additiveWienerField(double, gsl_vector *);
gsl_vector * linearWienerField(gsl_vector *, double, gsl_vector *);


// Definitions

// Euler-Maruyama integaration  of the normal form of cusp bifurcation with additive noise
gsl_matrix * generateCuspAdditiveWienerEM(gsl_vector *state,
					  double r, double h,
					  gsl_matrix *noiseSamples, double Q,
					  double length, double dt,
					  int sampling, double spinup)
{
  size_t nt = length / dt;
  size_t ntSpinup = spinup / dt;
  size_t dim = state->size;
  gsl_matrix *data = gsl_matrix_alloc((size_t) (nt/sampling), dim);
  gsl_vector *newState, *initState, *noiseSample;

  initState = gsl_vector_alloc(dim);
  gsl_vector_memcpy(initState, state);
  
  // Get spinup
  for (size_t i = 1; i <= ntSpinup; i++){
    // Get noise sample
    noiseSample = gsl_vector_alloc(dim);
    gsl_matrix_get_row(noiseSample, noiseSamples, i);
    
    // Get new state
    newState = cuspAdditiveWienerEM(initState, r, h,
				    noiseSample, Q,
				    dt);

    gsl_vector_memcpy(initState, newState);
    gsl_vector_free(newState);
    gsl_vector_free(noiseSample);
  }
  
  // Get record
  for (size_t i = 1; i <= nt; i++){
    // Get noise sample
    noiseSample = gsl_vector_alloc(dim);
    gsl_matrix_get_row(noiseSample, noiseSamples, ntSpinup+i);
    
    // Get new state
    newState = cuspAdditiveWienerEM(initState, r, h,
				    noiseSample, Q, dt);

    // Save new state
    if (i%sampling == 0)
      gsl_matrix_set_row(data, i/sampling-1, newState);

    gsl_vector_memcpy(initState, newState);
    gsl_vector_free(newState);
    gsl_vector_free(noiseSample);
  }
  gsl_vector_free(initState);
  
  return data;
}


// Euler-Maruyama integaration  of the normal form of cusp bifurcation with additive noise with transient parameter h
gsl_matrix * generateCuspAdditiveWienerEM(gsl_vector *state,
					  double r, gsl_vector *hTransient,
					  gsl_matrix *noiseSamples, double Q,
					  double length, double dt,
					  int sampling, double spinup)
{
  size_t nt = length / dt;
  size_t ntSpinup = spinup / dt;
  size_t dim = state->size;
  gsl_matrix *data = gsl_matrix_alloc((size_t) (nt/sampling), dim);
  gsl_vector *newState, *initState, *noiseSample;
  double h;

  initState = gsl_vector_alloc(dim);
  gsl_vector_memcpy(initState, state);
  
  // Get spinup
  for (size_t i = 1; i <= ntSpinup; i++){
    // Get noise sample
    noiseSample = gsl_vector_alloc(dim);
    gsl_matrix_get_row(noiseSample, noiseSamples, i);

    // Get transient parameter value
    h = gsl_vector_get(hTransient, i);
    
    // Get new state
    newState = cuspAdditiveWienerEM(initState, r, h,
				    noiseSample, Q,
				    dt);

    gsl_vector_memcpy(initState, newState);
    gsl_vector_free(newState);
    gsl_vector_free(noiseSample);
  }
  
  // Get record
  for (size_t i = 1; i <= nt; i++){
    // Get noise sample
    noiseSample = gsl_vector_alloc(dim);
    gsl_matrix_get_row(noiseSample, noiseSamples, ntSpinup+i);
    
    // Get transient parameter value
    h = gsl_vector_get(hTransient, i);
    
    // Get new state
    newState = cuspAdditiveWienerEM(initState, r, h,
				    noiseSample, Q, dt);

    // Save new state
    if (i%sampling == 0)
      gsl_matrix_set_row(data, i/sampling-1, newState);

    gsl_vector_memcpy(initState, newState);
    gsl_vector_free(newState);
    gsl_vector_free(noiseSample);
  }
  gsl_vector_free(initState);
  
  return data;
}


// Euler-Maruyama step of the normal form of cusp bifurcation with additive noise
gsl_vector * cuspAdditiveWienerEM(gsl_vector *state,
				  double r, double h,
				  gsl_vector *noiseSample, double Q,
				  double dt)
{
  size_t dim = state->size;
  gsl_vector *field, *newState;

  newState = gsl_vector_alloc(dim);
  gsl_vector_memcpy(newState, state);
    
  field = cuspField(state, r, h);
  gsl_vector_scale(field, dt);
  gsl_vector_add(newState, field);
  gsl_vector_free(field);
  
  field = additiveWienerField(Q, noiseSample);
  gsl_vector_scale(field, sqrt(dt));
  gsl_vector_add(newState, field);
  gsl_vector_free(field);

  return newState;
}


// Euler-Maruyama integaration  of the Lorenz with linear multiplicative noise
gsl_matrix * generateLorenzLinearWienerEM(gsl_vector *state, double rho,
					  double sigma, double beta,
					  gsl_matrix *noiseSamples, double Q,
					  double length, double dt,
					  int sampling, double spinup)
{
  size_t nt = length / dt;
  size_t ntSpinup = spinup / dt;
  size_t dim = state->size;
  gsl_matrix *data = gsl_matrix_alloc((size_t) (nt/sampling), dim);
  gsl_vector *newState, *initState, *noiseSample;

  initState = gsl_vector_alloc(dim);
  gsl_vector_memcpy(initState, state);
  
  // Get spinup
  for (size_t i = 1; i <= ntSpinup; i++){
    // Get noise sample
    noiseSample = gsl_vector_alloc(dim);
    gsl_matrix_get_row(noiseSample, noiseSamples, i);
    
    // Get new state
    newState = lorenzLinearWienerEM(initState, rho, sigma, beta,
			       noiseSample, Q,
			       dt);

    gsl_vector_memcpy(initState, newState);
    gsl_vector_free(newState);
    gsl_vector_free(noiseSample);
  }
  
  // Get record
  for (size_t i = 1; i <= nt; i++){
    // Get noise sample
    noiseSample = gsl_vector_alloc(dim);
    gsl_matrix_get_row(noiseSample, noiseSamples, ntSpinup+i);
    
    // Get new state
    newState = lorenzLinearWienerEM(initState, rho, sigma, beta,
			       noiseSample, Q, dt);

    // Save new state
    if (i%sampling == 0)
      gsl_matrix_set_row(data, i/sampling-1, newState);

    gsl_vector_memcpy(initState, newState);
    gsl_vector_free(newState);
    gsl_vector_free(noiseSample);
  }
  gsl_vector_free(initState);
  
  return data;
}


// Euler-Maruyama step of the Lorenz with linear multiplicative noise
gsl_vector * lorenzLinearWienerEM(gsl_vector *state,
				  double rho, double sigma, double beta,
				  gsl_vector *noiseSample, double Q,
				  double dt)
{
  size_t dim = state->size;
  gsl_vector *field, *newState;

  newState = gsl_vector_alloc(dim);
  gsl_vector_memcpy(newState, state);
    
  field = lorenzField(state, rho, sigma, beta);
  gsl_vector_scale(field, dt);
  gsl_vector_add(newState, field);
  gsl_vector_free(field);
  
  field = linearWienerField(state, Q, noiseSample);
  gsl_vector_scale(field, sqrt(dt));
  gsl_vector_add(newState, field);
  gsl_vector_free(field);

  return newState;
}

// Additive Wiener process vector field
gsl_vector * additiveWienerField(double Q, gsl_vector *noiseSample)
{
  size_t dim = noiseSample->size;
  gsl_vector *field = gsl_vector_alloc(dim);

  gsl_vector_memcpy(field, noiseSample);
  gsl_vector_scale(field, Q);
  
  return field;
}

// Linear Wiener process vector field
gsl_vector * linearWienerField(gsl_vector *state, double Q, gsl_vector *noiseSample)
{
  size_t dim = noiseSample->size;
  gsl_vector *field = gsl_vector_alloc(dim);

  gsl_vector_memcpy(field, noiseSample);
  gsl_vector_scale(field, Q);
  gsl_vector_mul(field, state);
  
  return field;
}

#endif
