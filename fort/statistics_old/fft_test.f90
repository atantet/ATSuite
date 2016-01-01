program fft_test
  use, intrinsic :: iso_c_binding
  implicit none
  include 'fftw3.f03'
  
  type(C_PTR) :: plan, p
  integer, parameter :: n = 900
  integer, parameter :: nfft = 2**(ceiling(log(real(n))/log(2.)))
  
  real(C_DOUBLE), dimension(n) :: in, in2
  real(C_DOUBLE), dimension(:), pointer :: in_tap, in2_tap
  complex(C_DOUBLE_COMPLEX), dimension(:), pointer :: out, out2

  p = fftw_alloc_real(int(nfft, C_SIZE_T))
  call c_f_pointer(p, in_tap, [nfft])
  call c_f_pointer(p, in2_tap, [nfft])
  p = fftw_alloc_complex(int((nfft/2+1), C_SIZE_T))
  call c_f_pointer(p, out, [nfft/2+1])

  call random_number(in)
  in = in - sum(in) / n
  in = in / sqrt(sum(in**2) / n)
  in_tap(:) = 0.
  in_tap(1:n) = in

  plan = fftw_plan_dft_r2c_1d(nfft, in_tap, out, FFTW_ESTIMATE)
  call fftw_execute_dft_r2c(plan, in_tap, out)
  call fftw_destroy_plan(plan)
  out = out / sqrt(real(nfft))

  plan = fftw_plan_dft_c2r_1d(nfft, out, in2_tap, FFTW_ESTIMATE)
  call fftw_execute_dft_c2r(plan, out, in2_tap)
  call fftw_destroy_plan(plan)
  in2_tap = in2_tap / sqrt(real(nfft))

  in2 = in2_tap(1:n)
  
  print *, nfft
  print *, 1. * sqrt(sum((in - in2)**2)) / n

  call fftw_free(p)
end program fft_test
