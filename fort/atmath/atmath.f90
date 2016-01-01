module atmath
  implicit none
  interface pearson_corr
     module procedure  pearson_corr_vect, pearson_corr_mat
  end interface pearson_corr

  interface auto_correlation
     module procedure auto_correlation, auto_correlation_msg
   end interface auto_correlation
   real, external :: sdot, snrm2
contains
  ! Pearson correlation on vectors
  function pearson_corr_vect(x, y) result(R)
    implicit none
    ! dummy arguments
    real, dimension(:), intent (in) :: x, y
    ! function result
    real :: R
    ! local variables
    real, dimension(size(x, 1)) :: xa, ya ! x, y anomalies
    integer :: l ! problem dimension
    real dotxy, nrmx, nrmy
    ! Regularize the unusual case of complete correlation
    ! see Press&Teukolsky 90

    ! Get problem dimensions and allocate
    l = size(x)

    ! Remove the means
    xa = x -  sum(x) / l
    ya = y -  sum(y) / l
        
    ! Calculate correlation
    dotxy = sdot(l, xa, 1, ya, 1)
    nrmx = snrm2(l, xa, 1)
    nrmy = snrm2(l, ya, 1)

    R = dotxy / (nrmx * nrmy)
  end function pearson_corr_vect

  function pearson_corr_mat(x, y) result(R)
    ! Pearson correlation on matrices
    implicit none
    ! dummy arguments
    real, dimension(:, :), intent (in) :: x, y
    ! function result
    real, dimension(size(x, 2), size(y, 2)) :: R
    ! local variables
    real, dimension(size(x, 1), size(x, 2)) :: xa
    real, dimension(size(y, 1), size(y, 2)) :: ya
    integer :: l, nx, ny ! problem dimension

    ! Get problem dimensions and allocate
    l = size(x, 1)
    nx = size(x, 2)
    ny = size(y, 2)

    ! Standardize
    xa = standardize(x)
    ya = standardize(y)

    ! Calculate sample correlation matrix
    call sGEMM('T', 'N', nx, ny, l, 1., xa, l, ya, l, 0, R, nx)
  end function pearson_corr_mat

  function pearson_cov_mat(x, y) result(R)
    ! Pearson covariance on matrices
    implicit none
    ! dummy arguments
    real, dimension(:, :), intent (in) :: x, y
    ! function result
    real, dimension(size(x, 2), size(y, 2)) :: R
    ! local variables
    integer :: l, nx, ny ! problem dimension

    ! Get problem dimensions and allocate
    l = size(x, 1)
    nx = size(x, 2)
    ny = size(y, 2)

    ! Calculate sample correlation matrix
    call sGEMM('T', 'N', nx, ny, l, 1., x, l, y, l, 0, R, nx)
    R = R / (l - 1)
    
  end function pearson_cov_mat

  function standardize(x) result(xa)
    implicit none
    real, dimension(:, :), intent(in) :: x
    real, dimension(size(x, 1), size(x, 2)) :: xa
    integer :: l

    l = size(x, 1)
    xa = x -  spread(sum(x, dim=1) / l, 1, l)
    xa = xa / spread(sqrt(sum(xa**2, dim=1) / l), 1, l)

  end function standardize

  function auto_correlation(x, lag_in)
    implicit none
    real, dimension(:), intent(in) :: x
    integer, optional, intent(in) :: lag_in
    real :: auto_correlation
    integer :: lag
    integer :: n
    real, dimension(:), allocatable :: xp, xn
    real :: dot_xpxn, nrm_xp, nrm_xn
    
    ! Default value of the lag is 0
    if (present(lag_in)) then
       lag = lag_in
    else
       lag = 0
    end if

    n = size(x)

    ! Get previous and next realisations
    allocate(xp(n - lag), xn(n - lag))
    xp = x(:n - lag)
    xn = x(1 + lag:)
    ! Remove means
    xp = xp - sum(xp) / (n - lag)
    xn = xn - sum(xn) / (n - lag)
    ! Calculate auto-correlation from dot products
    nrm_xp = snrm2(n - lag, xp, 1)
    nrm_xn = snrm2(n - lag, xn, 1)
    dot_xpxn = sdot(n - lag, xp, 1, xn, 1)
    
    auto_correlation = dot_xpxn / (nrm_xp * nrm_xn)
    deallocate(xp, xn)
  end function auto_correlation

  function auto_correlation_msg(x, lag_in, msg)
    implicit none
    real, dimension(:), intent(in) :: x
    integer, optional, intent(in) :: lag_in
    real, intent(in) :: msg ! missing value
    integer :: exp_msg
    real, parameter :: tiny = 1.e-6
    real :: auto_correlation_msg
    integer :: lag
    integer :: n, n_v
    real, dimension(:), allocatable :: xp, xn, xp_v, xn_v
    logical, dimension(size(x)) :: mask
    real :: dot_xpxn, nrm_xp, nrm_xn
    
    ! Default value of the lag is 0
    if (present(lag_in)) then
       lag = lag_in
    else
       lag = 0
    end if

    exp_msg = exponent(msg)

    n = size(x)

    ! Get previous and next realisations with missig values
    allocate(xp(n - lag), xn(n - lag))
    xp = x(:n-lag)
    xn = x(1+lag:)
    ! Find realisations valid in both vectors
    mask = (abs(xp - msg) .gt. (tiny * exp_msg)) &
         .and. (abs(xn - msg) .gt. (tiny * exp_msg))
    n_v = count(mask)
    allocate(xp_v(N_v), xn_v(N_v))
    xp_v = pack(xp, mask)
    xn_v = pack(xn, mask)
    ! Remove means
    xp_v = xp_v - sum(xp_v) / n_v
    xn_v = xn_v - sum(xn_v) / n_v
        ! Calculate auto-correlation from dot products
    nrm_xp = snrm2(n_v, xp_v, 1)
    nrm_xn = snrm2(n_v, xn_v, 1)
    dot_xpxn = sdot(n_v, xp_v, 1, xn_v, 1)
    
    auto_correlation_msg = dot_xpxn / (nrm_xp * nrm_xn)
    deallocate(xp, xn, xp_v, xn_v)
  end function auto_correlation_msg

  function logical2int(log) result(int)
    logical, intent(in) :: log
    integer :: int

    if (log) then
       int = 1
    else
       int = 0
    end if
  end function logical2int

  subroutine tri_indices(n, ind_ii, ind_jj)
    ! Return indices of triangular part of square matrix (n, n) columnwise
    integer, intent(in) :: n
    integer, dimension(n * (n - 1) / 2), intent(out) :: ind_ii, ind_jj
    integer :: ii, jj, k

    k = 1
    do jj = 1, n
       do ii = jj + 1, n
          ind_ii(k) = ii
          ind_jj(k) = jj
          k = k + 1
       end do
    end do
  end subroutine tri_indices
  
  recursive subroutine QsortC(A)
    real, intent(in out), dimension(:) :: A
    integer :: iq

    if(size(A) > 1) then
       call Partition(A, iq)
       call QsortC(A(:iq-1))
       call QsortC(A(iq:))
    endif
  end subroutine QsortC

  subroutine Partition(A, marker)
    real, intent(in out), dimension(:) :: A
    integer, intent(out) :: marker
    integer :: i, j
    real :: temp
    real :: x      ! pivot point
    x = A(1)
    i= 0
    j= size(A) + 1

    do
       j = j-1
       do
          if (A(j) <= x) exit
          j = j-1
       end do
       i = i+1
       do
          if (A(i) >= x) exit
          i = i+1
       end do
       if (i < j) then
          ! exchange A(i) and A(j)
          temp = A(i)
          A(i) = A(j)
          A(j) = temp
       elseif (i == j) then
          marker = i+1
          return
       else
          marker = i
          return
       endif
    end do

  end subroutine Partition

  function mean(x)
    real, dimension(:), intent(in) :: x
    real :: mean
    integer :: n

    n = size(x)
    mean = sum(x) / n
  end function mean

  function variance(x)
    real, dimension(:), intent(in) :: x
    real :: mean, variance
    integer :: n

    n = size(x)
    mean = sum(x) / n
    variance = sum((x - mean)**2) / n
  end function variance

  
  ! SUBROUTINE fasper(x, y, ofac, hifac, px, py, jmax)
  !   IMPLICIT NONE
  !   REAL, DIMENSION(:), INTENT(IN) :: x, y
  !   REAL, INTENT(IN) :: ofac, hifac
  !   INTEGER, INTENT(OUT) :: jmax
  !   REAL, DIMENSION(:), POINTER :: px, py
  !   INTEGER, PARAMETER :: MACC = 4
  !   INTEGER :: j,k,n,ndim,nfreq,nfreqt,nout
  !   REAL :: ave, ck, ckk, cterm, cwt, den, df, fac, fndim, hc2wt, &
  !        hs2wt,hypo,sterm,swt,var,xdif,xmax,xmin
  !   REAL, DIMENSION(:), ALLOCATABLE :: wk1,wk2
  !   LOGICAL, SAVE :: init=.true.

  !   n = size(x)
  !   if (init) then
  !      init=.false.
  !      nullify(px,py)
  !   else
  !      if (associated(px)) deallocate(px)
  !      if (associated(py)) deallocate(py)
  !   end if
  !   nfreqt = ofac * hifac * n * MACC
  !   nfreq = 64
  !   do ! Size the FFT as next power of 2 above nfreqt.
  !      if (nfreq >= nfreqt) exit
  !      nfreq = nfreq * 2
  !   end do
  !   ndim= 2 * nfreq
  !   allocate(wk1(ndim), wk2(ndim))
  !   ave = mean(y(1:n))
  !   var = variance(y(1:n))
  !   xmax = max(x)
  !   xmin = min(x)
  !   xdif = xmax - xmin
  !   wk1(1:ndim) = 0.0 ! Zero the workspaces.
  !   wk2(1:ndim) = 0.0
  !   fac = ndim / (xdif * ofac)
  !   fndim = ndim
  !   do j = 1, n ! Extirpolate the data into the workspaces.
  !      ck = 1.0 + mod((x(j) - xmin) * fac, fndim)
  !      ckk=1.0 + mod(2.0 * (ck - 1.0), fndim)
  !      call spreadval(y(j) - ave, wk1, ck, MACC)
  !      call spreadval(1.0, wk2, ckk, MACC)
  !   end do
  !   call realft(wk1(1:ndim), 1) ! Take the fast Fourier transforms.
  !   call realft(wk2(1:ndim), 1)
  !   df = 1.0 / (xdif * ofac)
  !   nout = 0.5 * ofac * hifac * n
  !   allocate(px(nout), py(nout))
  !   k = 3
  !   do j=1, nout ! Compute the Lomb value for each frequency.
  !      hypo = sqrt(wk2(k)**2 + wk2(k+1)**2)
  !      hc2wt = 0.5 * wk2(k) / hypo
  !      hs2wt = 0.5 * wk2(k+1) / hypo
  !      cwt = sqrt(0.5 + hc2wt)
  !      swt = sign(sqrt(0.5 - hc2wt), hs2wt)
  !      den = 0.5 * n + hc2wt * wk2(k) + hs2wt * wk2(k+1)
  !      cterm = (cwt * wk1(k) + swt * wk1(k+1))**2 / den
  !      sterm = (cwt * wk1(k+1) - swt * wk1(k))**2 / (n - den)
  !      px(j) = j * df
  !      py(j) = (cterm + sterm) / (2.0 * var)
  !      k = k + 2
  !   end do
  !   deallocate(wk1, wk2)
  !   jmax = imaxloc(py(1:nout))
  ! END SUBROUTINE fasper
  
  ! SUBROUTINE spreadval(y,yy,x,m)
  !   IMPLICIT NONE
  !   REAL, INTENT(IN) :: y,x
  !   REAL, DIMENSION(:), INTENT(INOUT) :: yy
  !   INTEGER, INTENT(IN) :: m
  !   INTEGER :: ihi,ilo,ix,j,nden,n
  !   REAL :: fac
  !   INTEGER, DIMENSION(10) :: nfac = (/ 1,1,2,6,24,120,720,5040,40320,362880 /)
  !   if (m > 10) then
  !      print *, 'factorial table too small in spreadval'
  !      return
  !   end if
  !   n = size(yy)
  !   ix = x
  !   if (x == real(ix)) then
  !      yy(ix) = yy(ix) + y
  !   else
  !      ilo = min(max(int(x - 0.5 * m + 1.0), 1), n - m + 1)
  !      ihi = ilo + m - 1
  !      nden = nfac(m)
  !      fac = product(x - arth(ilo, 1, m))
  !      yy(ihi) = yy(ihi) + y * fac / (nden * (x - ihi))
  !      do j = ihi - 1, ilo, -1
  !         nden = (nden / (j + 1 - ilo)) * (j - ihi)
  !         yy(j) = yy(j) + y * fac / (nden * (x - j))
  !      end do
  !   end if
  ! END SUBROUTINE spreadval

end module atmath

