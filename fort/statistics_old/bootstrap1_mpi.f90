module bootstrap_mpi
  use mpi
  use atmath
  use qsort
  implicit none

  ! Moving Block Bootstrap
  interface mbb_pval
     module procedure mbb_bivar_pval, mbb_bivar_pval_field
  end interface mbb_pval
  
  interface mbb_ci
     module procedure mbb_bivar_ci, mbb_bivar_ci_field
  end interface mbb_ci
  
  ! Resampling of the time series
  interface resample
     module procedure resample_bivar
  end interface resample

contains
  subroutine mbb_bivar_pval_field(X, statistic, sqrtNs, pval, estim, lopt, &
       msg)
    implicit none
    ! mbb_bivar_field_sl gives the p-values pval of estimators 
    ! estim = statistic(X) of the field X  using the mbb with block with
    ! estimated optimal block lengths lopt.
    ! If msg is given, msg values are treated as missing values.

    ! Dummy arguments
    real, dimension(:, :), intent(in) :: X ! Field time series
    integer, intent(in) :: sqrtNs ! Square root of the number of resamples
    interface     ! Function to give to calculate the bivariate statistic
       function statistic(var1, var2)
         real, dimension(:), intent(in) :: var1, var2
         real :: statistic
       end function statistic
    end interface
    ! Optional dummy arguments
    real, optional, intent(in) :: msg ! missing value
    
    ! Outputs: p-value, estimate and optimal block length
    real, dimension(size(x, 2), size(x, 2)), intent(out) :: pval, estim
    integer, dimension(size(x, 2), size(x, 2)), intent(out) :: lopt

    ! Local variables
    integer :: m, n ! problem dimension
    integer :: ii, jj, proc ! running indices
    real, dimension(size(x, 2)) :: ax ! lag-1 auto-correlation
    real, dimension(:), allocatable :: ax_p
    real :: pval_ij, estim_ij
    integer :: lopt_ij
    integer :: ierr ! error status for MPI
    integer :: my_rank ! process MPI-rank
    integer :: np ! number of processes
    integer, dimension(:), allocatable :: stride, nax_p
    integer :: ntasks, rem
    real, dimension(:), allocatable :: ax_p

    ! Init MPI
    call MPI_Init(ierr)
    ! Get process rank
    call MPI_Comm_Rank(MPI_COMM_WORLD, my_rank, ierr)
    ! Get how many process are being used
    call MPI_Comm_Size(MPI_COMM_WORLD, np, ierr)
    if (my_rank .eq. 0) then
       print *, "Number of processes = ", np

    ! Problem dimension
    m = size(x, 1)
    n = size(x, 2)

    ntasks = n / np
    rem = n - ntasks * np
    allocate(nax_p(np))
    do proc = 1, np
       nax_p(proc) = ntasks + int(proc .le. rem)
    end do
    stride(1) = 0
    do proc = 2, np
       stride(proc) = nax_p(proc - 1)
    end do
    allocate(ax_p(nax_p(my_rank + 1)))
    
    ! Calculate auto-correlation at lag 1
    if (present(msg)) then
       do jj = 1, nax_p(my_rank + 1)
          ax(jj) = auto_correlation(X(:, jj), 1, msg)
       end do
    else
       do jj = 1, nax_p(my_rank + 1)
          ax(jj) = auto_correlation(X(:, jj), 1)
       end do
    end if
    ! Let 0 gather the results of the auto-correlation at lag 1
    call MPI_Gatherv(ax_p, nax_p(my_rank + 1), MPI_REAL, &
         ax, nax_p, stride, 0, MPI_COMM_WORLD, ierr)

    deallocate(stride)
    deallocate(nax_p)

    ! Get p-values for each pair of node
    if (present(msg)) then
       do jj = 1, n
          pval(jj, jj) = 0
          estim(jj, jj) = 1
          lopt(jj, jj) = 0
          do ii = jj+1, n
             call mbb_bivar_pval(X(:, ii), X(:, jj), statistic, sqrtNs, &
                  pval_ij, estim_ij, lopt_ij, msg, ax(ii), ax(jj))
             pval(ii, jj) = pval_ij
             pval(jj, ii) = pval_ij
             estim(ii, jj) = estim_ij
             estim(jj, ii) = estim_ij
             lopt(ii, jj) = lopt_ij
             lopt(jj, ii) = lopt_ij
          end do
       end do
    else
       do jj = 1, n
          pval(jj, jj) = 0
          estim(jj, jj) = 1
          lopt(jj, jj) = 0
          do ii = ii+1, n
             call mbb_bivar_pval(X(:, ii), X(:, jj), statistic, sqrtNs, &
                  pval_ij, estim_ij, lopt_ij, ax_in=ax(ii), ay_in=ax(jj))
             pval(ii, jj) = pval_ij
             pval(jj, ii) = pval_ij
             estim(ii, jj) = estim_ij
             estim(jj, ii) = estim_ij
             lopt(ii, jj) = lopt_ij
             lopt(jj, ii) = lopt_ij
          end do
       end do
    end if
    
    ! Finalize MPI
    call MPI_Finalize(ierr)

  end subroutine mbb_bivar_pval_field
    
  subroutine mbb_bivar_pval(x, y, statistic, sqrtNs, pval, estim, lopt, &
       msg, ax_in, ay_in)
    implicit none
    ! mbb_bivar_sl gives the p-value pval of an estimator 
    ! estim = statistic(x, y) using the mbb with block with
    ! estimated optimal block length lopt.
    ! If msg is given, msg values are treated as missing values.
    ! If ax_in and ay_in are given, they are used as lag-1 auto-correlations
    ! to estimate the optimval block length.

    ! Dummy arguments
    real, dimension(:), intent(in) :: x, y ! Random vectors
    integer, intent(in) :: sqrtNs ! square root of the number of resamples
    interface     ! Function to give to calculate the bivariate statistic
       function statistic(var1, var2)
         real, dimension(:), intent(in) :: var1, var2
         real :: statistic
       end function statistic
    end interface

    ! Optional dummy arguments
    real, optional, intent(in) :: msg ! missing value
    real, optional, intent(in) :: ax_in, ay_in ! dummy lag-1 auto-correlation
    
    ! Outputs: p-value, estimate and optimal block length
    real, intent(out) :: pval, estim
    integer, intent(out) :: lopt

    ! Local variables
    integer :: exp_msg ! exponent of the missing value
    real, parameter :: tiny = 1.e-6 ! precision of the test of msg value
    real :: ax, ay ! lag-1 auto-correlation
    logical, dimension(size(x)) :: mask ! mask of missing values
    real, dimension(:), allocatable :: x_v, y_v ! valid values of random vectors x, y
    real, dimension(:, :), allocatable :: xb, yb ! resamples of x, y
    integer :: n, n_v ! number of realisations and valid realisations
    real, dimension(sqrtNs**2) :: set ! set of estimates as vector
    integer :: k
    
    ! Number of realisations
    n = size(x)

    if ((.not. present(ax_in)) .or. (.not. present(ay_in))) then
       ! Calculate lag-1-auto-correlation of x and y
       if (present(msg)) then
          ax = auto_correlation(x, 1, msg)
          ay = auto_correlation(y, 1, msg)
       else
          ax = auto_correlation(x, 1)
          ay = auto_correlation(y, 1)
       end if
    else
       ax = ax_in
       ay = ay_in
    end if
    
    ! Use valid data only
    ! Get missing value exponent
    exp_msg = exponent(msg)
    if (present(msg)) then
       mask = (abs(x - msg) .gt. (tiny * exp_msg)) &
            .and. (abs(y - msg) .gt. (tiny * exp_msg))
       n_v = count(mask)
       allocate(x_v(n_v), y_v(n_v))
       x_v = pack(x, mask)
       y_v = pack(y, mask)
    else
       n_v = n
       allocate(x_v(n), y_v(n))
       x_v = x
       y_v = y
    end if

    ! Get estimator of the parameter
    estim = statistic(x_v, y_v)

    ! Get optimal block length
    lopt = lopt_auto_bivar(ax, ay, n_v)
    if (lopt < n_v) then
       ! Get resamples
       allocate(xb(n_v, sqrtNs**2), yb(n_v, sqrtNs**2))
       call resample_bivar(x_v, y_v, lopt, sqrtNs**2, xb, yb)
       do k  = 1, sqrtNs**2
          set(k) = statistic(xb(:, k), yb(:, k))
       end do
       
       ! Estimate the p-value from the statistics of the resamples
       pval = get_pval_from_set(set, estim)
       deallocate(xb, yb)
    else
       pval = 1.
    end if
    deallocate(x_v, y_v)
  end subroutine mbb_bivar_pval

  subroutine resample_bivar(x, y, Lblock, Ns, xb, yb, overtake)
    implicit none
    ! resample_bivar returns Ns resampled time series xb, yb
    ! by randomly drawing blocks of length Lblock from x and y, respectivelly.
    ! If overtake is true the blocks from x and y are
    ! taken at the same position.

    ! Dummy arguments
    real, dimension(:), intent(in) :: x, y     ! Original time series
    ! Block length, sqrt(number of samples), number of picks
    ! and remaining picks
    integer, intent(in) :: Lblock, Ns
    logical, intent(in), optional :: overtake ! dummy overtake indices

    ! Outputs
    ! Matrices of resampled time series
    real, dimension(size(x), Ns), intent(out) :: xb, yb 

    ! Local variables
    integer, dimension(size(x) / Lblock) :: block_ind ! possible block indices
    real, dimension(:), allocatable :: rstd_x, rstd_y ! standard random indices
    integer, dimension(:), allocatable :: rind_x, rind_y ! random indices
    integer :: Lxy, ndraw, ndraw_rem, rem ! problem dimensions
    integer :: ii, jj ! running indices
    logical :: overtake_loc = .false. ! overtake indices

    if (present(overtake)) then
       overtake_loc = overtake
    end if
    
    ! Get number of realisations
    Lxy = size(x)

    ! Get number of drawings and remaining drawings
    ndraw = Lxy / Lblock
    ndraw_rem = ndraw
    rem = Lxy - Lblock * ndraw
    if (rem > 0) then
       ndraw_rem = ndraw + 1
    end if
    
    ! Allocate random indices
    allocate(rind_x(ndraw_rem), rind_y(ndraw_rem), &
         rstd_x(ndraw_rem), rstd_y(ndraw_rem))
    
    ! range(ndraw)
    block_ind = (/(ii, ii=1, ndraw)/)

    do jj = 1, Ns
       ! Generate uniform random numbers
       call random_number(rstd_x)
       if (overtake_loc) then
          rstd_y = rstd_x
       else
          call random_number(rstd_y)
       end if
       ! Scale and offset from 1 to last possible picking index
       rind_x = floor(rstd_x * (Lxy - Lblock) + 1)
       rind_y = floor(rstd_y * (Lxy - Lblock) + 1)

       ! Construct matrices of resampled time series by drawing blocks
       do ii = 1, Lblock
          xb((block_ind - 1) * Lblock + ii, jj) = x(rind_x(:ndraw) + ii - 1)
          yb((block_ind - 1) * Lblock + ii, jj) = y(rind_y(:ndraw) + ii - 1)
       end do
       if (rem .gt. 0) then
          xb(((ndraw_rem - 1) * Lblock + 1):, jj) &
               = x(rind_x(ndraw_rem):(rind_x(ndraw_rem) + rem - 1))
          yb(((ndraw_rem - 1) * Lblock + 1):, jj) &
               = y(rind_y(ndraw_rem):(rind_y(ndraw_rem) + rem - 1))
       end if
    end do
    deallocate(rind_x, rind_y, rstd_x, rstd_y)
  end subroutine resample_bivar

  function get_pval_from_set(set, estim) result(res)
    implicit none
    ! get_pval_from_set gives the pvalue of an estimator estim from set
    ! using the percentile method

    ! Dummy arguments
    real, dimension(:), intent(in) :: set
    real, intent(in) :: estim

    ! Result
    real :: res

    ! Local variables
    real, dimension(size(set)) :: aset ! absolute value of the set
    real :: aestim ! absolute value of the estimator
    integer :: Ns ! Number of samples
   
    ! Take the absolute values
    aset = abs(set)
    aestim = abs(estim)

    ! Get Sample size
    Ns = size(set)

    ! Get percentile p-value of estim from set
    res = real(count(aset .gt. aestim)) / Ns
  end function get_pval_from_set
  
  function lopt_auto_bivar(ax, ay, n) result(lopt)
    implicit none
    ! lopt_auto_bivar gives the optimal block length lopt of a bivariate
    ! problem (X, Y) with n realizations from the lag-1 auto-correlation ax, ay
    
    ! Dummy arguments
    real, intent(in) :: ax, ay ! lag-1 auto-correlation
    integer, intent(in) :: n ! number of realizations

    ! Result
    integer :: lopt ! optimal block length
    
    ! Local variable
    real :: a ! deduced lag-1 auto-correlation

    ! Special cases
    if (( ax .ge. 1.) .or. (ay .ge. 1)) then
       lopt = n - 1
    else if ((ax .le. 0) .and. (ay .gt. 0)) then
       lopt = lopt_auto(ay, n)
    else if ((ay .le. 0) .and. (ax .gt. 0)) then
       lopt = lopt_auto(ax, n)
    else if ((ax .le. 0) .and. (ay .le. 0)) then 
       lopt = 1
    else
       ! General case
       a = sqrt(ax * ay)
       lopt = lopt_auto(a, n)
    end if
  end function lopt_auto_bivar

  function lopt_auto(a, n) result(lopt)
    implicit none
    ! lopt_auto gives the optimal block length lopt of a random vector
    ! with n realizations from its lag-1 auto-correlation a
    
    ! Dummy arguments
    real, intent(in) :: a
    integer, intent(in) :: n

    ! Result
    integer :: lopt ! optimal block length
    
    ! Block length estimator
    lopt = nint((6.**(1./2) * a / (1. - a**2))**(2./3) * N**(1./3))

    ! Special case
    if (lopt .eq. 0) then
       lopt = 1
    end if
  end function lopt_auto

  subroutine mbb_bivar_ci_field(X, statistic, sqrtNs, alpha, ci, estim, lopt, msg, ax_in)
    implicit none
    ! Dummy arguments
    real, dimension(:, :), intent(in) :: X
    integer, intent(in) :: sqrtNs
    real, intent(in) :: alpha
    interface
       function statistic(var1, var2)
         real, dimension(:, :), intent(in) :: var1, var2
         real, dimension(size(var1, 2), size(var2, 2)) :: statistic
       end function statistic
    end interface

    ! Optional dummy arguments
    real, optional, intent(in) :: msg
    real, dimension(size(x, 2)), optional, intent(in) :: ax_in
    
    ! Outputs
    real, dimension(size(x, 2), size(x, 2)), intent(out) :: estim
    real, dimension(size(x, 2), size(x, 2), 2), intent(out) :: ci
    integer, dimension(size(x, 2), size(x, 2)), intent(out) :: lopt

    ! Local variables
    integer :: m, n, ii, jj
    real, dimension(size(x, 2)) :: ax
    real :: estim_ij
    real, dimension(2) :: ci_ij
    integer :: lopt_ij

    ! Problem dimension
    m = size(x, 1)
    n = size(x, 2)
    
    ! Get auto-correlation at lag 1
    if (.not. present(ax_in)) then 
       if (present(msg)) then
          !$OMP PARALLEL DO
          do jj = 1, n
             ax(jj) = auto_correlation(X(:, jj), 1, msg)
          end do
          !$OMP END PARALLEL DO
       else
          !$OMP PARALLEL DO
          do jj = 1, n
             ax(jj) = auto_correlation(X(:, jj), 1)
          end do
          !$OMP END PARALLEL DO
       end if
    else
       ax = ax_in
    end if

    ! Get p-values for each pair of node
    if (present(msg)) then
       !$OMP PARALLEL DO PRIVATE(ci_ij, estim_ij, lopt_ij)
       do jj = 1, n
          ci(jj, jj, :) = (/0., 0./)
          estim(jj, jj) = 1.
          lopt(jj, jj) = 0
          do ii = jj+1, n
             call mbb_bivar_ci(X(:, ii), X(:, jj), statistic, sqrtNs, alpha, &
                  ci_ij, estim_ij, lopt_ij, msg, ax(ii), ax(jj))
             ci(ii, jj, :) = ci_ij
             ci(jj, ii, :) = ci_ij
             estim(ii, jj) = estim_ij
             estim(jj, ii) = estim_ij
             lopt(ii, jj) = lopt_ij
             lopt(jj, ii) = lopt_ij
          end do
       end do
       !$OMP END PARALLEL DO
    else
       !$OMP PARALLEL DO PRIVATE(ci_ij, estim_ij, lopt_ij)
       do jj = 1, n
          ci(jj, jj, :) = (/0., 0./)
          estim(jj, jj) = 1.
          lopt(jj, jj) = 0
          do ii = ii+1, n
             call mbb_bivar_ci(X(:, ii), X(:, jj), statistic, sqrtNs, alpha, &
                  ci_ij, estim_ij, lopt_ij, ax_in=ax(ii), ay_in=ax(jj))
             ci(ii, jj, :) = ci_ij
             ci(jj, ii, :) = ci_ij
             estim(ii, jj) = estim_ij
             estim(jj, ii) = estim_ij
             lopt(ii, jj) = lopt_ij
             lopt(jj, ii) = lopt_ij
          end do
       end do
       !$OMP END PARALLEL DO
    end if
  end subroutine mbb_bivar_ci_field
  
  subroutine mbb_bivar_ci(x, y, statistic, sqrtNs, alpha, ci, estim, lopt, &
       msg, ax_in, ay_in)
    implicit none
    ! Dummy arguments
    real, dimension(:), intent(in) :: x, y
    integer, intent(in) :: sqrtNs
    real, intent(in) :: alpha
    interface
       function statistic(var1, var2)
         real, dimension(:, :), intent(in) :: var1, var2
         real, dimension(size(var1, 2), size(var2, 2)) :: statistic
       end function statistic
    end interface

    ! Optional dummy arguments
    real, optional, intent(in) :: msg
    real, optional, intent(in) :: ax_in, ay_in
    
    ! Outputs
    real, dimension(2) :: ci
    real, intent(out) :: estim
    real, dimension(1, 1) :: estim2
    integer, intent(out) :: lopt

    ! Local variables
    integer :: exp_msg
    real, parameter :: tiny = 1.e-6
    real :: ax, ay
    logical, dimension(size(x)) :: mask
    real, dimension(:), allocatable :: x_v, y_v
    real, dimension(:, :), allocatable :: x_v2, y_v2
    real, dimension(:, :), allocatable :: xb, yb
    integer :: n, n_v
    real, dimension(sqrtNs**2) :: set
    real, dimension(sqrtNs, sqrtNs) :: set2
    
    exp_msg = exponent(msg)
    n = size(x)
    if ((.not. present(ax_in)) .or. (.not. present(ay_in))) then
       ! Calculate lag-1-auto-correlation of x and y
       if (present(msg)) then
          ax = auto_correlation(x, 1, msg)
          ay = auto_correlation(y, 1, msg)
       else
          ax = auto_correlation(x, 1)
          ay = auto_correlation(y, 1)
       end if
    else
       ax = ax_in
       ay = ay_in
    end if
    
    ! Use valid data only
    if (present(msg)) then
       mask = (abs(x - msg) .gt. (tiny * exp_msg)) &
            .and. (abs(y - msg) .gt. (tiny * exp_msg))
       n_v = count(mask)
       allocate(x_v(n_v), y_v(n_v))
       x_v = pack(x, mask)
       y_v = pack(y, mask)
    else
       n_v = n
       allocate(x_v(n), y_v(n))
       x_v = x
       y_v = y
    end if
    allocate(x_v2(n_v, 1), y_v2(n_v, 1))
    x_v2(:, 1) = x_v
    y_v2(:, 1) = y_v

    ! Get estimator of the parameter
    estim2 = statistic(x_v2, y_v2)
    estim = estim2(1, 1)

    ! Get optimal block length
    lopt = lopt_auto_bivar(ax, ay, n_v)
    if (lopt < n_v) then
       ! Get resamples
       allocate(xb(n_v, sqrtNs), yb(n_v, sqrtNs))
       call resample_bivar(x_v, y_v, lopt, sqrtNs, xb, yb, overtake=.true.)
       ! Get statistic of samples
       set2 = pearson_corr(xb, yb)
       set = reshape(set2, (/sqrtNs**2/))
       ci = get_ci_from_set(set, alpha)
       deallocate(xb, yb)
    else
       ci = (/-1., 1./)
    end if
    deallocate(x_v, y_v, x_v2, y_v2)
  end subroutine mbb_bivar_ci

  function get_ci_from_set(set, alpha) result(res)
    implicit none
    real, dimension(:), intent(in) :: set
    real, intent(in) :: alpha
    real, dimension(2) :: res
    real, dimension(size(set)) :: sset
    integer Ns
   
    ! Sample size
    Ns = size(set)
    ! Sort set
    sset = set
    call QsortC(sset)
    ! Get lower and upper boundaries of the percentile CI
    res(1) = sset(floor(alpha / 2 * Ns))
    res(2) = sset(ceiling((1 - alpha / 2) * Ns))
  end function get_ci_from_set
  
end module bootstrap_mpi
