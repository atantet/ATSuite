program sl_hadsst3
  use bootstrap
  use atmath
  use netcdf
  character(len=128) :: src_file, varname
  integer :: ncid, varid
  integer :: status
  real, dimension(:, :), allocatable :: values
  integer, dimension(2) :: dimIDs
  integer :: no_fill, fill_value
  integer :: nt, nvar
  integer :: sqrtNs
  real, dimension(:, :), allocatable :: pval, estim
  integer, dimension(:, :), allocatable :: lopt
  integer :: unit
  real :: missing_value = -1.e30

  src_file = "/Users/atantet/PhD/data/Hadley/HadSST3/hadsst3_proc_5x5_oce_noavg.nc"
  varname = "sst"

  ! Open
  status = nf90_open(src_file, nf90_nowrite, ncid)
  ! Varid
  status = nf90_inq_varid(ncid, varname, varid)
  ! Variable dimensions
  status = nf90_inquire_variable(ncid, varid, dimids = dimIDs)
  status = nf90_inquire_dimension(ncid, dimIDs(1), len = nvar)
  status = nf90_inquire_dimension(ncid, dimIDs(2), len = nt)
  allocate(values(nvar, nt))
  ! Get values
  status = nf90_get_var(ncid, varid, values)
  ! Get fill value
  status = nf90_inq_var_fill(ncid, varid, no_fill, fill_value)

  allocate(pval(nvar, nvar))
  allocate(estim(nvar, nvar))
  allocate(lopt(nvar, nvar))

  print *, "Number of realisations : ", nt
  print *, "Number of nodes : ", nvar
  
  ! MBB
  sqrtNs = 100
  print *, "MBB msg"
  call mbb_bivar_pval_field(transpose(values), pearson_corr_mat, sqrtNs, pval, estim, lopt, msg=missing_value)
  print *, "Done."

  ! Write results
  open(unit=unit, file="pval.txt", status="replace", action="write")
  write (unit, "(f8.6)") pval
  close(unit)

  open(unit=unit, file="estim.txt", status="replace", action="write")
  write (unit, "(f8.6)") estim
  close(unit)

  open(unit=unit, file="lopt.txt", status="replace", action="write")
  write (unit, "(i3)") lopt
  close(unit)
  
end program sl_hadsst3
