program sl_hadsst3
  use nonparam
  use atmath
  use netcdf
  character(len=128) :: src_file, varname
  integer :: ncid, varid
  integer :: status
  real, dimension(:, :), allocatable :: values
  integer, dimension(2) :: dimIDs
  integer :: no_fill, fill_value
  integer :: nt, nvar
  integer :: Ns
  real :: sqrtNs
  real, dimension(:, :), allocatable :: pval, estim
  integer, dimension(:, :), allocatable :: lopt
  integer :: unit
  integer :: my_rank, ierr

  src_file = "/Users/atantet/PhD/data/Hadley/HadISST/sst/hadisst_sst_proc_20x20_oce_lp13m.nc"
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

  ! MBB
  Ns = 400
  sqrtNs = int(sqrt(real(Ns)))
  call mbb_bivar_pval_field(transpose(values), pearson_corr_vect, Ns, pval, estim, lopt, msg=-1.e30)

  if (my_rank .eq. 0) then
     ! Write results
     open(unit=unit, file="pval.txt", status="replace", action="write")
     write (unit, "(f8.6)") pval
     close(unit)
     
     open(unit=unit, file="estim.txt", status="replace", action="write")
     write (unit, "(f8.6)") estim
     close(unit)

     open(unit=unit, file="lopt.txt", status="replace", action="write")
     write (unit, "(f8.6)") lopt
     close(unit)
     
  end if
  

end program sl_hadsst3