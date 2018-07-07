program test
  implicit none

  integer :: dims(3)
  real*4, allocatable, dimension(:,:,:) :: p
  real*4, allocatable, dimension(:,:,:,:) :: v

  open(1, file="cavP.d", form="unformatted")
  read(1) dims
  print *, "P dims = ", dims 

  allocate(p(dims(1),dims(2),dims(3)))
  read(1) p
  close(1)

  open(1, file="cavV.d", form="unformatted")
  read(1) dims
  print *, "V dims = ", dims

  allocate(v(3, dims(1),dims(2),dims(3)))
  read(1) v
  close(1)

  stop
end program test

