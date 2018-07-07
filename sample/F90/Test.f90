program test
  implicit none
  include 'pi2d_f.inc'
  
  integer :: p_dims(3), v_dims(3), vidx(2)
  real*4, allocatable, dimension(:,:,:) :: p
  real*4, allocatable, dimension(:,:,:,:) :: v
  character*64 :: attrbuf, lutn
  integer :: i, iret

  open(1, file="../data/cavP.d", form="unformatted")
  read(1) p_dims
  print *, "P dims = ", p_dims 

  allocate(p(p_dims(1),p_dims(2),p_dims(3)))
  read(1) p
  close(1)

  open(1, file="../data/cavV.d", form="unformatted")
  read(1) v_dims
  print *, "V dims = ", v_dims

  allocate(v(3, v_dims(1),v_dims(2),v_dims(3)))
  read(1) v
  close(1)

  iret = pi2d_init()
  if ( iret .ne. 0 ) then
     print *, "pi2d_init failed"
     stop
  endif
  attrbuf = "imageSize=500,500"
  iret = pi2d_setattrib(attrbuf, len_trim(attrbuf))
  if ( iret .ne. 0 ) then
     print *, "pi2d_setattrib imageSize failed"
     stop
  endif

  write(attrbuf, *) "arraySize=", p_dims(1), ",", p_dims(2)
  iret = pi2d_setattrib(attrbuf, len_trim(attrbuf))
  if ( iret .ne. 0 ) then
     print *, "pi2d_setattrib arraySize failed"
     stop
  endif

  lutn = ""
  iret = pi2d_draws(0, p, lutn, 0, 20, 1)
  if ( iret .ne. 0 ) then
     print *, "pi2d_draws failed"
     stop
  endif

  write(attrbuf, *) "arraySize=", v_dims(1), ",", v_dims(2)
  iret = pi2d_setattrib(attrbuf, len_trim(attrbuf))
  iret = pi2d_setattrib(attrbuf, len_trim(attrbuf))
  if ( iret .ne. 0 ) then
     print *, "pi2d_setattrib arraySize failed"
     stop
  endif
  
  vidx(1) = 1
  vidx(2) = 2
  iret = pi2d_drawv(v, 3, vidx, lutn, 0, -2, 0)
  if ( iret .ne. 0 ) then
     print *, "pi2d_drawv failed"
     stop
  endif
  
  iret = pi2d_output(0, 0, 0, 0)
  if ( iret .ne. 0 ) then
     print *, "pi2d_output failed"
     stop
  endif
  
  iret = pi2d_finalize()
  
  stop
end program test

