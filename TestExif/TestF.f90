program testf
  implicit none
  include 'pi2d_f.inc'
  real*4 :: data(80, 60)
  real*4 :: pi = 3.14159
  integer :: i, j, iret
  character*128 :: atrarg, lutn

  do j = 1, 60
     do i = 1, 80
        data(i, j) = (i * 2d0 * pi / 79) * (j * 2d0 * pi / 59)
     enddo
  enddo

  iret = pi2d_init()
  atrarg = "arraySize=80,60"
  iret = pi2d_setattrib(atrarg, 15)
  lutn = "default"
  iret = pi2d_draws(0, data, lutn, 7, 10, 0)
  iret = pi2d_output(0, 0, 0, 0)
  iret = pi2d_finalize()
  
  stop
end program testf
