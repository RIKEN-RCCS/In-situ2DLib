/* MPI4PyTest.cpp
   compile: mpicxx -o MPI4PyTest MPI4PyTest.cpp `python-config --cflags --ldflags`
   execute: mpiexec -n 3 ./MPI4PyTest
*/
#include <Python.h>
#include <mpi.h>

const char helloworld[] = \
  "import mpi4py\n"
  "mpi4py.rc.initialize = False\n"
  "mpi4py.rc.finalize = False\n"
  "from mpi4py import MPI\n"
  "hwmess = 'Hello, World! I am process %d of %d on %s.'\n"
  "myrank = MPI.COMM_WORLD.Get_rank()\n"
  "nprocs = MPI.COMM_WORLD.Get_size()\n"
  "procnm = MPI.Get_processor_name()\n"
  "print (hwmess % (myrank, nprocs, procnm))\n";

int main(int argc, char *argv[]) {
  int i, n=5;

  MPI_Init(&argc, &argv);
  Py_Initialize();

  for (i=0; i<n; i++) {
    PyRun_SimpleString(helloworld);
  }

  Py_Finalize();
  MPI_Finalize();

  return 0;
}

