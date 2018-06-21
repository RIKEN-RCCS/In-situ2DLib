#if PY_MAJOR_VERSION < 3
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif
#include "MicroEnv.h"

static int numargs=0;

/* Return the number of arguments of the application command line */

static PyObject* emb_numargs(PyObject *self, PyObject *args) {
  if ( ! PyArg_ParseTuple(args, "") )
    return NULL;
  return Py_BuildValue("i", numargs);
}
static PyMethodDef EmbMethods[] = {
  {"numargs", emb_numargs, METH_VARARGS,
   "Return the number of arguments received by the process."},
  {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
static PyModuleDef EmbModule = {
  PyModuleDef_HEAD_INIT,
  "emb", NULL, -1, EmbMethods,
  NULL, NULL, NULL, NULL
};
static PyObject* PyInit_emb(void) {
  return PyModule_Create(&EmbModule);
}
#endif


int main(int argc, char** argv) {
  MicroEnv* me = MicroEnv::GetInstance();
  if ( ! me ) {
    printf("MicroEnv::GetInstance failed\n");
    return 1;
  }

#if PY_MAJOR_VERSION >= 3
  PyImport_AppendInittab("emb", &PyInit_emb);
#endif

  if ( ! me->initialize() ) {
    printf("MicroEnv::initialize failed\n");
    return 2;
  }

  numargs = argc;
#if PY_MAJOR_VERSION < 3
  Py_InitModule("emb", EmbMethods);
#endif
  if ( PyErr_Occurred() ) {
    PyErr_Print();
  }
  if ( ! me->execute(std::string("Y")) ) {
    printf("MicroEnv::execute failed\n");
    return 3;
  }

  me->finalize();
  delete me;
  return 0;
}
