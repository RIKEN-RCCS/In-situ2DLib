#ifndef _PI2D_C_H_
#define _PI2D_C_H_

#include "Pi2Ddefs.h"

extern int pi2d_init();
extern int pi2d_finalize();
extern int pi2d_setattrib(char* arg, int arglen);
extern int pi2d_setcord(Real* arr, int veclen, int vecidxt[2]);
extern int pi2d_createlut(char* name, int namelen);
extern int pi2d_setlutcolor(char* name, int namelen, int ncolor,
			    Real* valuelist, Real* colorlist);
extern int pi2d_setlutattrib(char* name, int namelen, char* arg, int arglen);
extern int pi2d_draws(int vt, Real* data, char* lutname, int lutnamelen,
		      int nlevels, int cbshow);
extern int pi2d_drawv(Real* data, int veclen, int vecidx[2],
		      char* lutname, int lutnamelen, int colidx, int cbshow);
extern int pi2d_output(int step, int row, int col, int proc);
extern int pi2d_importattrib(char* path, int pathlen);
extern int pi2d_exportattrib(char* path, int pathlen);

#endif // _PI2D_C_H_
