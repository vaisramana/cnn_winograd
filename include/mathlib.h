#ifndef MATHLAB_H
#define MATHLAB_H

#define USE_MKL 0
#define USE_OPENBLAS 1


#ifndef __aarch64__
#if USE_MKL
#include <mkl.h>

#elif USE_OPENBLAS
#include <cblas.h>

#endif
#endif

#endif