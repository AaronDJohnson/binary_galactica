/* Minimal standalone declarations for the extracted GLASS functions.
 * This harness is maintained by binary-galactica; routines are unchanged.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "vendor/kissfft/kiss_fft.h"
#include "glass_constants.h"
#define LARM 2.5e9
struct Orbit {
    double L, fstar, ecc, R, kappa_0, lambda_0;
    void (*orbit_function)(struct Orbit *, double, double *, double *, double *);
};
