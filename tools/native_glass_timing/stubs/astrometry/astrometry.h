#ifndef GLASS_TIMING_ASTROMETRY_STUB_H
#define GLASS_TIMING_ASTROMETRY_STUB_H
#include <stdint.h>

/* Declarations only.  The astrometry wrapper is dead-stripped from this
 * waveform-only executable, so no implementation is linked. */
int64_t healpixl_ring_to_xy(int64_t ipix, int nside);
void healpixl_to_radec(int64_t xy, int nside, double dx, double dy,
                       double *ra, double *dec);
#endif
