/* Standalone fixture I/O harness; the included GLASS routines are unchanged. */
#include "reference.c"

int main(void)
{
    double T, t0, params[9];
    int BW, NP;
    char fmt[32];
    if(scanf("%lf %lf %d %d %31s", &T, &t0, &BW, &NP, fmt) != 5) return 2;
    if(NP < 7 || NP > 9 || BW < 2 || BW % 2 || T <= 0) return 3;
    for(int j=0; j<NP; j++) if(scanf("%lf", &params[j]) != 1) return 4;
    struct Orbit orbit;
    initialize_analytic_orbit(&orbit);
    double *channels[5];
    /* Extra leading element keeps upstream X-1 pointer arithmetic in bounds. */
    for(int j=0; j<5; j++) channels[j] = calloc(2*BW+1, sizeof(double));
    ucb_waveform(&orbit, fmt, T, t0, params, NP, channels[0]+1,
                 channels[1]+1, channels[2]+1, channels[3]+1, channels[4]+1, BW, 3);
    long q = (long)((params[0]/T)*T);
    printf("%ld %ld\n", q, q-BW/2);
    for(int i=0; i<BW; i++) {
        printf("%ld ", q-BW/2+i);
        for(int j=0; j<5; j++)
            printf("%.17g %.17g ", channels[j][2*i+1], channels[j][2*i+2]);
        printf("\n");
    }
    for(int j=0; j<5; j++) free(channels[j]);
    return 0;
}
