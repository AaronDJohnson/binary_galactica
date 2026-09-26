/* Standalone test support. upstream.c contains unmodified pinned GLASS routines.
 * Analytic source/orbit evaluators replace the generic spline storage only;
 * geometry, eight retarded responses and complex extraction are upstream C.
 */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../glass_reference/glass_constants.h"
#define LARM 2.5e9
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
struct CubicSpline { int kind, spacecraft, axis; double p[9], T; };
struct Orbit {
    double L, fstar, ecc, R, kappa_0, lambda_0;
    void (*orbit_function)(struct Orbit *, double, double *, double *, double *);
    struct CubicSpline *dx[3], *dy[3], *dz[3];
};
struct TDI { double *X, *Y, *Z; };
static struct Orbit orbit;
static double *double_vector(int n) { return calloc(n, sizeof(double)); }
static void free_double_vector(double *p) { free(p); }
static void alloc_tdi(struct TDI *r, int n, int channels) {
    (void)channels; r->X=double_vector(n); r->Y=double_vector(n); r->Z=double_vector(n);
}
static void free_tdi(struct TDI *r) { free(r->X); free(r->Y); free(r->Z); free(r); }
void analytic_orbits(struct Orbit *, double, double *, double *, double *);
void extract_amplitude_and_phase(int, double *, double *, double *, double *, double *);
void unwrap_phase(int, double *);
static double spline_interpolation(struct CubicSpline *s, double t) {
    if(s->kind == 0) {
        double x[4], y[4], z[4]; analytic_orbits(&orbit, t, x, y, z);
        double *axis[] = {x,y,z}; return axis[s->axis][s->spacecraft+1];
    }
    double f=s->p[0]/s->T, fd=s->p[7]/s->T/s->T;
    if(s->kind == 1) return exp(s->p[3])*(1.+2./3.*fd/f*t);
    double fdd=s->p[8]/s->T/s->T/s->T;
    return -s->p[6]+PI2*(f*t+.5*fd*t*t+fdd*t*t*t/6.);
}
#include "upstream.c"
int main(void) {
    double T; int np, n;
    if(scanf("%lf %d %d", &T, &np, &n)!=3) return 2;
    struct CubicSpline amp={.kind=1,.T=T}, phase={.kind=2,.T=T};
    for(int i=0;i<np;i++) { if(scanf("%lf", &amp.p[i])!=1) return 3; phase.p[i]=amp.p[i]; }
    double *times=double_vector(n), *ref=double_vector(n);
    for(int i=0;i<n;i++) { if(scanf("%lf", &times[i])!=1) return 4; ref[i]=spline_interpolation(&phase,times[i]); }
    initialize_analytic_orbit(&orbit);
    struct CubicSpline coords[3][3]={0};
    for(int i=0;i<3;i++) for(int j=0;j<3;j++) { coords[i][j].spacecraft=i; coords[i][j].axis=j; }
    for(int i=0;i<3;i++) { orbit.dx[i]=&coords[i][0]; orbit.dy[i]=&coords[i][1]; orbit.dz[i]=&coords[i][2]; }
    struct TDI a, p; alloc_tdi(&a,n,3); alloc_tdi(&p,n,3);
    LISA_spline_response(&orbit,times,n,amp.p[1],amp.p[2],amp.p[4],amp.p[5],&amp,NULL,&phase,ref,&a,&p);
    for(int i=0;i<n;i++) {
        for(int sc=0;sc<3;sc++) for(int ax=0;ax<3;ax++) printf("%.17g ",spline_interpolation(&coords[sc][ax],times[i]));
        double av[]={a.X[i],a.Y[i],a.Z[i]}, pv[]={p.X[i],p.Y[i],p.Z[i]};
        for(int c=0;c<3;c++) printf("%.17g %.17g ",av[c]*cos(pv[c]),av[c]*sin(pv[c]));
        puts("");
    }
    free(a.X);free(a.Y);free(a.Z);free(p.X);free(p.Y);free(p.Z);free(times);free(ref);
    return 0;
}
