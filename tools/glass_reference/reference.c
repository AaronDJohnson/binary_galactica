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


/*
 * Copyright 2019 Tyson B. Littenberg & Neil J. Cornish
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

void analytic_orbits(struct Orbit *orbit, double t, double *x, double *y, double *z)
{
    
    double alpha = PI2*t/YEAR; // + kappa_0
    
    /*
     double beta1 = 0.;              //lambda_0
     double beta2 = 2.0943951023932; //2.*pi/3. + lambda_0;
     double beta3 = 4.18879020478639;//4.*pi/3. + lambda_0;
     */
    
    double sa = sin(alpha);
    double ca = cos(alpha);
    
    double sa2  = sa*sa;
    double ca2  = ca*ca;
    double saca = sa*ca;
    double AUca = AU*ca;
    double AUsa = AU*sa;
    
    double sb,cb;
    
    sb = 0.0;//sin(beta1);
    cb = 1.0;//cos(beta1);
    x[1] = AUca + orbit->R*(saca*sb - (1. + sa2)*cb);
    y[1] = AUsa + orbit->R*(saca*cb - (1. + ca2)*sb);
    z[1] = -SQ3*orbit->R*(ca*cb + sa*sb);
    
    sb = 0.866025403784439;//sin(beta2);
    cb = -0.5;//cos(beta2);
    x[2] = AUca + orbit->R*(saca*sb - (1. + sa2)*cb);
    y[2] = AUsa + orbit->R*(saca*cb - (1. + ca2)*sb);
    z[2] = -SQ3*orbit->R*(ca*cb + sa*sb);
    
    sb = -0.866025403784438;//sin(beta3);
    cb = -0.5;//cos(beta3);
    x[3] = AUca + orbit->R*(saca*sb - (1. + sa2)*cb);
    y[3] = AUsa + orbit->R*(saca*cb - (1. + ca2)*sb);
    z[3] = -SQ3*orbit->R*(ca*cb + sa*sb);

    // next order corrections (no longer equal arm)
     
    /*
    x[i] += 0.125*ec*ec*AU*(-10.*ca - 5.*ca*cb*cb + 3.*ca*ca*ca*cb*cb - 9.*ca*cb*cb*sa*sa - 10.*cb*sa*sb + 18.*ca*ca*cb*sa*sb - 6.*cb*sa*sa*sa*sb + 5.*ca*sb*sb - 3.*ca*ca*ca*sb*sb + 9.*ca*sa*sa*sb*sb);
    y[i] += 0.125*ec*ec*AU*(-10.*sa + 5.*cb*cb*sa + 9.*ca*ca*cb*cb*sa - 3.*cb*cb*sa*sa*sa - 10.*ca*cb*sb - 6.*ca*ca*ca*cb*sb + 18.*ca*cb*sa*sa*sb - 5.*sa*sb*sb - 9.*ca*ca*sa*sb*sb + 3.*sa*sa*sa*sb*sb);
    z[i] += sq3*AU*ec*ec*(1. + (sa*cb - sb*ca)*(sa*cb - sb*ca));
    */
}

void initialize_analytic_orbit(struct Orbit *orbit)
{
    //store armlenght & transfer frequency in orbit structure.
    orbit->L        = LARM;
    orbit->fstar    = CLIGHT/(2.0*M_PI*LARM);
    orbit->ecc      = LARM/(2.0*SQ3*AU);
    orbit->R        = AU*orbit->ecc;
    orbit->kappa_0  = 0.0;
    orbit->lambda_0 = 0.0;
    orbit->orbit_function = &analytic_orbits;
    
}

static void recursive_phase_evolution(double dre, double dim, double *cosPhase, double *sinPhase)
{
   /* Update re and im for the next iteration. */
   double cosphi = *cosPhase;
   double sinphi = *sinPhase;

   double newRe = cosphi*dre - sinphi*dim;
   double newIm = sinphi*dre + cosphi*dim;

   *cosPhase = newRe;
   *sinPhase = newIm;
   
}

static void double_angle(double cosA, double sinA, double *cos2A, double *sin2A)
{
    *cos2A = 2.*cosA*cosA - 1.0;
    *sin2A = 2.*sinA*cosA;
}

static void triple_angle(double cosA, double sinA, double cos2A, double *cos3A, double *sin3A)
{
    *cos3A = 2.*cosA*cos2A - cosA;
    *sin3A = 2.*sinA*cos2A + sinA;
}

void XYZ2AE(double X, double Y, double Z, double *A, double *E)
{
    /*
     * Conventions in ldc/common/series/tdi.py
     * A = (Z-X)/sqrt(2)
     * E = (X-2Y+Z)/sqrt(6)
     */
    double invSQ6 = 0.408248290463863;
    
    *A = (Z-X)*M_SQRT1_2;
    *E = (X-2*Y+Z)*invSQ6;

}

void LISA_tdi(double L, double fstar, double T, double ***d, double f0, long q, double *X, double *Y, double *Z, double *A, double *E, int BW, int NI)
{
    int i,j,k;
    int BWon2 = BW/2;
    double fonfs;
    double c3, s3, c2, s2, c1, s1;
    double f;
    double sqT=sqrt(T);
    double invfstar = 1./fstar;
    double dPhi = invfstar/T;
    double cosdPhi = cos(dPhi);
    double sindPhi = sin(dPhi);
    double prefactor;
    
    /* Initialize recursion */
    i = 0;
    f = ((double)(q + i-1 - BWon2))/T;
    fonfs = f*invfstar;
    
    c1 = cos(fonfs);
    s1 = sin(fonfs);

    double_angle(c1,s1,&c2,&s2);
    triple_angle(c1,s1,c2,&c3,&s3);

    
    for(i=1; i<=BW; i++)
    {
        k = 2*i;
        j = k-1;
        
        f = ((double)(q + i-1 - BWon2))/T;
        fonfs = f*invfstar;
        
        prefactor = sqT;

        /* make use of recursion relationships & identities to get rid of trig calls
        c3 = cos(3.*fonfs);  c2 = cos(fonfs2);  c1 = cos(fonfs);
        s3 = sin(3.*fonfs);  s2 = sin(fonfs2);  s1 = sin(fonfs);
         */

        recursive_phase_evolution(cosdPhi, sindPhi, &c1, &s1);
        double_angle(c1,s1,&c2,&s2);
        triple_angle(c1,s1,c2,&c3,&s3);

        X[j] =	prefactor*((d[1][2][j]-d[1][3][j])*c3 + (d[1][2][k]-d[1][3][k])*s3 +
        (d[2][1][j]-d[3][1][j])*c2 + (d[2][1][k]-d[3][1][k])*s2 +
        (d[1][3][j]-d[1][2][j])*c1 + (d[1][3][k]-d[1][2][k])*s1 +
        (d[3][1][j]-d[2][1][j]));
        
        X[k] =	prefactor*((d[1][2][k]-d[1][3][k])*c3 - (d[1][2][j]-d[1][3][j])*s3 +
        (d[2][1][k]-d[3][1][k])*c2 - (d[2][1][j]-d[3][1][j])*s2 +
        (d[1][3][k]-d[1][2][k])*c1 - (d[1][3][j]-d[1][2][j])*s1 +
        (d[3][1][k]-d[2][1][k]));
        
        /* Michelson channel for single IFO
        M[j] = prefactor*(X[j]*cLS - X[k]*sLS);
        M[k] =-prefactor*(X[j]*sLS + X[k]*cLS);
         */
        
        //save some CPU time when only X-channel is needed
        if(NI>1)
        {
            Y[j] =	prefactor*((d[2][3][j]-d[2][1][j])*c3 + (d[2][3][k]-d[2][1][k])*s3 +
            (d[3][2][j]-d[1][2][j])*c2 + (d[3][2][k]-d[1][2][k])*s2+
            (d[2][1][j]-d[2][3][j])*c1 + (d[2][1][k]-d[2][3][k])*s1+
            (d[1][2][j]-d[3][2][j]));
            
            Y[k] =	prefactor*((d[2][3][k]-d[2][1][k])*c3 - (d[2][3][j]-d[2][1][j])*s3+
            (d[3][2][k]-d[1][2][k])*c2 - (d[3][2][j]-d[1][2][j])*s2+
            (d[2][1][k]-d[2][3][k])*c1 - (d[2][1][j]-d[2][3][j])*s1+
            (d[1][2][k]-d[3][2][k]));
            
            Z[j] =	prefactor*((d[3][1][j]-d[3][2][j])*c3 + (d[3][1][k]-d[3][2][k])*s3+
            (d[1][3][j]-d[2][3][j])*c2 + (d[1][3][k]-d[2][3][k])*s2+
            (d[3][2][j]-d[3][1][j])*c1 + (d[3][2][k]-d[3][1][k])*s1+
            (d[2][3][j]-d[1][3][j]));
            
            Z[k] =	prefactor*((d[3][1][k]-d[3][2][k])*c3 - (d[3][1][j]-d[3][2][j])*s3+
            (d[1][3][k]-d[2][3][k])*c2 - (d[1][3][j]-d[2][3][j])*s2+
            (d[3][2][k]-d[3][1][k])*c1 - (d[3][2][j]-d[3][1][j])*s1+
            (d[2][3][k]-d[1][3][k]));
            
            /* LDC conventions for A & E channels */
            XYZ2AE(X[j],Y[j],Z[j],&A[j],&E[j]);
            XYZ2AE(X[k],Y[k],Z[k],&A[k],&E[k]);
            
            A[k] = -A[k];
            E[k] = -E[k];
        }
    }
}

void LISA_tdi_FF(double L, double fstar, double T, double ***d, double f0, long q, double *X, double *Y, double *Z, double *A, double *E, int BW, int NI)
{
    int i,j,k;
    int BWon2 = BW/2;
    double fonfs,fonfs2;
    double c3, s3, c2, s2, c1, s1;
    double f;
    double sqT=sqrt(T);
    double invfstar = 1./fstar;
    double dPhi = invfstar/T;
    double cosdPhi = cos(dPhi);
    double sindPhi = sin(dPhi);
    double prefactor;
    
    
    /* Initialize recursion */
    i = 0;
    f = ((double)(q + i-1 - BWon2))/T;
    fonfs = f*invfstar;
    
    c1 = cos(fonfs);
    s1 = sin(fonfs);

    double_angle(c1,s1,&c2,&s2);
    triple_angle(c1,s1,c2,&c3,&s3);

    for(i=1; i<=BW; i++)
    {
        k = 2*i;
        j = k-1;
        
        f = ((double)(q + i-1 - BWon2))/T;
        fonfs = f*invfstar;
        fonfs2= 2.*fonfs;
        
        /* prefactor = sqrt(T) * 4 * (f/fstar) * sin(f/fstar) */
        prefactor = sqT*fonfs2;

        /* make use of recursion relationships & identities to get rid of trig calls
        c3 = cos(3.*fonfs);  c2 = cos(fonfs2);  c1 = cos(fonfs);
        s3 = sin(3.*fonfs);  s2 = sin(fonfs2);  s1 = sin(fonfs);
         */

        recursive_phase_evolution(cosdPhi, sindPhi, &c1, &s1);
        double_angle(c1,s1,&c2,&s2);
        triple_angle(c1,s1,c2,&c3,&s3);
        
        
        X[j] =	prefactor*((d[1][2][j]-d[1][3][j])*c3 + (d[1][2][k]-d[1][3][k])*s3 +
        (d[2][1][j]-d[3][1][j])*c2 + (d[2][1][k]-d[3][1][k])*s2 +
        (d[1][3][j]-d[1][2][j])*c1 + (d[1][3][k]-d[1][2][k])*s1 +
        (d[3][1][j]-d[2][1][j]));
        
        X[k] =	prefactor*((d[1][2][k]-d[1][3][k])*c3 - (d[1][2][j]-d[1][3][j])*s3 +
        (d[2][1][k]-d[3][1][k])*c2 - (d[2][1][j]-d[3][1][j])*s2 +
        (d[1][3][k]-d[1][2][k])*c1 - (d[1][3][j]-d[1][2][j])*s1 +
        (d[3][1][k]-d[2][1][k]));
        
        /* Michelson channels for single IFO
        M[j] = prefactor*(X[j]*cSL - X[k]*sSL);
        M[k] = prefactor*(X[j]*sSL + X[k]*cSL);
         */
        
        //save some CPU time when only X-channel is needed
        if(NI>1)
        {
            Y[j] =	prefactor*((d[2][3][j]-d[2][1][j])*c3 + (d[2][3][k]-d[2][1][k])*s3 +
            (d[3][2][j]-d[1][2][j])*c2 + (d[3][2][k]-d[1][2][k])*s2+
            (d[2][1][j]-d[2][3][j])*c1 + (d[2][1][k]-d[2][3][k])*s1+
            (d[1][2][j]-d[3][2][j]));
            
            Y[k] =	prefactor*((d[2][3][k]-d[2][1][k])*c3 - (d[2][3][j]-d[2][1][j])*s3+
            (d[3][2][k]-d[1][2][k])*c2 - (d[3][2][j]-d[1][2][j])*s2+
            (d[2][1][k]-d[2][3][k])*c1 - (d[2][1][j]-d[2][3][j])*s1+
            (d[1][2][k]-d[3][2][k]));
            
            Z[j] =	prefactor*((d[3][1][j]-d[3][2][j])*c3 + (d[3][1][k]-d[3][2][k])*s3+
            (d[1][3][j]-d[2][3][j])*c2 + (d[1][3][k]-d[2][3][k])*s2+
            (d[3][2][j]-d[3][1][j])*c1 + (d[3][2][k]-d[3][1][k])*s1+
            (d[2][3][j]-d[1][3][j]));
            
            Z[k] =	prefactor*((d[3][1][k]-d[3][2][k])*c3 - (d[3][1][j]-d[3][2][j])*s3+
            (d[1][3][k]-d[2][3][k])*c2 - (d[1][3][j]-d[2][3][j])*s2+
            (d[3][2][k]-d[3][1][k])*c1 - (d[3][2][j]-d[3][1][j])*s1+
            (d[2][3][k]-d[1][3][k]));
            
            /* LDC conventions for A & E channels */
            XYZ2AE(X[j],Y[j],Z[j],&A[j],&E[j]);
            XYZ2AE(X[k],Y[k],Z[k],&A[k],&E[k]);

        }
    }
}

void LISA_tdi_Sangria(double L, double fstar, double T, double ***d, double f0, long q, double *X, double *Y, double *Z, double *A, double *E, int BW, int NI)
{
    int i,j,k;
    int BWon2 = BW/2;
    double prefactor,fonfs;
    double c2, s2, c1, s1;
    double f;
    double sqT=sqrt(T);
    double invfstar = 1./fstar;
    double norm = 4.0*invfstar*sqT;
    double dPhi = invfstar/T;
    double cosdPhi = cos(dPhi);
    double sindPhi = sin(dPhi);
        
    /* Initialize recursion */
    i = 0;
    f = ((double)(q + i-1 - BWon2))/T;
    fonfs = f*invfstar;
    
    c1 = cos(fonfs);
    s1 = sin(fonfs);

    double_angle(c1,s1,&c2,&s2);

    for(i=1; i<=BW; i++)
    {
        k = 2*i;
        j = k-1;

        f = ((double)(q + i-1 - BWon2))/T;

        /* make use of recursion relationships & identities to get rid of trig calls
        c3 = cos(3.*fonfs);  c2 = cos(2.*fonfs);  c1 = cos(fonfs);
        s3 = sin(3.*fonfs);  s2 = sin(2.*fonfs);  s1 = sin(fonfs);*/
        recursive_phase_evolution(cosdPhi, sindPhi, &c1, &s1);
        double_angle(c1,s1,&c2,&s2);
                
        /* prefactor = sqrt(T) * 4 * (f/fstar) * sin(f/fstar) */
        prefactor = f * norm * s1;
        
        
        /* LDC Conventions for X,Y,Z channels (circa Sangria) */
        X[j] = prefactor*((d[1][2][j]-d[1][3][j])*c2 + (d[1][2][k]-d[1][3][k])*s2 +
                   (d[2][1][j]-d[3][1][j])*c1 + (d[2][1][k]-d[3][1][k])*s1);
        
        X[k] = prefactor*((d[1][2][k]-d[1][3][k])*c2 - (d[1][2][j]-d[1][3][j])*s2 +
                   (d[2][1][k]-d[3][1][k])*c1 - (d[2][1][j]-d[3][1][j])*s1);
        
        Y[j] = prefactor*((d[2][3][j]-d[2][1][j])*c2 + (d[2][3][k]-d[2][1][k])*s2 +
                   (d[3][2][j]-d[1][2][j])*c1 + (d[3][2][k]-d[1][2][k])*s1);
        
        Y[k] = prefactor*((d[2][3][k]-d[2][1][k])*c2 - (d[2][3][j]-d[2][1][j])*s2+
                   (d[3][2][k]-d[1][2][k])*c1 - (d[3][2][j]-d[1][2][j])*s1);
        
        Z[j] = prefactor*((d[3][1][j]-d[3][2][j])*c2 + (d[3][1][k]-d[3][2][k])*s2+
                   (d[1][3][j]-d[2][3][j])*c1 + (d[1][3][k]-d[2][3][k])*s1);
        
        Z[k] = prefactor*((d[3][1][k]-d[3][2][k])*c2 - (d[3][1][j]-d[3][2][j])*s2+
                   (d[1][3][k]-d[2][3][k])*c1 - (d[1][3][j]-d[2][3][j])*s1);
        

        /* LDC conventions for A & E channels */
        XYZ2AE(X[j],Y[j],Z[j],&A[j],&E[j]);
        XYZ2AE(X[k],Y[k],Z[k],&A[k],&E[k]);
    }
}

void LISA_detector_tensor(double L, double eplus[4][4], double ecross[4][4], double x[4], double y[4], double z[4], double k[4], double dplus[4][4], double dcross[4][4], double kdotr[4][4])
{

    /*   Spacecraft position and separation vector   */
    double r12[4]={0},r13[4]={0},r23[4]={0};

    //Unit separation vector from spacecrafts i to j
    r12[1] = (x[2] - x[1])/L;   r13[1] = (x[3] - x[1])/L;   r23[1] = (x[3] - x[2])/L;
    r12[2] = (y[2] - y[1])/L;   r13[2] = (y[3] - y[1])/L;   r23[2] = (y[3] - y[2])/L;
    r12[3] = (z[2] - z[1])/L;   r13[3] = (z[3] - z[1])/L;   r23[3] = (z[3] - z[2])/L;
    
    
    //Zero arrays to be summed
    dplus[1][2]  = dplus[1][3]  = dplus[2][1]  = dplus[2][3]  = dplus[3][1]  = dplus[3][2]  = 0.;
    dcross[1][2] = dcross[1][3] = dcross[2][1] = dcross[2][3] = dcross[3][1] = dcross[3][2] = 0.;
    
    //Convenient quantities d+ & dx
    for(int i=1; i<=3; i++)
    {
        for(int j=1; j<=3; j++)
        {
            dplus[1][2]  += r12[i]*r12[j]*eplus[i][j];   dcross[1][2] += r12[i]*r12[j]*ecross[i][j];
            dplus[2][3]  += r23[i]*r23[j]*eplus[i][j];   dcross[2][3] += r23[i]*r23[j]*ecross[i][j];
            dplus[1][3]  += r13[i]*r13[j]*eplus[i][j];   dcross[1][3] += r13[i]*r13[j]*ecross[i][j];
        }
    }
    
    //Make use of symmetry
    dplus[2][1] = dplus[1][2];  dcross[2][1] = dcross[1][2];
    dplus[3][2] = dplus[2][3];  dcross[3][2] = dcross[2][3];
    dplus[3][1] = dplus[1][3];  dcross[3][1] = dcross[1][3];
    
    //Zero arrays to be summed
    kdotr[1][2] = kdotr[1][3] = kdotr[2][1] = kdotr[2][3] = kdotr[3][1] = kdotr[3][2] = 0.;
    for(int i=1; i<=3; i++)
    {
        kdotr[1][2] += k[i]*r12[i];   kdotr[1][3] += k[i]*r13[i];   kdotr[2][3] += k[i]*r23[i];
    }
    
    //Make use of antisymmetry
    kdotr[2][1] = -kdotr[1][2];
    kdotr[3][1] = -kdotr[1][3];
    kdotr[3][2] = -kdotr[2][3];
}

void LISA_polarization_tensor(double costh, double phi, double eplus[4][4], double ecross[4][4], double k[4])
{
    /*   Gravitational Wave basis vectors   */
    double u[4],v[4];

    /*   Sky location angles   */
    double sinth = sqrt(1.0 - costh*costh);
    double cosph = cos(phi);
    double sinph = sin(phi);

    /*   Tensor construction for building slowly evolving LISA response   */
    //Gravitational Wave source basis vectors
    u[1] =  costh*cosph;  u[2] =  costh*sinph;  u[3] = -sinth;
    v[1] =  sinph;        v[2] = -cosph;        v[3] =  0.;
    k[1] = -sinth*cosph;  k[2] = -sinth*sinph;  k[3] = -costh;
    
    //GW polarization basis tensors
    /*
     * LDC convention:
     * https://gitlab.in2p3.fr/LISA/LDC/-/blob/develop/ldc/waveform/fastGB/GB.cc
     */
    for(int i=1;i<=3;i++)
    {
        for(int j=1;j<=3;j++)
        {
            eplus[i][j]  = v[i]*v[j] - u[i]*u[j];
            ecross[i][j] = u[i]*v[j] + v[i]*u[j];
        }
    }

}

/*
 * Copyright 2023 Tyson B. Littenberg & Robert Rosati
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

void glass_forward_complex_fft(double *data, int N)
{
    kiss_fft_cfg cfg = kiss_fft_alloc(N, 0, NULL, NULL); // 0 indicates forward FFT;
    kiss_fft_cpx *freqdata = malloc(N*sizeof(kiss_fft_cpx));
    kiss_fft_cpx *timedata = malloc(N*sizeof(kiss_fft_cpx));
    
    for(int i=0; i<N; i++)
    {
        timedata[i].r = data[2*i];
        timedata[i].i = data[2*i+1];
    }
    
    // Perform the forward FFT
    kiss_fft(cfg, timedata, freqdata);
    
    
    for(int i=0; i<N; i++)
    {
        data[2*i]   = freqdata[i].r;
        data[2*i+1] = freqdata[i].i;
    }
    
    
    // Clean up and free memory
    kiss_fft_free(cfg);
    free(freqdata);
    free(timedata);
}

/*
 * Copyright 2019 Tyson B. Littenberg & Neil J. Cornish
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

void ucb_waveform(struct Orbit *orbit, char *format, double T, double t0, double *params, int NParams, double *X, double *Y, double *Z, double *A, double *E, int BW, int NI)
{
    /*   Indicies   */
    int i,j,n;
    /*   Carrier frequency bin  */
    long q;
    /*   Bandwidth      */
    int BW2   = BW*2;
    double invBW2 = 1./(double)BW2;
    
    /*   Gravitational Wave location vector   */
    double k[4];
    /*   Polarization basis tensors   */
    double eplus[4][4], ecross[4][4];
    /*   Spacecraft position and separation vector   */
    double *x, *y, *z;
    /*   Dot products   */
    double kdotx[4]={0},kdotr[4][4];
    /*   Convenient quantities   */
    double dplus[4][4],dcross[4][4];
    /*   GW source parameters   */
    double phi, psi, amp, Aplus, Across, f0, dfdt, d2fdt2, phi0;
    double costh, cosi, cosps, sinps;
    /*   Time and distance variables   */
    double t, xi[4] = {0};
    /*   Gravitational wave frequency & ratio of f and transfer frequency f*  */
    double f[4] = {0},fonfs[4] = {0};
    /*   LISA response to slow terms (Real & Imaginary pieces)   */
    //Static quantities (Re and Im)
    double DPr, DPi, DCr, DCi;
    //Time varrying quantities (Re & Im) broken up into convenient segments
    double TR[4][4], TI[4][4];
    //Miscellaneous constants used to speed up calculations
    double df;
    /*   Fourier coefficients before FFT and after convolution  */
    //Time series of slowly evolving terms at each vertex
    double *data12, *data13, *data21, *data23, *data31, *data32;
    //Fourier coefficients of slowly evolving terms (numerical)
    double a12[BW2+3], a13[BW2+3], a21[BW2+3], a23[BW2+3], a31[BW2+3], a32[BW2+3];
    //Package cij's into proper form for TDI subroutines
    double ***d;
    
    /*   Allocating Arrays   */
    x = calloc(4,sizeof(double));
    y = calloc(4,sizeof(double));
    z = calloc(4,sizeof(double));
    
    data12 = calloc((BW2+1),sizeof(double));
    data21 = calloc((BW2+1),sizeof(double));
    data31 = calloc((BW2+1),sizeof(double));
    data13 = calloc((BW2+1),sizeof(double));
    data23 = calloc((BW2+1),sizeof(double));
    data32 = calloc((BW2+1),sizeof(double));
    
    d = malloc(sizeof(double**)*4);
    for(i=0; i<4; i++)
    {
        d[i] = malloc(sizeof(double*)*4);
        for(j=0; j<4; j++)
        {
            	d[i][j] = calloc((BW2+1),sizeof(double));
        }
    }
    
    /*   Gravitational Wave source parameters   */
    
    f0     = params[0]/T;
    costh  = params[1];
    phi    = params[2];
    amp    = exp(params[3]);
    cosi   = params[4];
    psi    = params[5];
    phi0   = params[6];
    dfdt   = 0.0;
    d2fdt2 = 0.0;
    if(NParams>7)
        dfdt   = params[7]/(T*T);
    if(NParams>8)
        d2fdt2 = params[8]/(T*T*T);
    
    //Calculate carrier frequency bin
    q = (long)(f0*T);
    
    //Calculate cos and sin of sky position, inclination, polarization
    cosps	= cos(2.*psi);
    sinps	= sin(2.*psi);
    
    //Calculate GW polarization amplitudes
    Aplus  =  amp*(1.+cosi*cosi)*0.5;
    Across = -amp*(cosi);
    
    df = PI2*(((double)q)/T);
    
    //Calculate constant pieces of transfer functions
    DPr =  Aplus*cosps;
    DPi = -Across*sinps;
    DCr = -Aplus*sinps;
    DCi = -Across*cosps;
    
    LISA_polarization_tensor(costh, phi, eplus, ecross, k);
    
    /* Main loop over signal bandwidth */
    for(n=1; n<=BW; n++)
    {
        //First time sample must be at t=0 for phasing
        t = t0 + T*(double)(n-1)/(double)BW;
        
        //Calculate position of each spacecraft at time t
        (*orbit->orbit_function)(orbit, t, x, y, z);
        
        //Form LISA detector tensor et al based on spacecraft and source location
        LISA_detector_tensor(orbit->L,eplus,ecross,x,y,z,k,dplus,dcross,kdotr);
        
        //Calculating LISA Transfer function
        for(i=1; i<=3; i++)
        {
            //Dot product of propogation vector with location of spacecrat i
            kdotx[i] = (x[i]*k[1]+y[i]*k[2]+z[i]*k[3])/CLIGHT;
            
            //Wave arrival time at spacecraft i
            xi[i] = t - kdotx[i];
            
            //Zeroeth order approximation to frequency at spacecraft i
            f[i] = f0;
            
            //First order in frequency
            if(NParams>7) f[i] += dfdt*xi[i];
            
            //Second order in frequency
            if(NParams>8) f[i] += 0.5*d2fdt2*xi[i]*xi[i];
            
            //Ratio of true frequency to transfer frequency
            fonfs[i] = f[i]/orbit->fstar;

            //Argument of complex exponentials
            /*
             * LDC phase parameter in key files is
             * -phi0, hence the -phi0 in arg2
             */
            double arg2 = PI2*f0*xi[i] - phi0 - df*t;
            
            
            //First order frequency evolution
            if(NParams>7) arg2 += M_PI*dfdt*xi[i]*xi[i];
            
            //Second order frequency evolution
            if(NParams>8) arg2 += (M_PI/3.0)*d2fdt2*xi[i]*xi[i]*xi[i];
            
            //Evolution of amplitude
            double aevol = 1.0;
            
            //First order amplitude evolution
            if(NParams>7) aevol += 0.66666666666666666666*dfdt/f0*xi[i];
            
            //Second order amplitude evolution
            //if(NParams>8) aevol += const.*d2fdt2*xi[i]*xi[i]/f0;
            
            for(j=1; j<=3; j++)
            {
                if(i!=j)
                {
                    //Argument of transfer function
                    /*
                     * Set to match Radler LDC convention
                     *
                     https://gitlab.in2p3.fr/LISA/LDC/-/blob/develop/ldc/waveform/fastGB/GB.cc
                     */
                    double arg1 = 0.5*fonfs[i]*(1.0 + kdotr[i][j]);
                    
                    //Transfer function
                    double sinc = 0.25*sin(arg1)/arg1;
                    
                    //Real and imaginary pieces of time series (no complex exponential)
                    double tran1r = aevol*(dplus[i][j]*DPr + dcross[i][j]*DCr);
                    double tran1i = aevol*(dplus[i][j]*DPi + dcross[i][j]*DCi);
                    
                    /*
                     * Set to match Sangria LDC convention
                     * which defines the GW as e(-i Phi)
                    */
                    //Real and imaginary components of complex exponential
                    double tran2r = cos(arg1 - arg2);
                    double tran2i = sin(arg1 - arg2);

                    //Real & Imaginary part of the slowly evolving signal
                    TR[i][j] = sinc*( tran1r*tran2r + tran1i*tran2i);
                    TI[i][j] = sinc*(-tran1r*tran2i + tran1i*tran2r);
                }
            }
        }
        
        //Fill  time series data arrays with slowly evolving signal->
        //dataij corresponds to fractional arm length difference yij
        j = 2*n;
        i = j-1;
        data12[i] = TR[1][2];   data21[i] = TR[2][1];   data31[i] = TR[3][1];
        data12[j] = TI[1][2];   data21[j] = TI[2][1];   data31[j] = TI[3][1];
        data13[i] = TR[1][3];   data23[i] = TR[2][3];   data32[i] = TR[3][2];
        data13[j] = TI[1][3];   data23[j] = TI[2][3];   data32[j] = TI[3][2];
    }
    
    /*   Numerical Fourier transform of slowly evolving signal */
    glass_forward_complex_fft(data12+1, BW);
    glass_forward_complex_fft(data21+1, BW);
    glass_forward_complex_fft(data31+1, BW);
    glass_forward_complex_fft(data13+1, BW);
    glass_forward_complex_fft(data23+1, BW);
    glass_forward_complex_fft(data32+1, BW);

    //Unpack arrays from fft and normalize
    for(i=1; i<=BW; i++)
    {
        j = i + BW;
        a12[i] = data12[j]*invBW2;  a21[i] = data21[j]*invBW2;  a31[i] = data31[j]*invBW2;
        a12[j] = data12[i]*invBW2;  a21[j] = data21[i]*invBW2;  a31[j] = data31[i]*invBW2;
        a13[i] = data13[j]*invBW2;  a23[i] = data23[j]*invBW2;  a32[i] = data32[j]*invBW2;
        a13[j] = data13[i]*invBW2;  a23[j] = data23[i]*invBW2;  a32[j] = data32[i]*invBW2;
    }
    
    /*   Renormalize so that the resulting time series is real   */
    for(i=1; i<=BW2; i++)
    {
        d[1][2][i] = a12[i];  d[2][1][i] = a21[i];  d[3][1][i] = a31[i];
        d[1][3][i] = a13[i];  d[2][3][i] = a23[i];  d[3][2][i] = a32[i];
    }
    
    /*   Call subroutines for synthesizing different TDI data channels  */
    if(strcmp("phase",format) == 0)
        LISA_tdi(orbit->L, orbit->fstar, T, d, f0, q, X-1, Y-1, Z-1, A-1, E-1, BW, NI);
    else if(strcmp("frequency",format) == 0)
        LISA_tdi_FF(orbit->L, orbit->fstar, T, d, f0, q, X-1, Y-1, Z-1, A-1, E-1, BW, NI);
    else if(strcmp("sangria",format) == 0)
        LISA_tdi_Sangria(orbit->L, orbit->fstar, T, d, f0, q, X-1, Y-1, Z-1, A-1, E-1, BW, NI);
    else
    {
        fprintf(stderr,"Unsupported data format %s",format);
        exit(1);
    }

    /*   Free Arrays   */
    free(x);
    free(y);
    free(z);
    
    free(data12);
    free(data21);
    free(data31);
    free(data13);
    free(data23);
    free(data32);
    
    for(i=0; i<4; i++)
    {
        for(j=0; j<4; j++)
        {
            free(d[i][j]);
        }
        free(d[i]);
    }
    free(d);
    
    
    return;
}
