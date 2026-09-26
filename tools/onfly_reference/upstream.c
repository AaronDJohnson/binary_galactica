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

static void hplus_and_hcross(double t, double phase, double amp,  double Aplus, double Across, double cos2psi, double sin2psi,  double *hp, double *hc, double *hpf, double *hcf)
{
    double cp = cos(phase);
    double sp = sin(phase);

    *hp  = amp * ( Aplus*cos2psi*cp  + Across*sin2psi*sp );
    *hc  = amp * ( Across*cos2psi*sp - Aplus*sin2psi*cp  );
    
    *hpf = amp * (-Aplus*cos2psi*sp  + Across*sin2psi*cp );
    *hcf = amp * ( Across*cos2psi*cp + Aplus*sin2psi*sp  );              
}

static void LISA_TDI_spline(double *M, double *Mf, int a, int b, int c, double* tarray, int n, struct CubicSpline *amp_spline, struct CubicSpline *freq_spline, struct CubicSpline *phase_spline, double Aplus, double Across, double cos2psi, double sin2psi, double *App, double *Apm, double *Acp, double *Acm, double *kr, double *Larm)
{
    double t, f, amp, phase;
    double hp, hc, hpf, hcf;

    M[n] = 0.0;
    Mf[n] = 0.0;
    
    if(freq_spline) /* mbh */
    {
        //For TDI we want the overall amplitude scaled out of the waveform as it passes through zero
        amp = 1.0; //spline_interpolation(amp_spline,  tarray[n]);
        f   = spline_interpolation(freq_spline, tarray[n]);
    }

    t = tarray[n] - kr[a]-2.0*Larm[c]-2.0*Larm[b];
    if(phase_spline) /* ucb */
    {
        amp   = spline_interpolation(amp_spline,   t);
        phase = spline_interpolation(phase_spline, t);
    }
    if(freq_spline) phase = PI2*f*t; /* mbh */
    hplus_and_hcross(t, phase, amp, Aplus, Across, cos2psi, sin2psi, &hp, &hc, &hpf, &hcf);
    M[n] += hp*Apm[b]+hc*Acm[b];
    M[n] -= hp*App[c]+hc*Acp[c];
    Mf[n] += hpf*Apm[b]+hcf*Acm[b];
    Mf[n] -= hpf*App[c]+hcf*Acp[c];
    
    t = tarray[n] - kr[b]-Larm[c]-2.0*Larm[b];
    if(phase_spline) /* ucb */
    {
        amp   = spline_interpolation(amp_spline,   t);
        phase = spline_interpolation(phase_spline, t);
    }
    if(freq_spline) phase = PI2*f*t; /* mbh */
    hplus_and_hcross(t, phase, amp, Aplus, Across, cos2psi, sin2psi, &hp, &hc, &hpf, &hcf);
    M[n] -= hp*Apm[c]+hc*Acm[c];
    M[n] += hp*App[c]+hc*Acp[c];
    Mf[n] -= hpf*Apm[c]+hcf*Acm[c];
    Mf[n] += hpf*App[c]+hcf*Acp[c];
    
    t = tarray[n] - kr[c]-Larm[b]-2.0*Larm[c];
    if(phase_spline) /* ucb */
    {
        amp   = spline_interpolation(amp_spline,   t);
        phase = spline_interpolation(phase_spline, t);
    }
    if(freq_spline) phase = PI2*f*t; /* mbh */
    hplus_and_hcross(t, phase, amp, Aplus, Across, cos2psi, sin2psi, &hp, &hc, &hpf, &hcf);
    M[n] += hp*App[b]+hc*Acp[b];
    M[n] -= hp*Apm[b]+hc*Acm[b];
    Mf[n] += hpf*App[b]+hcf*Acp[b];
    Mf[n] -= hpf*Apm[b]+hcf*Acm[b];
    
    t = tarray[n] - kr[a]-2.0*Larm[b];
    if(phase_spline) /* ucb */
    {
        amp   = spline_interpolation(amp_spline,   t);
        phase = spline_interpolation(phase_spline, t);
    }
    if(freq_spline) phase = PI2*f*t; /* mbh */
    hplus_and_hcross(t, phase, amp, Aplus, Across, cos2psi, sin2psi, &hp, &hc, &hpf, &hcf);
    M[n] -= hp*Apm[b]+hc*Acm[b];
    M[n] += hp*Apm[c]+hc*Acm[c];
    Mf[n] -= hpf*Apm[b]+hcf*Acm[b];
    Mf[n] += hpf*Apm[c]+hcf*Acm[c];
    
    t = tarray[n] - kr[a]-2.0*Larm[c];
    if(phase_spline) /* ucb */
    {
        amp   = spline_interpolation(amp_spline,   t);
        phase = spline_interpolation(phase_spline, t);
    }
    if(freq_spline) phase = PI2*f*t; /* mbh */
    hplus_and_hcross(t, phase, amp, Aplus, Across, cos2psi, sin2psi, &hp, &hc, &hpf, &hcf);
    M[n] += hp*App[c]+hc*Acp[c];
    M[n] -= hp*App[b]+hc*Acp[b];
    Mf[n] += hpf*App[c]+hcf*Acp[c];
    Mf[n] -= hpf*App[b]+hcf*Acp[b];
    
    t = tarray[n] - kr[c]- Larm[b];
    if(phase_spline) /* ucb */
    {
        amp   = spline_interpolation(amp_spline,   t);
        phase = spline_interpolation(phase_spline, t);
    }
    if(freq_spline) phase = PI2*f*t; /* mbh */
    hplus_and_hcross(t, phase, amp, Aplus, Across, cos2psi, sin2psi, &hp, &hc, &hpf, &hcf);
    M[n] -= hp*App[b]+hc*Acp[b];
    M[n] += hp*Apm[b]+hc*Acm[b];
    Mf[n] -= hpf*App[b]+hcf*Acp[b];
    Mf[n] += hpf*Apm[b]+hcf*Acm[b];
    
    t = tarray[n] - kr[b]- Larm[c];
    if(phase_spline) /* ucb */
    {
        amp   = spline_interpolation(amp_spline,   t);
        phase = spline_interpolation(phase_spline, t);
    }
    if(freq_spline) phase = PI2*f*t; /* mbh */
    hplus_and_hcross(t, phase, amp, Aplus, Across, cos2psi, sin2psi, &hp, &hc, &hpf, &hcf);
    M[n] += hp*Apm[c]+hc*Acm[c];
    M[n] -= hp*App[c]+hc*Acp[c];
    Mf[n] += hpf*Apm[c]+hcf*Acm[c];
    Mf[n] -= hpf*App[c]+hcf*Acp[c];
    
    t = tarray[n] - kr[a];
    if(phase_spline) /* ucb */
    {
        amp   = spline_interpolation(amp_spline,   t);
        phase = spline_interpolation(phase_spline, t);
    }
    if(freq_spline) phase = PI2*f*t; /* mbh */
    hplus_and_hcross(t, phase, amp, Aplus, Across, cos2psi, sin2psi, &hp, &hc, &hpf, &hcf);
    M[n] -= hp*Apm[c]+hc*Acm[c];
    M[n] += hp*App[b]+hc*Acp[b];
    Mf[n] -= hpf*Apm[c]+hcf*Acm[c];
    Mf[n] += hpf*App[b]+hcf*Acp[b];    
}

void LISA_spline_response(struct Orbit *orbit, double *tarray, int N, double costh, double phi, double cosi, double psi, struct CubicSpline *amp_spline, struct CubicSpline *freq_spline, struct CubicSpline *phase_spline, double *phase_ref, struct TDI *tdi_amp, struct TDI *tdi_phase)
{
    /* work space for TDI response */
    struct TDI *R = malloc(sizeof(struct TDI));
    struct TDI *Rf = malloc(sizeof(struct TDI)); //_f has a phase flip for building up the full response later?
    alloc_tdi(R,N,3);
    alloc_tdi(Rf,N,3);

    /* aliases for TDI response */
    double *X = R->X;
    double *Y = R->Y;
    double *Z = R->Z;
    double *Xf = Rf->X;
    double *Yf = Rf->Y;
    double *Zf = Rf->Z;

    /*   Indicies    */
    int i,j;

    double k[4]; 
    double kdotr[3], kdotn[3];
    double dplus[3], dcross[3];
    
    /*   Polarization basis tensors   */
    double eplus[4][4], ecross[4][4];
    
    double cos2psi, sin2psi;
    double t;
    double Aplus, Across;
    double *App, *Apm, *Acp, *Acm;
    
    /*   Allocating Arrays   */
            
    App = malloc(sizeof(double)*3); 
    Apm = malloc(sizeof(double)*3);
    Acp = malloc(sizeof(double)*3);
    Acm = malloc(sizeof(double)*3);
    
    cos2psi = cos(2.*psi);  
    sin2psi = sin(2.*psi);
    
    Aplus = 0.5*(1.+cosi*cosi);
    Across = -cosi;

    LISA_polarization_tensor(costh, phi, eplus, ecross, k);


    double x[3][3];
    double n[3][3];
    double L[3];

    for(int m=0; m<N; m++)
    {
        
        t = tarray[m];

        // position vectors for each spacecraft (converted to seconds)
        for(i=0; i<3; i++)
        {
            x[i][0] = spline_interpolation(orbit->dx[i],t)/CLIGHT;
            x[i][1] = spline_interpolation(orbit->dy[i],t)/CLIGHT;
            x[i][2] = spline_interpolation(orbit->dz[i],t)/CLIGHT;
        }

        // arm vectors
        for(i=0; i<3; i++)
        {
            n[0][i] = x[1][i] - x[2][i];
            n[1][i] = x[2][i] - x[0][i];
            n[2][i] = x[0][i] - x[1][i];
        }

        //arm lengths
        for(i=0; i<3; i++) 
        {
            L[i]=0.0;
            for(j=0; j<3; j++)  L[i] += n[i][j]*n[i][j];
            L[i] = sqrt(L[i]);
        }

        //normalize arm vectors        
        for(i=0; i<3; i++)
            for(j=0;j<3;j++)
                n[i][j] /= L[i];
        


        // k dot r_i (source direction spacecraft locations)
        for(i=0; i<3; i++)
        {
            kdotr[i] = 0.0;
            for(j=0; j<3; j++)
                kdotr[i] += k[j+1] * x[i][j];
        }
        
        // k dot n_i (source direction w/ arm vectors)
        for(i=0; i<3; i++)
        {
            kdotn[i] = 0.0;
            for(j=0; j<3; j++)
                kdotn[i] += k[j+1] * n[i][j];
        }        
        
        //Antenna primitives
        for(i=0; i<3; i++)
        {
            dplus[i] = 0.0;
            dcross[i] = 0.0;
            for(j=0; j<3; j++)
            {
                for(int l=0; l<3; l++)
                {
                    dplus[i]  += (n[i][j] * n[i][l]) * eplus[j+1][l+1];
                    dcross[i] += (n[i][j] * n[i][l]) * ecross[j+1][l+1];
                }
            }
        }
        
        //Full Antenna patterns
        for(i=0; i<3; i++)
        {
            App[i] = dplus[i]  / (1.0 + kdotn[i]);
            Apm[i] = dplus[i]  / (1.0 - kdotn[i]);
            Acp[i] = dcross[i] / (1.0 + kdotn[i]);
            Acm[i] = dcross[i] / (1.0 - kdotn[i]);
        }
        
        /* build X, Y, Z responses */
        LISA_TDI_spline(X, Xf, 0, 1, 2, tarray, m, amp_spline, freq_spline, phase_spline, Aplus, Across, cos2psi, sin2psi, App, Apm, Acp, Acm, kdotr, L);
        LISA_TDI_spline(Y, Yf, 1, 2, 0, tarray, m, amp_spline, freq_spline, phase_spline, Aplus, Across, cos2psi, sin2psi, App, Apm, Acp, Acm, kdotr, L);
        LISA_TDI_spline(Z, Zf, 2, 0, 1, tarray, m, amp_spline, freq_spline, phase_spline, Aplus, Across, cos2psi, sin2psi, App, Apm, Acp, Acm, kdotr, L);

    }//end loop over samples m
    
    /*
    Separate TDI responses back into terms of phase and amplitude
    */

    //extract_amplitude_and_phase() is removing the carrier phase
    extract_amplitude_and_phase(N, tdi_amp->X, tdi_phase->X, R->X, Rf->X, phase_ref);
    extract_amplitude_and_phase(N, tdi_amp->Y, tdi_phase->Y, R->Y, Rf->Y, phase_ref);
    extract_amplitude_and_phase(N, tdi_amp->Z, tdi_phase->Z, R->Z, Rf->Z, phase_ref);
    unwrap_phase(N, tdi_phase->X);
    unwrap_phase(N, tdi_phase->Y);
    unwrap_phase(N, tdi_phase->Z);
    
    free(App);
    free(Apm);
    free(Acp); 
    free(Acm);
    
    free_tdi(R);
    free_tdi(Rf);

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

void extract_amplitude_and_phase(int Ns, double *As, double *Dphi, double *M, double *Mf, double *phiR)
{

    int i;
    double v;
    double dA1, dA2, dA3;
        
    double *flip  = double_vector(Ns);
    double *pjump = double_vector(Ns);
    
    
    for(i=0; i<Ns ;i++)
    {
        As[i] = sqrt(M[i]*M[i]+Mf[i]*Mf[i]);
    }
    
    // This catches sign flips in the amplitude. Can't catch flips at either end of array
    flip[0]  = 1.0;
    pjump[0] = 0.0;

    i = 1;
    do
    {
        flip[i] = flip[i-1];
        pjump[i] = pjump[i-1];
        
        //local min
        if((As[i] < As[i-1]) && (As[i] < As[i+1]))
        {
            dA1 =  As[i+1] + As[i-1] - 2.0*As[i];  // regular second derivative
            dA2 = -As[i+1] + As[i-1] - 2.0*As[i];  // second derivative if i+1 first negative value
            dA3 = -As[i+1] + As[i-1] + 2.0*As[i];  // second derivative if i first negative value

            if(fabs(dA2/dA1) < 0.1)
            {
                flip[i+1]  = -1.0*flip[i];
                pjump[i+1] = pjump[i]+M_PI;
                i++; // skip an extra place since i+1 already dealt with
            }
            if(fabs(dA3/dA1) < 0.1)
            {
                flip[i]  = -1.0*flip[i-1];
                pjump[i] = pjump[i-1]+M_PI;
            }
        }
        
        i++;
        
    }while(i < Ns-1);
    
    flip[Ns-1]  = flip[Ns-2];
    pjump[Ns-1] = pjump[Ns-2];
    
    
    for(i=0; i<Ns ;i++)
    {
        As[i] = flip[i]*As[i];
        v = remainder(phiR[i], PI2);
        Dphi[i] = -atan2(Mf[i],M[i])+pjump[i]-v;
    }
    
    free_double_vector(flip);
    free_double_vector(pjump);

}

void unwrap_phase(int N, double *phase)
{
    double u, v, q;
    int i;
    
    v = phase[0];
    for(i=0; i<N ;i++)
    {
        u = phase[i];
        q = rint(fabs(u-v)/(PI2));
        if(q > 0.0)
        {
           if(v > u) u += q*PI2;
           else      u -= q*PI2;
        }
        v = u;
        phase[i] = u;
    }
}
