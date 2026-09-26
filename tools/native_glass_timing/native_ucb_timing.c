/*
 * Timing harness for GLASS commit 079c6a3.
 *
 * This deliberately compares the two functions that actually exist upstream:
 * ucb_waveform (banded Fourier XYZ/A/E) and ucb_waveform_wavelet (WDM XYZ).
 * They do not have identical output representations, so this is a cost
 * comparison, not a spectral-equality test.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include <glass_utils.h>
#include <glass_ucb.h>
#include <glass_noise.h>

#define BINARY_YEAR 31457280.0

static double now_seconds(void)
{
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return value.tv_sec + 1.0e-9 * value.tv_nsec;
}

static double checksum(const double *values, int count)
{
    double sum = 0.0;
    for(int i = 0; i < count; i++) sum += values[i] * (double)(i + 1);
    return sum;
}

static double time_response(struct Orbit *orbit, double *params, double duration, double epoch, int repeats)
{
    const int n = orbit->Norb;
    double *times = double_vector(n), *shifted = double_vector(n);
    double *amplitude = double_vector(n), *phase = double_vector(n), *reference = double_vector(n);
    for(int i=0; i<n; i++) times[i] = epoch + i*duration/(n-1);
    ucb_barycenter_waveform(params,n,orbit->t,phase,amplitude,duration);
    struct CubicSpline *a = alloc_cubic_spline(n), *p = alloc_cubic_spline(n);
    initialize_cubic_spline(a,orbit->t,amplitude,SPLINE_EVEN_SAMPLED);
    initialize_cubic_spline(p,orbit->t,phase,SPLINE_EVEN_SAMPLED);
    LISA_spacecraft_to_barycenter_time(orbit,params[1],params[2],times,shifted,n,-1);
    for(int i=0; i<n; i++) reference[i] = spline_interpolation(p,shifted[i]);
    struct TDI *ta=malloc(sizeof(*ta)), *tp=malloc(sizeof(*tp));
    alloc_tdi(ta,n,3); alloc_tdi(tp,n,3);
    LISA_spline_response(orbit,times,n,params[1],params[2],params[4],params[5],a,NULL,p,reference,ta,tp);
    double start=now_seconds();
    for(int r=0; r<repeats; r++)
        LISA_spline_response(orbit,times,n,params[1],params[2],params[4],params[5],a,NULL,p,reference,ta,tp);
    double elapsed=(now_seconds()-start)/repeats;
    for(int i=0; i<n; i++)
        if(!isfinite(ta->X[i]) || !isfinite(ta->Y[i]) || !isfinite(ta->Z[i]) ||
           !isfinite(tp->X[i]) || !isfinite(tp->Y[i]) || !isfinite(tp->Z[i])) abort();
    free_tdi(ta); free_tdi(tp); free_cubic_spline(a); free_cubic_spline(p);
    free(times); free(shifted); free(amplitude); free(phase); free(reference);
    return elapsed;
}

static void print_number(double value)
{
    printf("%.17g", value);
}

static void print_vector(const double *values, int n)
{
    printf("[");
    for(int i=0; i<n; i++) {
        if(i) printf(",");
        print_number(values[i]);
    }
    printf("]");
}

static void print_positions(struct Orbit *orbit, const double *times, int n)
{
    printf("[");
    for(int spacecraft=0; spacecraft<3; spacecraft++) {
        if(spacecraft) printf(",");
        printf("[");
        for(int i=0; i<n; i++) {
            if(i) printf(",");
            printf("[");
            print_number(spline_interpolation(orbit->dx[spacecraft], times[i]));
            printf(",");
            print_number(spline_interpolation(orbit->dy[spacecraft], times[i]));
            printf(",");
            print_number(spline_interpolation(orbit->dz[spacecraft], times[i]));
            printf("]");
        }
        printf("]");
    }
    printf("]");
}

static void print_complex_response(const double *amplitude, const double *phase, int n)
{
    printf("[");
    for(int i=0; i<n; i++) {
        if(i) printf(",");
        printf("[");
        print_number(amplitude[i]*cos(phase[i]));
        printf(",");
        print_number(amplitude[i]*sin(phase[i]));
        printf("]");
    }
    printf("]");
}

static void print_complex_spectrum(const double *values, int bw)
{
    printf("[");
    for(int i=0; i<bw; i++) {
        if(i) printf(",");
        printf("[");
        print_number(values[2*i+1]);
        printf(",");
        print_number(values[2*i+2]);
        printf("]");
    }
    printf("]");
}

static void set_params(double *params, double frequency, double duration)
{
    params[0] = frequency * duration + 0.25;
    params[1] = 0.27;
    params[2] = 1.11;
    params[3] = log(1.0e-22);
    params[4] = -0.37;
    params[5] = 0.63;
    params[6] = 0.41;
    params[7] = 0.2;
}

static int valid_case(double frequency, double years, int bandwidth)
{
    return isfinite(frequency) && frequency > 0.0 &&
           frequency < 1.0/(2.0*LISA_CADENCE) &&
           (years == 0.25 || years == 1.0 || years == 4.0) &&
           bandwidth >= 2 && !(bandwidth & (bandwidth-1));
}

static void dump_response(double frequency, double years, int bandwidth)
{
    const double duration = years * BINARY_YEAR;
    const double epoch = 12345.0;
    (void)bandwidth;
    struct Orbit *orbit = malloc(sizeof(*orbit));
    double params[8];
    set_params(params, frequency, duration);
    initialize_analytic_orbit(orbit);
    initialize_interpolated_analytic_orbits(orbit, duration, epoch);

    const int n = orbit->Norb;
    double *times = double_vector(n), *shifted = double_vector(n);
    double *amplitude = double_vector(n), *phase = double_vector(n), *reference = double_vector(n);
    for(int i=0; i<n; i++) times[i] = epoch + i*duration/(n-1);
    ucb_barycenter_waveform(params, n, orbit->t, phase, amplitude, duration);
    struct CubicSpline *a = alloc_cubic_spline(n), *p = alloc_cubic_spline(n);
    initialize_cubic_spline(a, orbit->t, amplitude, SPLINE_EVEN_SAMPLED);
    initialize_cubic_spline(p, orbit->t, phase, SPLINE_EVEN_SAMPLED);
    LISA_spacecraft_to_barycenter_time(orbit, params[1], params[2], times, shifted, n, -1);
    for(int i=0; i<n; i++) reference[i] = spline_interpolation(p, shifted[i]);
    struct TDI *ta = malloc(sizeof(*ta)), *tp = malloc(sizeof(*tp));
    alloc_tdi(ta, n, 3); alloc_tdi(tp, n, 3);
    LISA_spline_response(orbit, times, n, params[1], params[2], params[4], params[5],
                         a, NULL, p, reference, ta, tp);

    printf("{\"kind\":\"response\",\"T\":"); print_number(duration);
    printf(",\"t0\":"); print_number(epoch);
    printf(",\"params\":"); print_vector(params, 8);
    printf(",\"times\":"); print_vector(times, n);
    printf(",\"positions\":"); print_positions(orbit, times, n);
    printf(",\"xyz\":[");
    print_complex_response(ta->X, tp->X, n); printf(",");
    print_complex_response(ta->Y, tp->Y, n); printf(",");
    print_complex_response(ta->Z, tp->Z, n);
    printf("],\"reference_phase_sc\":"); print_vector(reference, n);
    printf(",\"geometry\":{\"L_m\":"); print_number(orbit->L);
    printf(",\"fstar_hz\":"); print_number(orbit->fstar);
    printf(",\"orbit_samples\":%d}}\n", n);

    free_tdi(ta); free_tdi(tp); free_cubic_spline(a); free_cubic_spline(p);
    free(times); free(shifted); free(amplitude); free(phase); free(reference);
    free_orbit(orbit);
}

static void dump_fourier(double frequency, double years, int bandwidth)
{
    const double duration = years * BINARY_YEAR;
    const double epoch = 12345.0;
    struct Orbit *orbit = malloc(sizeof(*orbit));
    double params[8];
    double *channels[5];
    set_params(params, frequency, duration);
    initialize_analytic_orbit(orbit);
    initialize_interpolated_analytic_orbits(orbit, duration, epoch);
    for(int channel=0; channel<5; channel++) {
        channels[channel] = calloc(2*bandwidth+1, sizeof(*channels[channel]));
        if(channels[channel] == NULL) abort();
    }
    ucb_waveform(orbit, "sangria", duration, epoch, params, 8,
                 channels[0]+1, channels[1]+1, channels[2]+1,
                 channels[3]+1, channels[4]+1, bandwidth, 3);
    printf("{\"kind\":\"fourier\",\"T\":"); print_number(duration);
    printf(",\"t0\":"); print_number(epoch);
    printf(",\"params\":"); print_vector(params, 8);
    printf(",\"BW\":%d,\"qmin\":%ld,\"channels\":{\"X\":", bandwidth,
           (long)params[0] - bandwidth/2);
    print_complex_spectrum(channels[0], bandwidth); printf(",\"Y\":");
    print_complex_spectrum(channels[1], bandwidth); printf(",\"Z\":");
    print_complex_spectrum(channels[2], bandwidth); printf(",\"A\":");
    print_complex_spectrum(channels[3], bandwidth); printf(",\"E\":");
    print_complex_spectrum(channels[4], bandwidth);
    printf("},\"geometry\":{\"L_m\":"); print_number(orbit->L);
    printf(",\"fstar_hz\":"); print_number(orbit->fstar);
    printf(",\"orbit_samples\":%d}}\n", orbit->Norb);
    for(int channel=0; channel<5; channel++) free(channels[channel]);
    free_orbit(orbit);
}

static void time_case(double frequency, double years, int bandwidth, int repeats)
{
    const double duration = years * BINARY_YEAR;
    const double epoch = 12345.0;
    const int samples = (int)(duration / LISA_CADENCE);
    double params[8];
    set_params(params, frequency, duration);
    struct Orbit *orbit = malloc(sizeof(*orbit));
    struct Wavelets *wavelets = malloc(sizeof(*wavelets));
    struct TDI *wavelet_xyz = malloc(sizeof(*wavelet_xyz));
    int *active = int_vector(samples);
    int active_count = 0;
    double *channels[5];

    initialize_analytic_orbit(orbit);
    initialize_interpolated_analytic_orbits(orbit, duration, epoch);
    initialize_wavelet(wavelets, duration);
    alloc_tdi(wavelet_xyz, samples, 3);
    for(int channel = 0; channel < 5; channel++) {
        /* Preserve ucb_waveform's X-1 pointer arithmetic inside allocation. */
        channels[channel] = calloc(2 * bandwidth + 1, sizeof(*channels[channel]));
        if(channels[channel] == NULL) abort();
    }

    /* Warm both paths after all orbit/WDM and output allocation is complete. */
    ucb_waveform_wavelet(
        orbit, wavelets, duration, epoch, params, active, &active_count,
        wavelet_xyz->X, wavelet_xyz->Y, wavelet_xyz->Z
    );
    ucb_waveform(
        orbit, "sangria", duration, epoch, params, 8,
        channels[0] + 1, channels[1] + 1, channels[2] + 1,
        channels[3] + 1, channels[4] + 1, bandwidth, 3
    );

    double start = now_seconds();
    for(int iteration = 0; iteration < repeats; iteration++) {
        ucb_waveform_wavelet(
            orbit, wavelets, duration, epoch, params, active, &active_count,
            wavelet_xyz->X, wavelet_xyz->Y, wavelet_xyz->Z
        );
    }
    double wavelet_seconds = (now_seconds() - start) / repeats;

    start = now_seconds();
    for(int iteration = 0; iteration < repeats; iteration++) {
        ucb_waveform(
            orbit, "sangria", duration, epoch, params, 8,
            channels[0] + 1, channels[1] + 1, channels[2] + 1,
            channels[3] + 1, channels[4] + 1, bandwidth, 3
        );
    }
    double fourier_seconds = (now_seconds() - start) / repeats;

    double response_seconds=time_response(orbit,params,duration,epoch,repeats);
    if(active_count % wavelets->NT) abort();
    int layer_count = active_count / wavelets->NT;
    if(active_count<=0 || active_count>samples) abort();
    for(int i=0; i<active_count; i++) {
        int k=active[i];
        if(k<0 || k>=samples || !isfinite(wavelet_xyz->X[k]) ||
           !isfinite(wavelet_xyz->Y[k]) || !isfinite(wavelet_xyz->Z[k])) abort();
    }
    for(int channel=0; channel<5; channel++)
        for(int i=1; i<=2*bandwidth; i++)
            if(!isfinite(channels[channel][i])) abort();
    printf(
        "{\"frequency_hz\":%.17g,\"years\":%.17g,\"BW\":%d,"
        "\"repeats\":%d,\"response_samples\":%d,\"wavelet_time_pixels\":%d,\"wavelet_layers\":%d,"
        "\"response_seconds\":%.17g,\"wavelet_xyz_seconds\":%.17g,"
        "\"fourier_xyzae_seconds\":%.17g,\"wavelet_active_pixels\":%d,"
        "\"wavelet_checksum\":%.17g,\"fourier_checksum\":%.17g}\n",
        frequency, years, bandwidth, repeats, orbit->Norb, wavelets->NT, layer_count, response_seconds, wavelet_seconds, fourier_seconds,
        active_count, checksum(wavelet_xyz->X, samples),
        checksum(channels[0] + 1, 2 * bandwidth)
    );

    for(int channel = 0; channel < 5; channel++) free(channels[channel]);
    free(active);
    free_tdi(wavelet_xyz);
    for(int n=0; n<wavelets->fdot_steps; n++) free(wavelets->table[n]);
    free(wavelets->table);
    free(wavelets->n_table);
    free(wavelets->fdot);
    free(wavelets->window);
    free(wavelets->window_freq_forward);
    free(wavelets);
    free_orbit(orbit);
}

int main(int argc, char **argv)
{
    int repeats = 100;
    if(argc==5 && (strcmp(argv[1], "--dump-response") == 0 ||
                    strcmp(argv[1], "--dump-fourier") == 0)) {
        double f = atof(argv[2]), years = atof(argv[3]);
        int bw = atoi(argv[4]);
        if(!valid_case(f, years, bw)) return 2;
        if(strcmp(argv[1], "--dump-response") == 0) dump_response(f, years, bw);
        else dump_fourier(f, years, bw);
        return 0;
    }
    if(argc==6 && strcmp(argv[1],"--case")==0) {
        double f=atof(argv[2]), years=atof(argv[3]);
        int bw=atoi(argv[4]), n=atoi(argv[5]);
        if(!valid_case(f, years, bw) || n<1) return 2;
        time_case(f,years,bw,n);
        return 0;
    }
    if(argc == 3 && strcmp(argv[1], "--repeats") == 0) repeats = atoi(argv[2]);
    if(repeats < 1 || (argc != 1 && argc != 3)) {
        fprintf(stderr, "usage: %s [--repeats positive-integer]\n"
                "       %s --case f years BW repeats\n"
                "       %s --dump-response f years BW\n"
                "       %s --dump-fourier f years BW\n", argv[0], argv[0], argv[0], argv[0]);
        return 2;
    }
    for(int year_index = 0; year_index < 3; year_index++) {
        const double years[] = {0.25, 1.0, 4.0};
        for(int frequency_index = 0; frequency_index < 3; frequency_index++) {
            const double frequencies[] = {0.001, 0.005, 0.03};
            for(int bandwidth_index = 0; bandwidth_index < 3; bandwidth_index++) {
                const int bandwidths[] = {128, 512, 2048};
                time_case(
                    frequencies[frequency_index], years[year_index],
                    bandwidths[bandwidth_index], repeats
                );
            }
        }
    }
    return 0;
}
