// #includes {{{1
#include <assert.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>

#include "iio.h"

#ifndef M_PI
#define M_PI		3.14159265358979323846	/* pi */
#endif


// typedefs {{{1
typedef void (*generic_optical_flow)(
		float *out_u, float *out_v,
		float *in_a, float *in_b,
		int width, int height,
		void *data);

typedef void (*iterative_optical_flow)(
		float *out_u, float *out_v,
		float *in_a, float *in_b,
		float *in_u, float *in_v,
		int width, int height,
		void *data);

typedef float (*extension_operator_float)(
		float *image, int width, int height,
		int i, int j);

typedef float (*interpolation_operator_float)(
		float *image, int width, int height,
		float x, float y);

// utility functions {{{1
static float extend_float_image_constant(float *xx, int w, int h, int i, int j)
{
	float (*x)[w] = (void*)xx;
	if (i < 0) i = 0;
	if (j < 0) j = 0;
	if (i >= w) i = w - 1;
	if (j >= h) j = h - 1;
	return x[j][i];
}

static float cell_interpolate_bilinear(float a, float b, float c, float d,
					float x, float y)
{
	float r = 0;
	r += a*(1-x)*(1-y);
	r += b*(1-x)*(y);
	r += c*(x)*(1-y);
	r += d*(x)*(y);
	return r;
}

static float cell_interpolate_nearest(float a, float b, float c, float d,
					float x, float y)
{
	// return a;
	if (x<0.5) return y<0.5 ? a : b;
	else return y<0.5 ? c : d;
}

static float cell_interpolate(float a, float b, float c, float d,
					float x, float y, int method)
{
	switch(method) {
	case 0: return cell_interpolate_nearest(a, b, c, d, x, y);
	//case 1: return marchi(a, b, c, d, x, y);
	case 2: return cell_interpolate_bilinear(a, b, c, d, x, y);
	default: return 0;
	}
}

static float interpolate_float_image_bilinearly(float *x, int w, int h,
		float i, float j)
{
	int ii = i;
	int jj = j;
	extension_operator_float p = extend_float_image_constant;
	float a = p(x, w, h, ii  , jj  );
	float b = p(x, w, h, ii  , jj+1);
	float c = p(x, w, h, ii+1, jj  );
	float d = p(x, w, h, ii+1, jj+1);
	return cell_interpolate(a, b, c, d, i-ii, j-jj, 2);
}

static void downsa_v2(float *out, float *in,
		int outw, int outh, int inw, int inh)
{
	//fprintf(stderr, "\n\ndownsa_v2 (%d x %d) => (%d x %d)\n\n",
	//		inw,inh,outw,outh);
	assert(2*outw == inw);
	assert(2*outh == inh);

	float (*y)[outw] = (void*)out;
	float (*x)[inw] = (void*)in;

	for (int j = 0; j < outh; j++)
	for (int i = 0; i < outw; i++) {
		float g = 0;
		g += x[2*j+0][2*i+0];
		g += x[2*j+0][2*i+1];
		g += x[2*j+1][2*i+0];
		g += x[2*j+1][2*i+1];
		y[j][i] = g/4;
	}
}

static float MAGIC_SIGMA() { return 1.6; }

void downscale_image(float *out, float *in,
		int outw, int outh, int inw, int inh,
		float scalestep)
{
	if (scalestep == -2) {downsa_v2(out,in,outw,outh,inw,inh); return;}
	//fprintf(stderr, "downscale(%g): %dx%d => %dx%d\n",
	//		scalestep, inw, inh, outw, outh);

	assert(scalestep > 1);
	assert(scalestep * outw >= inw);
	//assert(scalestep * outw <= inw + 1);
	assert(scalestep * outh >= inh);
	//assert(scalestep * outh <= inh + 1);

	float factorx = inw/(float)outw;
	float factory = inh/(float)outh;

	float blur_size = MAGIC_SIGMA()*sqrt((factorx*factory-1)/3);

	fprintf(stderr, "blur_size = %g\n", blur_size);

	float *gin = malloc(inw * inh * sizeof(float));
	if (outw < inw || outh < inh) {
		void gblur_gray(float*, float*, int, int, float);
		gblur_gray(gin, in, inw, inh, blur_size);
	} else {
		assert(inw == outw);
		assert(inh == outh);
		for (int i = 0; i < inw*inh; i++)
			gin[i] = in[i];
	}

	// XXX ERROR FIXME
	// TODO: zoom by fourier, or zoom by bicubic interpolation
	interpolation_operator_float ev = interpolate_float_image_bilinearly;

	for (int j = 0; j < outh; j++)
	for (int i = 0; i < outw; i++)
	{
		float x = factorx*i;
		float y = factory*j;
		out[outw*j + i] = ev(gin, inw, inh, x, y);
	}

	free(gin);
}

#define FORI(n) for(int i=0;i<(n);i++)
#define FORJ(n) for(int j=0;j<(n);j++)
#define FORK(n) for(int k=0;k<(n);k++)
#define FORL(n) for(int l=0;l<(n);l++)

// wrapper around FFTW3 that computes the complex-valued Fourier transform
// of a real-valued image
static void fft_2dfloat(fftwf_complex *fx, float *x, int w, int h)
{
	fftwf_complex *a = fftwf_malloc(w*h*sizeof*a);

	//fprintf(stderr, "planning...\n");
	fftwf_plan p = fftwf_plan_dft_2d(h, w, a, fx,
						FFTW_FORWARD, FFTW_ESTIMATE);
	//fprintf(stderr, "...planned!\n");

	FORI(w*h) a[i] = x[i]; // complex assignment!
	fftwf_execute(p);

	fftwf_destroy_plan(p);
	fftwf_free(a);
	fftwf_cleanup();
}

// Wrapper around FFTW3 that computes the real-valued inverse Fourier transform
// of a complex-valued frequantial image.
// The input data must be hermitic.
static void ifft_2dfloat(float *ifx,  fftwf_complex *fx, int w, int h)
{
	fftwf_complex *a = fftwf_malloc(w*h*sizeof*a);
	fftwf_complex *b = fftwf_malloc(w*h*sizeof*b);

	//fprintf(stderr, "planning...\n");
	fftwf_plan p = fftwf_plan_dft_2d(h, w, a, b,
						FFTW_BACKWARD, FFTW_ESTIMATE);
	//fprintf(stderr, "...planned!\n");

	FORI(w*h) a[i] = fx[i];
	fftwf_execute(p);
	float scale = 1.0/(w*h);
	FORI(w*h) {
		fftwf_complex z = b[i] * scale;
		ifx[i] = crealf(z);
	}
	fftwf_destroy_plan(p);
	fftwf_free(a);
	fftwf_free(b);
	fftwf_cleanup();
}

static void pointwise_complex_multiplication(fftwf_complex *w,
		fftwf_complex *z, fftwf_complex *x, int n)
{
	FORI(n)
		w[i] = z[i] * x[i];
}

static void fill_2d_gaussian_image(float *gg, int w, int h, float inv_s)
{
	float (*g)[w] = (void *)gg;
	float alpha = inv_s*inv_s/(M_PI);

	FORJ(h) FORI(w) {
		float x = i < w/2 ? i : i - w;
		float y = j < h/2 ? j : j - h;
		float r = hypot(x, y);
		g[j][i] = alpha * exp(-r*r*inv_s*inv_s);
	}

	// if the kernel is too large, it escapes the domain, so the
	// normalization above must be corrected
	double m = 0;
	FORJ(h) FORI(w) m += g[j][i];
	FORJ(h) FORI(w) g[j][i] /= m;
}


// gaussian blur of a gray 2D image
void gblur_gray(float *y, float *x, int w, int h, float s)
{
	s = 1/s;

	fftwf_complex *fx = fftwf_malloc(w*h*sizeof*fx);
	fft_2dfloat(fx, x, w, h);

	float *g = malloc(w*h*sizeof*g);
	fill_2d_gaussian_image(g, w, h, s);

	fftwf_complex *fg = fftwf_malloc(w*h*sizeof*fg);
	fft_2dfloat(fg, g, w, h);

	pointwise_complex_multiplication(fx, fx, fg, w*h);
	ifft_2dfloat(y, fx, w, h);

	fftwf_free(fx);
	fftwf_free(fg);
	free(g);
}

