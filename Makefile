
CXXFLAGS=-O3 -std=c++14
LDFLAGS=-fopenmp -ltiff -ljpeg -lpng -lfftw3f -O3

all: stochastic_deconvolution

stochastic_deconvolution: stochastic_deconvolution.cpp image.o iio.o downscale_image.o

