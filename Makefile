
CXXFLAGS=-O3 -std=c++14
LDFLAGS=-fopenmp
LDLIBS=-ltiff -ljpeg -lpng -lfftw3f -lm -lstdc++

all: stochastic_deconvolution

stochastic_deconvolution: stochastic_deconvolution.o image.o iio.o downscale_image.o

clean:
	-rm image.o iio.o downscale_image.o

