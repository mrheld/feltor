INCLUDE = -I/home/c744/c7441112/include -I$(UIBK_CUSP_INC)
INCLUDE += -I../ # other libraries

NVCC = nvcc
NVCCFLAGS = --compiler-options -Wall -arch=sm_20

CXX = g++
CFLAGS = -Wall -fopenmp
CINCLUDE = -I$(UIBK_CUDA_INC) -I$(UIBK_CUSP_INC)

#GLFLAGS=-lglfw -lrt -lGL -lGLEW -lpthread #you might add some libs here, check your glfw installation
GLFLAGS   = -lm #-lglfw -lXxf86vm -lXext   -lX11 -lGLU   -lGL -lpthread -lm -lGLEW 
LIBS = cusp_eigen.o

all: version cusp_thrust arakawa_t blas1_t \
	cg2d_b cg2d_t cg_t \
	dlt2d_b dlt2d_t dlt_b dlt_t dx_t \
	evaluation_b evaluation_t gamma_t host_window_t \
	laplace_b laplace_t \
	operator_matrix_t operator_t polarisation2d_b polarisation2d_t polarisation_b polarisation_t \
	preconditioner2d_b preconditioner2d_t preconditioner_b preconditioner_t \
	rk2d_t rk_t tensor_b tensor_t toefl_b toefl_t xspacelib_t



version: version.cu 
	$(NVCC) $< $(NVCCFLAGS) -o $@
	./$@

cusp_thrust: cusp_thrust.cu 
	$(NVCC) $< $(NVCCFLAGS) -o $@
	./$@

host_window_t: host_window_t.cu
	$(NVCC) $< $(NVCCFLAGS) -o $@ $(GLFLAGS) $(INCLUDE)


blas%_t: blas%_t.cu blas%.h
	$(NVCC) $< $(NVCCFLAGS) -o $@ 

%.o: %.cpp
	$(CXX) -O3 -c $< $(CFLAGS) $(CINCLUDE) $(INCLUDE) 

%_t: %_t.cu %.cuh $(LIBS)
	$(NVCC) -DDG_DEBUG $< $(LIBS) -o $@ $(NVCCFLAGS) $(GLFLAGS)  $(INCLUDE) 

%_b: %_b.cu %.cuh $(LIBS)
	$(NVCC) -O2 -w  $< -o $@ $(NVCCFLAGS) $(INCLUDE) $(LIBS) $(GLFLAGS)
	echo "Attention: No compiler warnings are issued!" 

%2d_t: %2d_t.cu %.cuh $(LIBS)
	$(NVCC) -DDG_DEBUG $< -o $@ $(NVCCFLAGS) $(LIBS) $(INCLUDE)

%2d_b: %2d_b.cu %.cuh $(LIBS)
	$(NVCC) -O2 -w  $< -o $@ $(NVCCFLAGS) $(LIBS) $(INCLUDE)
	echo "Attention: No compiler warnings are issued!" 

.PHONY: clean doc

doc: 
	doxygen Doxyfile

clean:
	rm -f *_t *_b cusp_thrust version