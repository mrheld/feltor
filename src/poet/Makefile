device=gpu

#configure machine
include ../../config/default.mk
include ../../config/*.mk
include ../../config/devices/devices.mk

INCLUDE+= -I../         # other src libraries
INCLUDE+= -I../../inc   # other project libraries

all: poet poet_hpc poet_mpi

poet: poet.cu poet.h init.h diag.h parameters.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(GLFLAGS) $(LIBS) $(JSONLIB) -DDG_BENCHMARK  -g
	
poet_hpc: poet.cu poet.h init.h diag.h parameters.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -DDG_BENCHMARK -DWITHOUT_GLFW

poet_mpi: poet.cu poet.h init.h diag.h parameters.h
	$(MPICC) $(OPT) $(MPICFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -DWITH_MPI -DDG_BENCHMARK -DWITHOUT_GLFW

doc:
	mkdir -p doc;
	pdflatex -output-directory doc ./poet.tex;
	bibtex doc/poet.aux;
	pdflatex -output-directory doc ./poet.tex;
	pdflatex -output-directory doc ./poet.tex;


.PHONY: clean doc

clean:
	rm -rf poet poet_hpc poet_mpi doc
