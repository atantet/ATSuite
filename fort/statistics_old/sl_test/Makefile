#Compiler: Gnu or MPI?
USE_MPI=1

ifeq ($(USE_MPI), 1)
FC=openmpif90
FILE=sl_hadisst_mpi
else
FC=gfortran
FILE=sl_hadisst
endif

# Compiler flags
INC_PHD=$(HOME)/PhD/dev/include
LIB_PHD=$(HOME)/PhD/dev/lib
INC_LOCAL=$(HOME)/local/include
LIB_LOCAL=$(HOME)/local/lib
INC_OPT = /opt/local/include
CFLAGS=-c -Wall -I$(INC_PHD) -I$(INC_LOCAL) -I$(INC_OPT)
LDFLAGS= -lblas -lfftw3 -lnetcdff -latmath -lnonparam_mpi -L$(LIB_PHD) -L$(LIB_LOCAL)

# Installation folder
DSTPREFIX=$(HOME)/PhD/dev
DSTINCDIR=$(DSTPREFIX)/include/
DSTLIBDIR=$(DSTPREFIX)/lib/

all:$(FILE)

$(FILE).o: $(FILE).f90
	$(FC) $(CFLAGS) $(FILE).f90

$(FILE): $(FILE).o
	$(FC) $(LDFLAGS) $(FILE).o -o $(FILE)

clean:
	rm $(FILE).o
