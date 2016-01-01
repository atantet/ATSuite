#Compiler: Gnu or MPI?
USE_MPI=0

# Compiler flags
INC_PHD=$(HOME)/PhD/dev/include
INC_LOCAL=$(HOME)/local/include
INC_OPT = /opt/local/include

ifeq ($(USE_MPI), 1)
FC=openmpif90
FILE=nonparam_mpi
CFLAGS=-c -Wall -I$(INC_PHD) -I$(INC_LOCAL) -I$(INC_OPT)
else
FC=gfortran
FILE=nonparam
CFLAGS=-c -fopenmp -Wall -I$(INC_PHD) -I$(INC_LOCAL) -I$(INC_OPT)
endif


# Installation folder
DSTPREFIX=$(HOME)/PhD/dev
DSTINCDIR=$(DSTPREFIX)/include/
DSTLIBDIR=$(DSTPREFIX)/lib/

all:lib$(FILE).a

$(FILE).o $(FILE).mod: $(FILE).f90
	$(FC) $(CFLAGS) $(FILE).f90

lib$(FILE).a: $(FILE).o
	ar -r lib$(FILE).a $(FILE).o

install: lib$(FILE).a $(FILE).mod
	mv $(FILE).mod $(DSTINCDIR)
	mv lib$(FILE).a $(DSTLIBDIR)

clean:
	rm $(FILE).o $(FILE).mod lib$(FILE).a
