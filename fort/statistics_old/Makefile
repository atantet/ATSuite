#Compiler: Gnu or MPI?
USE_MPI=0

ifeq ($(USE_MPI), 1)
FC=openmpif90
FILE=nonparam_mpi
else
FC=gfortran
FILE=nonparam
endif

# Compiler flags
INC_PHD=$(HOME)/PhD/dev/include
INC_LOCAL=$(HOME)/local/include
INC_OPT = /opt/local/include
CFLAGS=-c -Wall -I$(INC_PHD) -I$(INC_LOCAL) -I$(INC_OPT)

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
