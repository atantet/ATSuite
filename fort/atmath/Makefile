FC=gfortran
CFLAGS=-c -Wall
INCDIR=$(HOME)/PhD/dev/include/
LIBDIR=$(HOME)/PhD/dev/lib/

all:libatmath.a

atmath.o: atmath.f90
	$(FC) $(CFLAGS) atmath.f90

libatmath.a: atmath.o
	ar -r libatmath.a atmath.o

install: libatmath.a atmath.mod
	cp atmath.mod $(INCDIR)
	cp libatmath.a $(LIBDIR)

clean:
	rm atmath.o atmath.mod libatmath.a
