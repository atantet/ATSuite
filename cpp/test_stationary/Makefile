OPT=/opt/local/
INC=-I$(PHD)/dev/include -I$(HOME)/local/include -I$(OPT)/include/eigen3
LIBS=-L$(HOME)/local/lib -ligraph
CC=g++ -fopenmp
CFLAGS=-Wall

EXE=code_com

all:$(EXE)

$(EXE):$(EXE).cpp
	$(CC) $(CFLAGS) $(INC) $(LIBS) $(EXE).cpp -o $(EXE)

clean:
	rm -f *.o $(EXE)
