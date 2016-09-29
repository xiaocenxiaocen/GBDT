CXX = icpc -O3 -Wall -g -std=c++0x -DDEBUG
CFLAGS = -openmp
LDFLAGS = -lm -lpthread

target = main \
GBDT

objects = TreeUtil.o \
BinaryTree.o \
DecisionTreeCART.o \
CSVParser.o 

all: $(target)

main: main.o $(objects)
	$(CXX) -o main -openmp main.o $(objects) $(LDFLAGS)

GBDT: GBDT.o $(objects)
	$(CXX) -o GBDT -openmp GBDT.o $(objects) $(LDFLAGS)

.SUFFIXES: .cpp .o

.cpp.o:
	$(CXX) $(CFLAGS) -c $< -o $@

.PHONY: clean

clean:
	-rm *.o
	-rm main
	-rm GBDT
