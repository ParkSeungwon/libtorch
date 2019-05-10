CFLAG = -g -c -std=c++17 -fmax-errors=1 -pthread -D_GLIBCXX_USE_CXX11_ABI=0
INCLUDE = -I../ -I/home/zeta/Programming/libtorch/include/torch/csrc/api/include -I/home/zeta/Programming/libtorch/include/
DIR = $(notdir $(shell pwd))
CC = g++
OBJ = $(patsubst %.cc, ../obj/%.$(DIR).o, $(wildcard *.cc))
MAIN_OBJ = $(patsubst %.cpp, ../obj/%.$(DIR).x, $(wildcard *.cpp))

all : $(OBJ) $(MAIN_OBJ)

../obj/%.$(DIR).o : %.cc 
	$(CC) $(INCLUDE) $(CFLAG) $< -o $@

../obj/%.$(DIR).x : %.cpp
	$(CC) $(INCLUDE) $(CFLAG) $< -o $@
