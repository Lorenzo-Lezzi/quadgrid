# Compilatore NVIDIA per Thrust/CUDA
CXX = nvcc

# Flag per il cluster: standard C++20, ottimizzazione massima e architettura GPU L4
CXXFLAGS = -std=c++20 -O4 -arch=sm_89 -DNDEBUG
LDFLAGS = 

# Nome dell'eseguibile (su Linux senza .exe)
TARGET = simulazione

# Lista dei file sorgente. Se nel progetto hai anche file .cu, aggiungili qui
SRCS = branching_finale_pacs.cpp mmorton.cpp particles.cpp

OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -x cu -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean