CC = /usr/local/cuda-8.0//bin/nvcc
GENCODE_FLAGS = -arch=sm_30
LD_FLAGS = -lsndfile
CC_FLAGS = -G -g --compiler-options -Wall,-Wextra,-O3,-m64
NVCCFLAGS = -m64 

blur: blur.cu
	$(CC) $(CC_FLAGS) $(NVCCFLAGS) $(GENCODE_FLAGS) $(LD_FLAGS) blur.cu -o blur

clean:
	rm blur
