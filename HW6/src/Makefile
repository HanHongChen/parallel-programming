default: conv

# Cuda
NVCC = nvcc
CUDA_FLAGS = -arch=compute_61 -Xcompiler '-fPIC' 

# OpenCL
CC = gcc-10
FLAGS = -O3 -lOpenCL -m64 -ffloat-store -w -g -D CL_TARGET_OPENCL_VERSION=220

OBJS = main.o bmpfuncs.o hostFE.o serialConv.o helper.o

conv: $(OBJS)
	$(CC) -o $@ $(OBJS) $(FLAGS)

%.o: %.c
	$(CC) -c $(FLAGS) $< -o $@

cuda: link.o $(OBJS)
	$(NVCC) -o cuda kernel.o $(OBJS) -lcudart -lcuda -arch=compute_61 -Xcompiler '-fPIC' -O3 -lOpenCL -m64

link.o:
	$(NVCC) $(CUDA_FLAGS) -dc kernel.cu -o kernel.o

zip:
	rm -f HW6_0816153.zip
	zip HW6_0816153.zip kernel.cl hostFE.c url.txt

clean:
	rm -f cuda conv *.o output.bmp ref.bmp