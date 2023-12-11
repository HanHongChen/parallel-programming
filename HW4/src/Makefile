all:
	mpicxx -O3 -Ofast -ffast-math main.cc matrix_func.cc -o matmul
run:
	make; mpirun -np 4 --hostfile hosts.txt matmul < ./src/data2_9