#include <mpi.h>
#include <cstdio>

#pragma GCC optimize("O3", "Ofast", "fast-math")
// ./run < test_data 
// ./run < test_data > run_output
// ./run > run_output < test_data 
// Read size of matrix_a and matrix_b (n, m, l) and whole data of matrixes from stdin
//
// n_ptr:     pointer to n
// m_ptr:     pointer to m
// l_ptr:     pointer to l
// a_mat_ptr: pointer to matrix a (a should be a continuous memory space for placing n * m elements of int)
// b_mat_ptr: pointer to matrix b (b should be a continuous memory space for placing m * l elements of int)
void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr,
                        int **a_mat_ptr, int **b_mat_ptr){
    
    int world_rank, world_size;

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if(world_rank == 0){
        scanf("%d %d %d", n_ptr, m_ptr, l_ptr);
    }
    //將陣列資訊廣播出去
    MPI_Bcast(n_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(m_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(l_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);

    //確保都獲得資訊
    MPI_Barrier(MPI_COMM_WORLD);

    int a_size = (*n_ptr) * (*m_ptr);
    int b_size = (*m_ptr) * (*l_ptr);

    *a_mat_ptr = (int *)malloc(a_size * sizeof(int));
    *b_mat_ptr = (int *)malloc(b_size * sizeof(int));

    int* a = *a_mat_ptr;
    int* b = *b_mat_ptr;
    if(world_rank == 0){
        for(int i = 0; i < a_size; i++){
            scanf("%d", &a[i]);
        }

        for(int i = 0; i < b_size; i++){
            scanf("%d", &b[i]);
        }
    }
    MPI_Bcast(*a_mat_ptr, a_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(*b_mat_ptr, b_size, MPI_INT, 0, MPI_COMM_WORLD);

}

// Just matrix multiplication (your should output the result in this function)
// 
// n:     row number of matrix a
// m:     col number of matrix a / row number of matrix b
// l:     col number of matrix b
// a_mat: a continuous memory placing n * m elements of int
// b_mat: a continuous memory placing m * l elements of int
void matrix_multiply(const int n, const int m, const int l,
                     const int *a_mat, const int *b_mat){

    int world_rank, world_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int size = n / world_size;
    int start = world_rank * size;
    int end = (world_rank + 1) * size;
    if(world_rank == world_size - 1){
        end = n;
        size = end - start;
    }

    int *local_ptr = (int *)malloc(n * l * sizeof(int));
    int a_starter = start * m;
    for(int i = start; i < end; i++){

        for(int j = 0; j < l; j++){
            int sum = 0;
            int b_idx = j;
            for(int k = 0; k < m; k++){
                sum += a_mat[a_starter + k] * b_mat[b_idx];
                b_idx += l;
            }
            local_ptr[i * l + j] = sum;
        }
        a_starter += m;
    }
    int *c_mat = (int *)malloc(n * l * sizeof(int));
    MPI_Reduce(local_ptr, c_mat, n * l, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if(world_rank == 0){
        int index = 0;
        for(int i = 0; i < n; i++){
            for(int j = 0; j < l; j++){
                printf("%d ", c_mat[index++]);
            }
            printf("\n");
        }
    }

}

// Remember to release your allocated memory
void destruct_matrices(int *a_mat, int *b_mat){
    int world_rank, world_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	// 只能free 1次
	if (world_rank == 0) {
		free(a_mat);
		free(b_mat);
	}

}