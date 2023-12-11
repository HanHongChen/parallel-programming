#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <mpi.h>

#pragma GCC optimize("O3", "Ofast", "fast-math")

void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr, int **a_mat_ptr, int **b_mat_ptr) {
	int world_rank, world_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	if (world_rank == 0) {
		scanf("%d%d%d", n_ptr, m_ptr, l_ptr);
	}
	MPI_Bcast(n_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(m_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(l_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
 
	int a_size = (*n_ptr) * (*m_ptr);
	int b_size = (*m_ptr) * (*l_ptr);

	*a_mat_ptr = (int*) calloc(a_size, sizeof(int));
	*b_mat_ptr = (int*) calloc(b_size, sizeof(int));

	int* a = *a_mat_ptr;
	int* b = *b_mat_ptr;

	if (world_rank == 0) {
		for (int i = 0; i < a_size; i++) {
			scanf("%d", &a[i]);
		}
		for (int i = 0; i < b_size; i++) {
			scanf("%d", &b[i]);
		}
	}

	// MPI_Bcast(n_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // MPI_Bcast(m_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // MPI_Bcast(l_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Bcast(*a_mat_ptr, a_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(*b_mat_ptr, b_size, MPI_INT, 0, MPI_COMM_WORLD);

	// MPI_Barrier(MPI_COMM_WORLD);
}

void matrix_multiply(const int n, const int m, const int l, const int *a_mat, const int *b_mat) {
	int world_rank, world_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	int ans_len = n * l;
	int* ans = (int*) calloc(ans_len, sizeof(int));
	int* tmp = (int*) calloc(ans_len, sizeof(int));

	int len = n / world_size;
	int upr = len * world_rank;
	int dow = upr + len;

	if (world_rank == world_size - 1) {
		dow = n;
		len = dow - upr;
	}

	int a_base = upr * m;
	for (int i = upr; i < dow; i++) {
		for (int j = 0; j < l; j++) {
			
			int sum = 0, b_idx = j;
			for (int k = 0; k < m; k++) {
				sum += a_mat[a_base + k] * b_mat[b_idx];
				b_idx += l;
			
				// printf("a_base + k = %d\n", sum);
			}
			// printf("a_base %d, j %d, %d = %d\n", a_base, j, i * l + j, sum);
			tmp[i * l + j] = sum;
		}
		a_base += m;
	}
	// MPI_Barrier(MPI_COMM_WORLD);

	MPI_Reduce(tmp, ans, ans_len, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	
	if (world_rank == 0) {
		int id = 0;
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < l; j++) {
				printf("%d ", ans[id]);
				id ++;
			}
			printf("\n");
		}
		// for (int i = 0; i < ans_len; i++) printf("%d ", ans[i]);
	}
}

void destruct_matrices(int *a_mat, int *b_mat) {
	int world_rank, world_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	// Free memory only once
	if (world_rank == 0) {
		free(a_mat);
		free(b_mat);
	}
}