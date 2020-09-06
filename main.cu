#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include <climits>
#include <curand.h>
#include <curand_kernel.h>
using namespace std;

#define N_batches 100 //Matrix size = N_batches*batch_size
#define batch_size 1024

__global__ fill_matr(int* d_matr, int matrix_size)
{
	int idx=blockIdx.x*blockDim.x + threadIdx.x;
	int idy=blockIdx.y*blockDim.y + threadIdx.y;
	for(int i=0;i<N_batches;i++)//find gcd of elements in the batch, then move on to next
	{
		//d_matr[idx][idy]=gcd(idx,idy)
		//idx+=batch_size
		//idy+=batch_size
		
	}
}

int main()
{
    clock_t start,end;
	start=clock();

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	printf("kernel timeout enabled: %d\n",prop.kernelExecTimeoutEnabled);

	const int matrix_size=N_batches*batch_size;
	//files
	FILE *out_matr;
	out_matr=fopen("out_matr.txt","w");
	//matr 1d array
	int* d_matr;
	cudaMalloc((void**)&d_matr, matrix_size*sizeof(int));
	int* h_matr;
	h_matr=(int*)malloc(matrix_size*sizeof(int));
	
	//kernel launch config
	dim3 grid_conf(matrix_size,1,1);
	dim3 block_conf(1,batch_size,1);

	fill_matr<<<grid_conf,block_conf>>>(d_matr, matrix_size);

	cudaMemcpy(h_matr, d_matr, matrix_size*sizeof(int), cudaMemcpyDeviceToHost);

	print_matr(h_matr,matrix_size);

	cudaFree(d_matr);
	free(h_matr);
	fclose(out_matr);

	printf("===launch status report===\n");
	//check for errors
	cudaError_t err=cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("CUDA ERROR!!!\n");
		printf("err code: %d\n",err);
		if (err == 702)
		{
			printf("702 is similar to WDDM TDR false trigger; suggest running from tty3\n");
		}
		if (err == 700)
		{
			printf("700 is out of range call\n");
		}
	}
	else
	{
		printf("No CUDA errors!!!\n");
	}

	end=clock();
	double total_time=(double)(end-start)/CLOCKS_PER_SEC;//in seconds
	printf("TOTAL TIME: %.1lf seconds (%.1lf minutes)\n",total_time,total_time/60);
}
