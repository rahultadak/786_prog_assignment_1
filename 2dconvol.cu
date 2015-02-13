#include <stdio.h>
#include <stdlib.h>

#define DEBUG

__global__ void convol2D (float *a, float *h, float *c, int *a_rows, int *a_cols, int *h_rows, int *h_cols)
{
    int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    int grid_width = gridDim.x * blockDim.x;
    int index = index_y * grid_width + index_x;

    int block_index = blockDim.x * threadIdx.y + threadIdx.x;

    extern __shared__ float h_sh[];
    if(block_index < h_rows[0]* h_cols[0])
        h_sh[block_index] = h[block_index];

    __syncthreads();
    //Try shared memory loading for h

    c[index] = index;
    int sum = 0;
    for(int i = 0; i<h_rows[0]; i++) 
    {
        for(int j = 0; j<h_cols[0];j++)
        {
            if((index_y-i)>=0 & (index_y-i)<=(a_rows[0]-1) & (index_x-j)>=0 & (index_x-j)<=(a_cols[0]-1))
                sum = sum + (a[(index_y-i)*a_cols[0]+(index_x-j)] * h_sh[i*h_cols[0] + j]);
        }
    }
    c[index_y*grid_width + index_x] = sum;
}

        
int main (int argc, char *argv[])
{
    if(argc != 2)
    {
        fprintf(stderr,"Incorrect arguments passed.\nUse \"./2dconvol.o <input_file>\"\n");
        exit(1);
    }

    FILE *f;
    
    int a_cols = 1;
    int h_cols = 1;
    int a_rows = 0;
    int h_rows = 0;
    int c_cols = 0;
    int c_rows = 0;

    float *a_h = 0;
    float *a_d = 0;

    float *hinv_h = 0;
    float *hinv_d = 0;
    
    float *c_h = 0;
    float *c_d = 0;

    size_t a_size = 0;
    size_t h_size = 0;
    size_t c_size = 0;

    int *a_rows_d = 0;
    int *a_cols_d = 0;
    int *h_rows_d = 0;
    int *h_cols_d = 0;

    dim3 block_size;
    dim3 grid_size;

    int i=0,j=0;

    char junk,junk_old;
    
    //Opening File
    f = fopen(argv[1],"r");

    //First pass to find out size of the matrices
    junk = fgetc(f);

    while (junk != EOF)
    {
        if(junk == '\n')
        {
            a_rows++;
        }
        else if(junk == 0x20 & a_rows == 0)
        {
            a_cols++;
        }

        junk_old = junk;
        junk = fgetc(f);
        if(junk == '\n' & junk == junk_old)
        {
            break;
        }
    }

    junk = fgetc(f);
    while (junk != EOF)
    {
        if(junk == '\n')
        {
            h_rows++;
        }
        else if(junk == 0x20 & h_rows == 0)
        {
            h_cols++;
        }

        junk = fgetc(f);
    }

    //Calculating op dimensions
    c_rows = a_rows + h_rows - 1;
    block_size.y = c_rows > 512 ? 512 : c_rows;
    c_cols = a_cols + h_cols - 1;
    block_size.x = c_cols > 512 ? 512 : c_cols;

    grid_size.y = (c_rows/512)+1;
    grid_size.x = (c_cols/512)+1;

    #ifdef DEBUG
        printf("Size of A: %dx%d\n",a_rows,a_cols);
        printf("Size of H: %dx%d\n",h_rows,h_cols);
        printf("Size of C: %dx%d\n",c_rows,c_cols);
        printf("Size of grid: %dx%d\n",grid_size.y,grid_size.x);
        printf("Size of block: %dx%d\n",block_size.y,block_size.x);
    #endif

    //Calculating the sizes of all the involved matrices
    a_size = a_rows * a_cols *sizeof(float);
    h_size = h_rows * h_cols *sizeof(float);
    c_size = c_rows * c_cols *sizeof(float);

    //Allocating memory on host
    a_h = (float *) malloc(a_size);
    hinv_h = (float *) malloc(h_size);
    c_h = (float *) malloc(c_size);
    
    //Rewinding file to read the actual data
    rewind(f);

    //Reading all the data matrices
    for(i = 0;i<a_rows;i++)
    {
        for (j = 0; j<a_cols;j++)
            fscanf(f,"%f",&a_h[i*a_cols + j]);
    }

    for(i = 0 ; i<h_rows;i++)
    {
       for (j = 0; j<h_cols ;j++)
       {
            fscanf(f,"%f",&hinv_h[i*h_cols + j]);
       }
    }

#ifdef DEBUG
    for(i = 0;i<a_rows;i++)
    {
        for (j = 0; j<a_cols;j++)
            printf("%f ",a_h[i*a_cols + j]);
        printf("\n");
    }
    for(i = 0;i<h_rows;i++)
    {
        for (j = 0; j<h_cols;j++)
        {
            printf("%f ",hinv_h[i*h_cols + j]);
        }
        printf("\n");
    }
    printf("Completed Loading Matrices...\n");
#endif

    //cudaMalloc to allocate required matrices on the device
    cudaMalloc((void **)&a_d,a_size);
    cudaMalloc((void **)&hinv_d,h_size);
    cudaMalloc((void **)&c_d,c_size);
    cudaMalloc((void **)&a_rows_d,sizeof(int));
    cudaMalloc((void **)&a_cols_d,sizeof(int));
    cudaMalloc((void **)&h_rows_d,sizeof(int));
    cudaMalloc((void **)&h_cols_d,sizeof(int));

    //Copying input data from the Host to Device
    cudaMemcpy(a_d,a_h,a_size,cudaMemcpyHostToDevice);
    cudaMemcpy(hinv_d,hinv_h,h_size,cudaMemcpyHostToDevice);
    cudaMemcpy(a_rows_d,&a_rows,sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(a_cols_d,&a_cols,sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(h_rows_d,&h_rows,sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(h_cols_d,&h_cols,sizeof(int),cudaMemcpyHostToDevice);
    
    //Setting Op matrix to all zeros
    cudaMemset(c_d,0,c_size);

    //Convolution function
    convol2D<<<grid_size,block_size,h_size>>>(a_d,hinv_d,c_d,a_rows_d,a_cols_d,h_rows_d,h_cols_d);

    //Synchronize to wait for the kernel to complete exectution
    cudaThreadSynchronize();

    //Copy the output matrix from the Device to host
    cudaMemcpy(c_h,c_d,c_size,cudaMemcpyDeviceToHost);

    //Print Output
    for(i=0;i<c_rows;i++)
    {
        for(j=0;j<c_cols;j++)
        {
            printf("%f ",c_h[i*c_cols + j]);
        }
        printf("\n");
    }

    //Freeing all the allocated memory from the device
    cudaFree(a_d);
    cudaFree(hinv_d);
    cudaFree(c_d);

    //Freeing all the allocated memory from the host
    free(a_h);
    free(hinv_h);
    free(c_h);

    fclose(f);

    return 0;
}

