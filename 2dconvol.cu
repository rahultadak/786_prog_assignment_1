#include <stdio.h>
#include <stdlib.h>

#define DEBUG

__global__ void conv(void)
{
}

int main (int argc, char *argv[])
{
    if(argc != 2)
    {
        fprintf(stderr,"Incorrect arguments passed.\nUse \"/conv.o <input_file>\"\n");
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
    c_cols = a_cols + h_cols - 1;

    #ifdef DEBUG
        printf("Size of A: %dx%d\n",a_rows,a_cols);
        printf("Size of H: %dx%d\n",h_rows,h_cols);
        printf("Size of C: %dx%d\n",c_rows,c_cols);
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

    for(i = h_rows-1 ; i>=0 ;i--)
    {
       for (j = h_cols-1 ; j>=0 ;j--)
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

    //Copying input data from the Host to Device
    cudaMemcpy(a_d,a_h,a_size,cudaMemcpyHostToDevice);
    cudaMemcpy(hinv_d,hinv_h,h_size,cudaMemcpyHostToDevice);

    //Setting Op matrix to all zeros
    cudaMemset(c_d,0,c_size);

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

