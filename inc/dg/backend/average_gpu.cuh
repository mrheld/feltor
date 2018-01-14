#pragma once

#include "exblas/exdot.fpe.cu"
#ifdef MPI_VERSION
#include "exblas/mpi_accumulate.h"
#endif //MPI_VERSION

namespace dg
{

template<class value_type>
__global__ void transpose_gpu_kernel( unsigned nx, unsigned ny, const value_type* __restrict__ in, value_type* __restrict__ out)
{
    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const int grid_size = gridDim.x*blockDim.x;
    const int size = nx*ny;
    for( int row = thread_id; row<size; row += grid_size)
    {
        int i=row/nx, j = row%nx;
        out[j*ny+i] = in[i*nx+j];
    }
}
template<class value_type>
void transpose_dispatch( CudaTag, unsigned nx, unsigned ny, const value_type* in, value_type* out){
    const size_t BLOCK_SIZE = 256; 
    const size_t NUM_BLOCKS = std::min<size_t>((nx*ny-1)/BLOCK_SIZE+1, 65000);
    transpose_gpu_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>( nx, ny, in, out);
}

template<class value_type>
__global__ void extend_line_kernel( unsigned nx, unsigned ny, const value_type* __restrict__ in, value_type* __restrict__ out)
{
    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const int grid_size = gridDim.x*blockDim.x;
    const int size = nx*ny;
    for( int row = thread_id; row<size; row += grid_size)
    {
        int i=row/nx, j = row%nx;
        out[i*nx+j] = in[j];
    }
}
template<class value_type>
void extend_line( CudaTag, unsigned nx, unsigned ny, const value_type* in, value_type* out){
    const size_t BLOCK_SIZE = 256; 
    const size_t NUM_BLOCKS = std::min<size_t>((nx*ny-1)/BLOCK_SIZE+1, 65000);
    extend_line_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>( nx, ny, in, out);
}

template<class value_type>
__global__ void extend_column_kernel( unsigned nx, unsigned ny, const value_type* __restrict__ in, value_type* __restrict__ out)
{
    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const int grid_size = gridDim.x*blockDim.x;
    const int size = nx*ny;
    for( int row = thread_id; row<size; row += grid_size)
    {
        int i=row/nx, j = row%nx;
        out[i*nx+j] = in[i];
    }
}
template<class value_type>
void extend_column( CudaTag, unsigned nx, unsigned ny, const value_type* in, value_type* out){
    const size_t BLOCK_SIZE = 256; 
    const size_t NUM_BLOCKS = std::min<size_t>((nx*ny-1)/BLOCK_SIZE+1, 65000);
    extend_column_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>( nx, ny, in, out);
}

void average( CudaTag, unsigned nx, unsigned ny, const double* in0, const double* in1, double* out)
{
    static thrust::device_vector<int64_t> d_accumulator;
    int64_t* d_ptr = thrust::raw_pointer_cast( d_accumulator.data());
    static thrust::host_vector<int64_t> h_accumulator;
    static thrust::host_vector<double> h_round;
    d_accumulator.resize( ny*exblas::BIN_COUNT);
    for( unsigned i=0; i<ny; i++)
        exblas::exdot_gpu(nx, &in0[i*nx], &in1[i*nx], &d_ptr[i*exblas::BIN_COUNT]);
    h_accumulator = d_accumulator;
    h_round.resize( ny);
    for( unsigned i=0; i<ny; i++)
        h_round[i] = exblas::cpu::Round( &h_accumulator[i*exblas::BIN_COUNT]);
    cudaMemcpy( out, &h_round[0], ny*sizeof(double), cudaMemcpyHostToDevice);
}

#ifdef MPI_VERSION
//local data plus communication
void average_mpi( CudaTag, unsigned nx, unsigned ny, const double* in0, const double* in1, double* out, MPI_Comm comm, MPI_Comm comm_mod, MPI_Comm comm_mod_reduce )
{
    static thrust::device_vector<int64_t> d_accumulator;
    int64_t* d_ptr = thrust::raw_pointer_cast( d_accumulator.data());
    static thrust::host_vector<int64_t> h_accumulator;
    static thrust::host_vector<int64_t> h_accumulator2;
    static thrust::host_vector<double> h_round;
    d_accumulator.resize( ny*exblas::BIN_COUNT);
    for( unsigned i=0; i<ny; i++)
        exblas::exdot_gpu(nx, &in0[i*nx], &in1[i*nx], &d_ptr[i*exblas::BIN_COUNT]);
    h_accumulator2 = d_accumulator;
    h_accumulator.resize( h_accumulator2.size());
    exblas::reduce_mpi_cpu( ny, &h_accumulator2[0], &h_accumulator[0], comm, comm_mod, comm_mod_reduce);

    h_round.resize( ny);
    for( unsigned i=0; i<ny; i++)
        h_round[i] = exblas::cpu::Round( &h_accumulator[i*exblas::BIN_COUNT]);
    cudaMemcpy( out, &h_round[0], ny*sizeof(double), cudaMemcpyHostToDevice);
}
#endif //MPI_VERSION

}//namespace dg
