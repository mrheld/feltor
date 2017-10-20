#pragma once
#include <cusp/transpose.h>
#ifdef MPI_VERSION
#include "mpi_matrix.h"
#endif //MPI_VERSION

namespace dg
{

///@cond
namespace detail
{
template <class Matrix>
Matrix doTranspose( const Matrix& src, CuspMatrixTag)
{
    Matrix out;
    cusp::transpose( src, out);
    return out;
}
#ifdef MPI_VERSION
template <class LocalMatrix, class Collective>
MPIDistMat<LocalMatrix, Collective> doTranspose( const MPIDistMat<LocalMatrix, Collective>& src, MPIMatrixTag)
{
    LocalMatrix tr = doTranspose( src.matrix(), typename MatrixTraits<LocalMatrix>::matrix_category());
    MPIDistMat<LocalMatrix, Collective> out( tr, src.collective());
    return out;
}
#endif// MPI_VERSION
}//namespace detail
///@endcond

/**
 * @brief Generic matrix transpose method
 *
 * @tparam Matrix one of 
 *  - any cusp matrix
 *  - any MPIDistMatrix with a cusp matrix as template parameter
 * @param src the marix to transpose
 *
 * @return the matrix that acts as the transpose of src
 * @ingroup lowlevel 
 */
template<class Matrix>
Matrix transpose( const Matrix& src)
{
    //%Transposed matrices work only for csr_matrix due to bad matrix form for ell_matrix!!!
    return detail::doTranspose( src, typename MatrixTraits<Matrix>::matrix_category());
}

} //namespace dg
