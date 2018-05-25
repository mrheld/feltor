#pragma once

#include <cassert>
#include <cusp/array1d.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/dia_matrix.h>
#include <cusp/ell_matrix.h>
#include <cusp/hyb_matrix.h>

#include "vector_categories.h"
#include "matrix_categories.h"
#include "type_traits.h"

namespace dg
{

///@addtogroup vec_list
///@{
template<class T>
struct TypeTraits<cusp::array1d<T,cusp::host_memory>,
    typename std::enable_if< std::is_arithmetic<T>::value>::type>
{
    using value_type        = T;
    using tensor_category   = CuspVectorTag;
    using execution_policy  = SerialTag;
};
template<class T>
struct TypeTraits<cusp::array1d<T,cusp::device_memory>,
    typename std::enable_if< std::is_arithmetic<T>::value>::type>
{
    using value_type        = T;
    using tensor_category   = CuspVectorTag;
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
    using execution_policy  = CudaTag ; //!< if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
#else
    using execution_policy  = OmpTag ; //!< if THRUST_DEVICE_SYSTEM!=THRUST_DEVICE_SYSTEM_CUDA

#endif
};
///@}
///@addtogroup mat_list

template< class I, class V, class M>
struct TypeTraits< cusp::coo_matrix<I,V,M> >
{
    using value_type = V;
    using tensor_category = CuspMatrixTag;
};
template< class I, class V, class M>
struct TypeTraits< cusp::csr_matrix<I,V,M> >
{
    using value_type = V;
    using tensor_category = CuspMatrixTag;
};
template< class I, class V, class M>
struct TypeTraits< cusp::dia_matrix<I,V,M> >
{
    using value_type = V;
    using tensor_category = CuspMatrixTag;
};
template< class I, class V, class M>
struct TypeTraits< cusp::ell_matrix<I,V,M> >
{
    using value_type = V;
    using tensor_category = CuspMatrixTag;
};
template< class I, class V, class M>
struct TypeTraits< cusp::hyb_matrix<I,V,M> >
{
    using value_type = V;
    using tensor_category = CuspMatrixTag;
};

///@}

} //namespace dg
