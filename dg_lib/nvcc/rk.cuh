#ifndef _DG_RK_
#define _DG_RK_

#include <cassert>
#include <vector>

#include "blas1.h"

namespace dg{

//namespace detail <--------- ??

/*! @brief coefficients for explicit RK methods
 *
 * To derive these coefficients from the butcher tableau
 * consider 
 * \f[ y = Bhk = (B-D)hk + Dhk = (B-D)B^{-1}y + Dhk 
 *       = ( 1- DB^{-1})y + Dhk = \alpha y + \beta h k\f]
 *  where \f[ B\f] is the butcher tableau of order k+1 and \f[ D\f] its
 *  diagonal part. 
 * @tparam k Order of the method. Currently 2,3 and 4 are available
 */
template< size_t k>
struct rk_coeff
{
    static const double alpha[k][k]; 
    static const double beta[k];
};
///@cond
/*
template<>
const double rk_coeff<1>::alpha[1][1] = { {1}};
template<>
const double rk_coeff<1>::beta[1] = {1};
*/

//from Cockburn paper
template<>
const double rk_coeff<2>::alpha[2][2] = {
    { 1,   0},
    { 0.5, 0.5}
};
template<>
const double rk_coeff<2>::beta[2] = {
     1, 0.5
};
//from Cockburn paper
template<>
const double rk_coeff<3>::alpha[3][3] = {
    { 1,     0,    0},
    { 0.75,  0.25, 0},
    { 1./3., 0.,   2./3.}
};
template<>
const double rk_coeff<3>::beta[3] = {
     1, 0.25, 2./3.
};
//classic RK4 coefficients (matlab used to compute from normal form)
template<>
const double rk_coeff<4>::alpha[4][4] = {
    { 1,    0., 0, 0 },
    { 1.,   0., 0, 0 },
    { 1.,   0., 0, 0 },
    {-1./3., 1./3., 2./3., 1./3.}
};
template<>
const double rk_coeff<4>::beta[4] = {
     0.5, 0.5, 1.0, 1./6.
};
///@endcond
//RHS contains Information about Vector type it uses
//k is the order of the method
// Vector f( const Vector& v)
// Vector should probably be rvalue assignable

/**
* @brief Struct for RungeKutta integration
*
* @ingroup algorithms
* Uses only blas1::axpby routines to integrate one step.
* The coefficients are chosen in a form that require a minimum of 
* axpby function calls (check for alpha==0, beta==1) and else 
* assumes that most of the work is done in the computation of the rhs.
* @tparam k Order of the method
* @tparam Functor models BinaryFunction with no return type (subroutine)
        The first argument is the actual argument, The second contains
        the return value, i.e. y' = f(y) translates to f( y, y'). Moreover the 
        class must typedef the argument type to Vector. 
*/
template< size_t k, class Functor>
struct RK
{
    /**
    * @brief The vector type
    * 
    * Must model Assignable and blas1::axpby( alpha, x, beta, y) 
    * where x and y are of type Vector must be a valid expression.
    */
    typedef typename Functor::Vector Vector; 
    /**
    * @brief Reserve memory for the integration
    *
    * @param copyable Vector of size which is used in integration. 
    * A Vector object must be copy-constructible from copyable.
    */
    RK( const Vector& copyable): u_(k-1, Vector(copyable)){ }
    /**
    * @brief Advance u0 one timestep
    *
    * @param f right hand side function
    * @param u0 initial value
    * @param u1 contains result on output. u0 and u1 may currently not be the same.
    * @param dt The timestep.
    */
    void operator()( Functor& f, const Vector& u0, Vector& u1, double dt);
  private:
    std::vector<Vector> u_; //TODO std::array might more natural here
};

//u0 and u1 may not be the same vector
//TO DO: this might be cured by adding u0 first to u_[0] in the last step
//f( y, yp) where y is const and yp contains the result
template< size_t k, class Functor>
void RK<k, Functor>::operator()( Functor& f, const Vector& u0, Vector& u1, double dt)
{
    assert( &u0 != &u1);
    assert( k>1 && "Euler still has to be implemented!" );
    f(u0, u_[0]);
    blas1::axpby( rk_coeff<k>::alpha[0][0], u0, dt*rk_coeff<k>::beta[0], u_[0]);
    cudaThreadSynchronize();
    for( unsigned i=1; i<k-1; i++)
    {
        f( u_[i-1], u_[i]);
        blas1::axpby( rk_coeff<k>::alpha[i][0], u0, dt*rk_coeff<k>::beta[i], u_[i]);
        cudaThreadSynchronize();
        for( unsigned l=1; l<=i; l++)
        {
            blas1::axpby( rk_coeff<k>::alpha[i][l], u_[l-1],1., u_[i]); //Fall alpha = 0 muss axpby abfangen!!
            cudaThreadSynchronize();
        }

    }
    //Now add everything up to u1
    f( u_[k-2], u1);
    blas1::axpby( rk_coeff<k>::alpha[k-1][0], u0, dt*rk_coeff<k>::beta[k-1], u1);
    cudaThreadSynchronize();
    for( unsigned l=1; l<=k-1; l++)
    {
        blas1::axpby( rk_coeff<k>::alpha[k-1][l], u_[l-1],1., u1);
        cudaThreadSynchronize();
    }
}


} //namespace dg

#endif //_DG_RK_