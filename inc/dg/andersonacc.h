#pragma once

#include <functional>
#include "blas.h"

namespace dg{
///@cond
//const double m_EPS = 2.2204460492503131e-16;
namespace detail{

template<class ContainerType, class Mat>
void QRdelete1(std::vector<ContainerType>& Q,Mat& R, unsigned m)
{
    using value_type = dg::get_value_type<ContainerType>;
    for(unsigned i = 0; i<m-1;i++){
        value_type temp = sqrt(R[i][i+1]*R[i][i+1]+R[i+1][i+1]*R[i+1][i+1]);
        value_type c = R[i][i+1]/temp;
        value_type s = R[i+1][i+1]/temp;
        R[i][i+1] = temp;
        R[i+1][i+1] = 0;
        if (i < m-2) {
            for (unsigned j = i+2; j < m; j++){
                temp = c * R[i][j] + s * R[i+1][j];
                R[i+1][j] = - s * R[i][j] + c * R[i+1][j];
                R[i][j] = temp;
            }
        }
        //Collapse look into blas1 routines
        dg::blas1::axpby(-s,Q[i],c,Q[i+1]); // Q(i + 1) = s ∗ Q(ℓ, i) + c ∗ Q(ℓ, i + 1).
        dg::blas1::axpbypgz(c,Q[i],s,Q[i+1],0.,Q[i]); //Q(i) = c ∗ Q(ℓ, i) + s ∗ Q(ℓ, i + 1).
    } //Check for error in keeping the last row.!!!
    for(int i = 0; i<(int)m-1;i++)
        for(int j = 0; j < (int)m-1; j++)
            R[i][j] = R[i][j+1];
    return;
}
}//namespace detail
///@endcond



/*!@brief Anderson Acceleration of Fixed Point Iteration for \f[ f(x) = 0\f]
 *
 * This class implements the Anderson acceleration of the fixed point iteration algorithm for the problem
 * \f[
 *  f(x) = 0
 *  \f]
 *  described by https://users.wpi.edu/~walker/Papers/Walker-Ni,SINUM,V49,1715-1735.pdf
 * @copydoc hide_ContainerType
 */
template<class ContainerType>
struct AndersonAcceleration
{
    using container_type = ContainerType;
    using value_type = get_value_type<ContainerType>;
    AndersonAcceleration(){}
//mMax worth trying something between 3 and 10
//AAstart 0
//damping Fixed Point damping
//restart: restart the iteration
    AndersonAcceleration(const ContainerType& copyable, unsigned mMax ):
        m_gval( copyable), m_g_old( m_gval), m_fval( m_gval), m_f_old(m_gval),
        m_df( m_gval), m_DG( mMax, copyable), m_Q( m_DG),
        m_gamma( mMax, 0.), m_Ry( m_gamma),
        m_R( mMax, m_gamma)
    {}

    template<class BinarySubroutine>
    unsigned solve( BinarySubroutine& f, ContainerType& x, const ContainerType& b, const ContainerType& weights,
        value_type rtol, value_type atol, value_type beta, unsigned AAstart, unsigned max_iter,
        value_type damping, unsigned restart, bool verbose);

    private:
    ContainerType m_gval, m_g_old, m_fval, m_f_old, m_df;
    std::vector<ContainerType> m_DG, m_Q;
    std::vector<value_type> m_gamma, m_Ry;
    std::vector<std::vector<value_type>> m_R;

    value_type m_tol;
    unsigned m_max_iter, m_mMax;
};

/*
template<class ContainerType>
template<class BinarySubroutine>
unsigned AndersonAcceleration<ContainerType>::solve(
    BinarySubroutine& func, ContainerType& x, const ContainerType& b, const ContainerType& weights,
    value_type rtol, value_type atol, value_type beta, unsigned AAstart, unsigned max_iter,
    value_type damping, unsigned restart, bool verbose )
{
    if (m_mMax == 0){
        if(verbose)std::cout<< "No acceleration will occur" << std::endl;
    }

    unsigned mAA = 0;

    for(unsigned iter=0;iter < max_iter; iter++)
    {

        // Restart if a certain number of iterations are reached. Does not need to be mMax... Just seems to work nicely right now.
        if (iter % (restart) == 0) {
            if(verbose)std::cout << "Iter = " << iter << std::endl;
            mAA = 0;
        }

        func(x,m_fval);
        value_type res_norm = sqrt(dg::blas2::dot(m_fval,weights,m_fval));          //l2norm(m_fval)

        dg::blas1::axpby(1.,x,-damping,m_fval,m_gval);                      // m_gval = x - damping*m_fval

        if(iter == 0){
            m_tol = std::max(atol*sqrt(numu),rtol*res_norm);
            if(verbose)std::cout << "tol = " << m_tol << std::endl;
        }
        if(verbose)std::cout << "res_norm = " << res_norm << " Against tol = " << m_tol << std::endl;
        // Test for stopping
        if (res_norm <= m_tol){
            if(verbose)std::cout << "Terminate with residual norm = " << res_norm << std::endl;
            break;
        }

        if (m_mMax == 0){
            // Without acceleration, update x <- g(x) to obtain the next approximate solution.

            dg::blas1::copy(m_gval,x);                                    //x = m_gval;

        } else {
            // Apply Anderson acceleration.

            if(iter > AAstart){                                         // Update the m_df vector and the m_DG array.t,

                dg::blas1::axpby(1.,m_fval,-1.,m_f_fold,m_df);                 //m_df = m_fval-m_f_fold;

                if (mAA < m_mMax) {

                    dg::blas1::axpby(1.,m_gval,-1.,m_g_old,m_DG[mAA]);        //Update m_DG = [m_DG   m_gval-m_g_old];

                } else {

                    std::rotate(m_DG.begin(), m_DG.begin() + 1, m_DG.end());  //Rotate to the left hopefully this works... otherwise for i = 0 .. mMax-2 m_DG[i] = m_DG[i+1], and finally m_DG[mMax-1] = update...
                    dg::blas1::axpby(1.,m_gval,-1.,m_g_old,m_DG[m_mMax-1]);     //Update last m_DG entry

                }
                mAA = mAA + 1;
            }

            dg::blas1::copy(m_fval,m_f_fold);                                //m_f_fold = m_fval;

            dg::blas1::copy(m_gval,m_g_old);                                //m_g_old = m_gval;

            if(mAA==0.){

                dg::blas1::copy(m_gval,x);                                // If mAA == 0, update x <- g(x) to obtain the next approximate solution.

            } else {                                                    // If mAA > 0, solve the least-squares problem and update the solution.

                if (mAA == 1) {                                         // If mAA == 1, form the initial QR decomposition.

                    R[0][0] = sqrt(dg::blas2::dot(m_df,weights, m_df));
                    dg::blas1::axpby(1./R[0][0],m_df,0.,Q[0]);

                } else {                                                // If mAA > 1, update the QR decomposition.

                    if ((mAA > m_mMax)) {                                 // If the column dimension of Q is mMax, delete the first column and update the decomposition.


                        mAA = mAA - 1;
                        QRdelete1(Q,R,mAA);

                    }
                    // Now update the QR decomposition to incorporate the new column.
                    for (unsigned j = 1; j < mAA; j++) {
                        R[j-1][mAA-1] = dg::blas2::dot(Q[j-1],weights,m_df);      //Q(:,j)’*m_df; //Changed mAA -> mAA-1

                        dg::blas1::axpby(-R[j-1][mAA-1],Q[j-1],1.,m_df);  //m_df = m_df - R(j,mAA)*Q(:,j);
                    }
                    R[mAA-1][mAA-1] = sqrt(dg::blas2::dot(m_df,weights,m_df));
                    dg::blas1::axpby(1./R[mAA-1][mAA-1],m_df,0.,Q[mAA-1]);
                }

                //Calculate condition number of R to figure whether to keep going or call QR delete to reduce Q and R.
                //value_type condDF = cond(R,mAA);
                //Here should be the check for whether to proceed.

                //Solve least squares problem.
                for(int i = (int)mAA-1; i>=0; i--){
                    m_gamma[i] = dg::blas2::dot(Q[i],weights,m_fval);
                    for(int j = i + 1; j < mAA; j++){
                        m_gamma[i] -= R[i][j]*m_gamma[j];
                    }
                    m_gamma[i] /= R[i][i];
                }

                //Update new approximate solution x = m_gval - m_DG*gamma
                dg::blas1::copy(m_gval,x);
                for (unsigned i = 0; i < mAA; i++) {
                    dg::blas1::axpby(-m_gamma[i],m_DG[i],1.,x);
                }

                //In the paper there is a damping terms included.
                if ((abs(beta) > 0) && (beta != 1)){
                    for(unsigned i = 0; i < mAA; i++) {
                        m_Ry[i] = 0;
                        for(unsigned j = i; j < mAA; j++) {
                            m_Ry[i] += R[i][j]*m_gamma[j];                  //Check correctness
                        }
                    }
                    dg::blas1::axpby(-(1.-beta),m_fval,1,x);              //x = x -(1-beta)*m_fval
                    for(unsigned i = 0; i < mAA; i++) {
                        dg::blas1::axpby((1.0-beta)*m_Ry[i],Q[i],1.,x);   // x = x - (1-beta)*(-1*Q*R*gamma) = x + (1-beta)*Q*R*gamma = x + (1-beta)*Q*Ry
                    }
                }
            }//Should all mAA
        }

    }
    return max_iter;

}
*/
}//namespace dg
