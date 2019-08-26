#pragma once

#include "implicit.h"


/*! @file
  @brief contains multistep explicit& implicit time-integrators
  */
namespace dg{


/*! @class hide_explicit_implicit
 * @tparam Explicit The explicit part of the right hand side
        is a functor type with no return value (subroutine)
        of signature <tt> void operator()(value_type, const ContainerType&, ContainerType&)</tt>
        The first argument is the time, the second is the input vector, which the functor may \b not override, and the third is the output,
        i.e. y' = f(t, y) translates to f(t, y, y').
        The two ContainerType arguments never alias each other in calls to the functor.
 * @tparam Implicit The implicit part of the right hand side
        is a functor type with no return value (subroutine)
        of signature <tt> void operator()(value_type, const ContainerType&, ContainerType&)</tt>
        The first argument is the time, the second is the input vector, which the functor may \b not override, and the third is the output,
        i.e. y' = f(t, y) translates to f(t, y, y').
        The two ContainerType arguments never alias each other in calls to the functor.
    Furthermore, if the \c DefaultSolver is used, the routines %weights(), %inv_weights() and %precond() must be callable
    and return diagonal weights, inverse weights and the preconditioner for the conjugate gradient method.
    The return type of these member functions must be useable in blas2 functions together with the ContainerType type.
 * @param ex explic part
 * @param im implicit part ( must be linear in its second argument and symmetric up to weights)
 */
/*!@class hide_note_multistep
* @note Uses only \c blas1::axpby routines to integrate one step.
* @note The difference between a multistep and a single step method like RungeKutta
* is that the multistep only takes one right-hand-side evaluation per step.
* This might be advantageous if the right hand side is expensive to evaluate like in
* partial differential equations. However, it might also happen that the stability
* region of the one-step method is larger so that a larger timestep can be taken there
* and on average they take just the same rhs evaluations.
* @note a disadvantage of multistep is that timestep adaption is not easily done.
*/

/**
* @brief Struct for Adams-Bashforth explicit multistep time-integration
* \f[ u^{n+1} = u^n + \Delta t\sum_{j=0}^{s-1} b_j f\left(t^n - j \Delta t, u^{n-j}\right) \f]
*
* with coefficients taken from https://en.wikipedia.org/wiki/Linear_multistep_method
* @copydoc hide_note_multistep
* @copydoc hide_ContainerType
* @ingroup time
*/
template<class ContainerType>
struct AdamsBashforth
{
    using value_type = get_value_type<ContainerType>;//!< the value type of the time variable (float or double)
    using container_type = ContainerType; //!< the type of the vector class in use
    ///copydoc RungeKutta::RungeKutta()
    AdamsBashforth(){}
    ///@copydoc AdamsBashforth::construct()
    AdamsBashforth( unsigned order, const ContainerType& copyable){
        construct( order, copyable);
    }
    /**
    * @brief Reserve internal workspace for the integration
    *
    * @param order (global) order (= number of steps in the multistep) of the method (Currently, one of 1, 2, 3, 4 or 5)
    * @param copyable ContainerType of the size that is used in \c step
    * @note it does not matter what values \c copyable contains, but its size is important
    */
    void construct(unsigned order, const ContainerType& copyable){
        m_k = order;
        m_f.assign( order, copyable);
        m_u = copyable;
        m_ab.resize( order);
        switch (order){
            case 1: m_ab = {1}; break;
            case 2: m_ab = {1.5, -0.5}; break;
            case 3: m_ab = { 23./12., -4./3., 5./12.}; break;
            case 4: m_ab = {55./24., -59./24., 37./24., -3./8.}; break;
            case 5: m_ab = { 1901./720., -1387./360., 109./30., -637./360., 251./720.}; break;
            default: throw dg::Error(dg::Message()<<"Order not implemented in AdamsBashforth!");
        }
    }
    ///@brief Return an object of same size as the object used for construction
    ///@return A copyable object; what it contains is undefined, its size is important
    const ContainerType& copyable()const{ return m_u;}

    /**
     * @brief Initialize first step. Call before using the step function.
     *
     * This routine initiates the first steps in the multistep method by integrating
     * backwards in time with Euler's method. This routine has to be called
     * before the first timestep is made.
     * @copydoc hide_rhs
     * @param rhs The rhs functor
     * @param t0 The intital time corresponding to u0
     * @param u0 The initial value of the integration
     * @param dt The timestep
     * @note the implementation is such that on output the last call to the rhs is at (t0,u0). This might be interesting if the call to the rhs changes its state.
     */
    template< class RHS>
    void init( RHS& rhs, value_type t0, const ContainerType& u0, value_type dt);
    /**
    * @brief Advance u0 one timestep
    *
    * @copydoc hide_rhs
    * @param f right hand side function or functor
    * @param t (write-only) contains timestep corresponding to \c u on output
    * @param u (write-only) contains next step of the integration on output
    * @note the implementation is such that on output the last call to the rhs is at the new (t,u). This might be interesting if the call to the rhs changes its state.
    */
    template< class RHS>
    void step( RHS& f, value_type& t, ContainerType& u);
  private:
    value_type m_tu, m_dt;
    std::vector<ContainerType> m_f;
    ContainerType m_u;
    std::vector<value_type> m_ab;
    unsigned m_k;
};

template< class ContainerType>
template< class RHS>
void AdamsBashforth<ContainerType>::init( RHS& f, value_type t0, const ContainerType& u0, value_type dt)
{
    m_tu = t0, m_dt = dt;
    f( t0, u0, m_f[0]);
    //now do k Euler steps
    ContainerType u1(u0);
    for( unsigned i=1; i<m_k; i++)
    {
        blas1::axpby( 1., u1, -dt, m_f[i-1], u1);
        m_tu -= dt;
        f( m_tu, u1, m_f[i]);
    }
    m_tu = t0;
    blas1::copy(  u0, m_u);
    //finally evaluate f at u0 once more to set state in f
    f( m_tu, m_u, m_f[0]);
}

template<class ContainerType>
template< class RHS>
void AdamsBashforth<ContainerType>::step( RHS& f, value_type& t, ContainerType& u)
{
    for( unsigned i=0; i<m_k; i++)
        blas1::axpby( m_dt*m_ab[i], m_f[i], 1., m_u);
    //permute m_f[k-1]  to be the new m_f[0]
    for( unsigned i=m_k-1; i>0; i--)
        m_f[i-1].swap( m_f[i]);
    blas1::copy( m_u, u);
    t = m_tu = m_tu + m_dt;
    f( m_tu, m_u, m_f[0]); //evaluate f at new point
}

/**
* @brief Struct for Karniadakis semi-implicit multistep time-integration
* \f[
* \begin{align}
    v^{n+1} = \sum_{q=0}^2 \alpha_q v^{n-q} + \Delta t\left[\left(\sum_{q=0}^2\beta_q  \hat E(t^{n}-q\Delta t, v^{n-q})\right) + \gamma_0\hat I(t^{n}+\Delta t, v^{n+1})\right]
    \end{align}
    \f]

    which discretizes
    \f[
    \frac{\partial v}{\partial t} = \hat E(t,v) + \hat I(t,v)
    \f]
    where \f$ \hat E \f$ contains the explicit and \f$ \hat I \f$ the implicit part of the equations.
    The coefficients are
    \f[
    \alpha_0 = \frac{18}{11}\ \alpha_1 = -\frac{9}{11}\ \alpha_2 = \frac{2}{11} \\
    \beta_0 = \frac{18}{11}\ \beta_1 = -\frac{18}{11}\ \beta_2 = \frac{6}{11} \\
    \gamma_0 = \frac{6}{11}
\f]
*
* The necessary Inversion in the imlicit part is provided by the \c SolverType class.
* Per Default, a conjugate gradient method is used (therefore \f$ \hat I(t,v)\f$ must be linear in \f$ v\f$).
* @note The implicit part equals a third order backward differentiation formula (BDF) https://en.wikipedia.org/wiki/Backward_differentiation_formula
*
The following code example demonstrates how to implement the method of manufactured solutions on a 2d partial differential equation with the dg library:
* @snippet multistep_t.cu function
* In the main function:
* @snippet multistep_t.cu karniadakis
* @note In our experience the implicit treatment of diffusive or hyperdiffusive
terms can significantly reduce the required number of time steps. This
outweighs the increased computational cost of the additional matrix inversions.
* @copydoc hide_note_multistep
* @copydoc hide_SolverType
* @copydoc hide_ContainerType
* @ingroup time
*/
template<class ContainerType, class SolverType = dg::DefaultSolver<ContainerType>>
struct Karniadakis
{
    using value_type = get_value_type<ContainerType>;//!< the value type of the time variable (float or double)
    using container_type = ContainerType; //!< the type of the vector class in use
    ///@copydoc RungeKutta::RungeKutta()
    Karniadakis(){}

    ///@copydoc construct()
    template<class ...SolverParams>
    Karniadakis( SolverParams&& ...ps):m_solver( std::forward<SolverParams>(ps)...){
        m_f.fill(m_solver.copyable()), m_u.fill(m_solver.copyable());
        init_coeffs();
    }
    /**
     * @brief Reserve memory for the integration
     *
     * @param ps Parameters that are forwarded to the constructor of \c SolverType
     * @tparam SolverParams Type of parameters (deduced by the compiler)
    */
    template<class ...SolverParams>
    void construct( SolverParams&& ...ps){
        m_solver = Solver( std::forward<SolverParams>(ps)...);
        m_f.fill(m_solver.copyable()), m_u.fill(m_solver.copyable());
        init_coeffs();
    }
    ///@brief Return an object of same size as the object used for construction
    ///@return A copyable object; what it contains is undefined, its size is important
    const ContainerType& copyable()const{ return m_u[0];}

    ///Write access to the internal solver for the implicit part
    SolverType& solver() { return m_solver;}
    ///Read access to the internal solver for the implicit part
    const SolverType& solver() const { return m_solver;}

    /**
     * @brief Initialize by integrating two timesteps backward in time
     *
     * The backward integration uses the Lie operator splitting method, with explicit Euler substeps for both explicit and implicit part
     * @copydoc hide_explicit_implicit
     * @param t0 The intital time corresponding to u0
     * @param u0 The initial value of the integration
     * @param dt The timestep saved for later use
     * @note the implementation is such that on output the last call to the explicit part \c ex is at \c (t0,u0). This might be interesting if the call to \c ex changes its state.
     */
    template< class Explicit, class Implicit>
    void init( Explicit& ex, Implicit& im, value_type t0, const ContainerType& u0, value_type dt);

    /**
    * @brief Advance one timestep
    *
    * @copydoc hide_explicit_implicit
    * @param t (write-only), contains timestep corresponding to \c u on output
    * @param u (write-only), contains next step of time-integration on output
     * @note the implementation is such that on output the last call to the explicit part \c ex is at the new \c (t,u). This might be interesting if the call to \c ex changes its state.
    */
    template< class Explicit, class Implicit>
    void step( Explicit& ex, Implicit& im, value_type& t, ContainerType& u);

  private:
    void init_coeffs(){
        //a[0] =  1.908535476882378;  b[0] =  1.502575553858997;
        //a[1] = -1.334951446162515;  b[1] = -1.654746338401493;
        //a[2] =  0.426415969280137;  b[2] =  0.670051276940255;
        a[0] =  18./11.;    b[0] =  18./11.;
        a[1] = -9./11.;     b[1] = -18./11.;
        a[2] = 2./11.;      b[2] = 6./11.;   //Karniadakis !!!
    }
    SolverType m_solver;
    std::array<ContainerType,3> m_u, m_f;
    value_type t_, m_dt;
    value_type a[3];
    value_type b[3], g0 = 6./11.;
};

///@cond
template< class ContainerType, class SolverType>
template< class RHS, class Diffusion>
void Karniadakis<ContainerType, SolverType>::init( RHS& f, Diffusion& diff, value_type t0, const ContainerType& u0, value_type dt)
{
    //operator splitting using explicit Euler for both explicit and implicit part
    t_ = t0, m_dt = dt;
    blas1::copy(  u0, m_u[0]);
    f( t0, u0, m_f[0]); //f may not destroy u0
    blas1::axpby( 1., m_u[0], -dt, m_f[0], m_f[1]); //Euler step
    detail::Implicit<Diffusion, ContainerType> implicit( -dt, t0, diff);
    implicit( m_f[1], m_u[1]); //explicit Euler step backwards
    f( t0-dt, m_u[1], m_f[1]);
    blas1::axpby( 1.,m_u[1], -dt, m_f[1], m_f[2]);
    implicit.time() = t0 - dt;
    implicit( m_f[2], m_u[2]);
    f( t0-2*dt, m_u[2], m_f[2]); //evaluate f at the latest step
    f( t0, u0, m_f[0]); // and set state in f to (t0,u0)
}

template<class ContainerType, class SolverType>
template< class RHS, class Diffusion>
void Karniadakis<ContainerType, SolverType>::step( RHS& f, Diffusion& diff, value_type& t, ContainerType& u)
{
    blas1::axpbypgz( m_dt*b[0], m_f[0], m_dt*b[1], m_f[1], m_dt*b[2], m_f[2]);
    blas1::axpbypgz( a[0], m_u[0], a[1], m_u[1], a[2], m_u[2]);
    //permute m_f[2], m_u[2]  to be the new m_f[0], m_u[0]
    for( unsigned i=2; i>0; i--)
    {
        m_f[i-1].swap( m_f[i]);
        m_u[i-1].swap( m_u[i]);
    }
    blas1::axpby( 1., m_f[0], 1., m_u[0]);
    //compute implicit part
    value_type alpha[2] = {2., -1.};
    //value_type alpha[2] = {1., 0.};
    blas1::axpby( alpha[0], m_u[1], alpha[1],  m_u[2], u); //extrapolate previous solutions
    t = t_ = t_+ m_dt;
    m_solver.solve( -m_dt*g0, diff, t, u, m_u[0]);
    blas1::copy( u, m_u[0]); //store result
    f(t_, m_u[0], m_f[0]); //call f on new point
}
///@endcond
/*
template<class ContainerType,class RHS> // , class RHS
struct BDF{
    using value_type = dg::get_value_type<ContainerType>;
    using container_type = ContainerType;

    BDF(){}

    //template<class RHS>
    BDF( unsigned order, const ContainerType& copyable, RHS& rhs, value_type dt):rhs_(rhs){ //
        construct( order, copyable, rhs, dt); //rhs, 
    }

    //template<class RHS>
    void construct(unsigned order, const ContainerType& copyable, RHS& rhs, value_type dt){ // 
        //RHS& rhs_ = rhs;
        //rhs_ = rhs;
        m_k = order;
        u_.assign( order, copyable);
        f_ = copyable;
        sum_prev_x = copyable;
        dt_ = dt;
        //Construct Newton
        
        switch (order){
            case 1: 
            m_bdf = {1.}; 
            f_factor = 1.;
            break;
            case 2: 
            m_bdf = {4./3., -1./3.}; 
            f_factor = 2./3.;
            break;
            case 3: 
            m_bdf = { 18./11., -9./11., 2./11.}; 
            f_factor = 6./11.;
            break;
            case 4: 
            m_bdf = {48./25., -36./25., 16./25., -3./25.}; 
            f_factor = 12./25.;
            break;
            case 5: 
            m_bdf = { 300./137., -300./137., 200./137., -75./137., 12./137.}; 
            f_factor = 60/137.;
            break;
            case 6: 
            m_bdf = { 360./147., -450./147., 400./147., -225./147., 172./147., -10./147.}; 
            f_factor = 60/147.;
            break;
            default: throw dg::Error(dg::Message()<<"Order not implemented in BDF!");
        }
        std::cout << "Hello, i'm the BDF constructor :) " << std::endl;
    }

    void func(const ContainerType& uin, ContainerType& uout);           //Function to find root of.

    void init(value_type t0, const ContainerType& u0);                  //Initialise

    void set_mMax(int input_mMax){
        mMax = input_mMax;
    }

    void set_itmax(int input_itmax){
        itmax = input_itmax;
    }

    void set_AAstart(int input_AAstart){
        AAstart = input_AAstart;
    }    

    void step(value_type t0, value_type& t1, container_type& u, value_type rtol, value_type atol, int solver, value_type damping, value_type beta);
    private:
        int mMax = 10;
        int itmax = 100;
        int AAstart = 0;
        RHS& rhs_;
        value_type tu_, dt_, f_factor;
        std::vector<ContainerType> u_;
        ContainerType f_;
        ContainerType sum_prev_x;
        std::vector<value_type> m_bdf;
        unsigned m_k;
};

template< class ContainerType, class RHS>
void BDF<ContainerType, RHS>::func(const ContainerType& x, ContainerType& fval){ //Function to pass to JFNK
    rhs_(tu_,x,fval);                                                   //fval = rhs(x,t) //tu_ needs to be set
    dg::blas1::scal(fval,dt_);                                          //fval = fx*dt
    dg::blas1::axpby(1.,sum_prev_x,f_factor,fval);                      //fval = sum_x_prev + f_factor*fval where sum_x_prev = sum_order x_order
    dg::blas1::axpby(1.,x,-1.,fval);                                    //fx = x-fx... 
}

template< class ContainerType, class RHS>
void BDF<ContainerType, RHS>::init(value_type t0, const ContainerType& u0){
    //Perform a number of backward euler steps
    std::cout << "Initialising method" << std::endl;
    ContainerType fout(u0);
    dg::blas1::copy(u0, u_[0]);
    std::cout << "l2norm of u0 = " << std::setprecision(15) << dg::l2norm(u0) << std::endl;
    for (unsigned i = 0; i<m_bdf.size()-1; i++){
        rhs_(t0-i*dt_,u_[i],fout);
        dg::blas1::axpby(-dt_,fout,1.,u_[i],u_[i+1]);
        std::cout << "l2norm of u0 = " << std::setprecision(15) << dg::l2norm(u_[i+1]) << std::endl;
    }
    
}

template< class ContainerType, class RHS>
void BDF<ContainerType, RHS>::step(value_type t0, value_type& t1, container_type& u, value_type rtol, value_type atol, int solver, value_type damping, value_type beta){
    dg::blas1::axpby(m_bdf[0],u_[0],0.,sum_prev_x);
    for (unsigned i = 1; i < m_bdf.size(); i++){
        dg::blas1::axpby(m_bdf[i],u_[i],1.,sum_prev_x);
    }
    tu_ = t1 = t0+dt_;
    ContainerType upred(u);
    ContainerType uout(u);
    ContainerType diff(u);
    value_type alpha[2] = {2., -1.};
    dg::blas1::axpby( alpha[0], u_[0], alpha[1],  u_[1], u);
    std::cout << "AAstart = " << AAstart << std::endl;
    std::cout << "mMax = " << mMax << std::endl;
    std::cout << "itmax = " << itmax << std::endl;
    using namespace std::placeholders;
    auto fun = std::bind(&BDF<ContainerType, RHS>::func,*this,_1,_2);
    if (solver == 0) {
        std::cout << "You have chosen the Anderson Acceleration" << std::endl;
        andacc<ContainerType>(fun, u, mMax, itmax, atol, rtol, beta, AAstart, damping);
    } else if (solver == 1){
        std::cout << "You have chosen the Nonlinear GMRES" << std::endl;
        ngmres<ContainerType>(fun, u, mMax, itmax, atol, rtol, damping);
    } else if (solver == 2){
        std::cout << "You have chosen the de Sterck Nonlinear GMRES" << std::endl;
        dsngmres<ContainerType>(fun, u, mMax, itmax, atol, rtol, damping);
    } else if (solver == 3){
        std::cout << "You have chosen the objective acceleration" << std::endl;
        o_accel<ContainerType>(fun, u, mMax, itmax, atol, rtol, damping);
    }else if (solver == 4){
        std::cout << "You have chosen the Broyden solver" << std::endl;
        broyden<ContainerType>(fun, u, mMax, itmax, atol, rtol, damping);
    } else {
        std::cout << "Solver not specified. Defaulting to Anderson Acceleration" << std::endl;
        andacc<ContainerType>(fun, u, mMax, itmax, atol, rtol, beta, AAstart,damping);
    }
    ////Update u_
    std::rotate(u_.rbegin(), u_.rbegin() + 1, u_.rend()); //Rotate 1 to the right.
    dg::blas1::copy(u,u_[0]);
    std::cout<<"Exited nonlin solver" << std::endl;
    ////Update sum_x_prev
    
}
}
*/



} //namespace dg
