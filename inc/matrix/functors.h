#pragma once
#include <boost/math/special_functions.hpp>
namespace dg {
namespace mat {
/**
 * @brief \f$ f(x) = I_0 (x)\f$ with \f$I_0\f$ the zeroth order modified Bessel function
 *
 * @tparam T value type
 */
template < class T = double>
struct BESSELI0
{
    BESSELI0( ) {}
    /**
     * @brief return \f$ f(x) = I_0 (x)\f$ with \f$I_0\f$ the zeroth order modified Bessel function
     *
     * @param x x
     *
     * @return \f$ I_0 (x)\f$
     */
    T operator() ( T x) const
    {
        return boost::math::cyl_bessel_i(0, x);
    }
};

/**
 * @brief \f$ f(x) = J_n (x)\f$ with \f$J_n\f$ the n-th order modified Bessel function
 *
 * @tparam T value type
 */
template< class T = double >
struct BesselJ
{
    BesselJ(unsigned n): m_n(n) {}

    DG_DEVICE T operator() ( T x) const
    {
        return boost::math::cyl_bessel_j(m_n, x);
    }
    private:
    	unsigned m_n;
};

/**
 * @brief \f$ f(x) = L_n (x)\f$ with \f$L_n\f$ the n-th order Laguerre polynomial
 *
 * @tparam T value type
 */
template< class T = double >
struct LaguerreL
{
    LaguerreL(unsigned n): m_n(n) {}

    DG_DEVICE T operator() ( T x) const
    {
        return boost::math::laguerre(m_n, x);
    }
    private:
        unsigned m_n;
};


/**
 * @brief \f$ f(x) = x^p\f$
 *
 * @tparam T value type
 */
template< class T = double >
struct POW
{
    POW(double p): m_p(p) {}
    DG_DEVICE T operator() (T x) const
    {
	    return pow(x, m_p);
    }
	private:
        double m_p;
};

/**
 * @brief \f$ f(x) = \Gamma_0 (x) := I_0 (x) exp(x) \f$ with \f$I_0\f$ the zeroth order modified Bessel function
 *
 * @tparam T value type
 */
template < class T = double>
struct GAMMA0
{
    GAMMA0( ) {}
    /**
     * @brief return \f$ f(x) = I_0 (x) exp(x) \f$ with \f$I_0\f$ the zeroth order modified Bessel function
     *
     * @param x x
     *
     * @return \f$ \Gamma_0 (x)\f$
     */
    T operator() ( T x) const
    {
        return exp(x)*boost::math::cyl_bessel_i(0, x);
    }
};


/**
 * @brief \f$ f(x) = (-a*x)^n/n! exp(a*x) \f$ 
 *
 * @tparam T value type
 */
template<class T = double>
struct GyrolagK
{
    GyrolagK(double n, double a): m_n(n), m_a(a) {}

    DG_DEVICE T  operator()(double x) const { return pow(-x*m_a,m_n)/tgamma(m_n+1)*exp(x*m_a); }

    private:
        double  m_n, m_a;
};

/**
 * @brief \f$ f(x) =exp(-x)*L_n(x)*J_0(Sqrt(4.0*x*t))\f$ 
 *
 * @tparam T value type
 */
template<class T = double>
struct Gyrointegrant
{
    Gyrointegrant(double t, unsigned n):m_t(t), m_n(n) {}

    DG_DEVICE T  operator()(double x) const
    {
        return exp(-x)*boost::math::laguerre(m_n, x)*boost::math::cyl_bessel_j(0, sqrt(4.0*x*m_t));
    }
  private:
    double  m_t;
    unsigned m_n;
};

/**
 * @brief \f$ f(x) =x/2* exp(-x^2/4)*L_n(x^2/4)*J_0(x*Sqrt(t))\f$ 
 *
 * @tparam T value type
 */
template<class T = double>
struct Gyrointegranttrans
{
    Gyrointegranttrans(double t, unsigned n):m_t(t), m_n(n) {}

    DG_DEVICE T  operator()(double x) const
    {
        return x/2.0*exp(-x*x/4.)*boost::math::laguerre(m_n, x*x/4.)*boost::math::cyl_bessel_j(0, x*sqrt(m_t));
    }
  private:
    double m_t;
    unsigned m_n;
};

/**
 * @brief \f$ f(x) =x/(2*t) exp(-x^2/(4*t))*L_n((x^2/(4*t))*J_0(x)\f$ 
 *
 * @tparam T value type
 */
template<class T = double>
struct Gyrointegranttrans2
{
    Gyrointegranttrans2(double t, unsigned n):m_t(t), m_n(n) {}

    DG_DEVICE T  operator()(double x) const
    {
        return x/(2.0*m_t)*exp(-x*x/(4.*m_t))*boost::math::laguerre(m_n, x*x/(4.*m_t))*boost::math::cyl_bessel_j(0, x);
    }
  private:
    double m_t;
    unsigned m_n;
};

}//namespace mat
}//namespace dg
