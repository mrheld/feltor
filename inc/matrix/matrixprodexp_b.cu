// #define DG_DEBUG

#include <iostream>
#include <iomanip>

#include "dg/algorithm.h"
#include "lanczos.h"
#include "mcg.h"
#include "matrixfunction.h"

#include "gl_quadrature.h"

#include <cusp/transpose.h>
#include <cusp/array1d.h>
#include <cusp/array2d.h>
#include <cusp/print.h>

#include <cusp/lapack/lapack.h>


const double lx = 2.*M_PI;
const double ly = 2.*M_PI;
dg::bc bcx = dg::DIR;
dg::bc bcy = dg::PER;
const double m=4.;
const double n=4.;
const double alpha = 0.5;
const double ell_fac = (m*m+n*n);

double lhs( double x, double y){ return sin(x*m)*sin(y*n);}

///@brief \f$ f(x) = BesselJ(n,x)\f$
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

///@brief \f$ f(x) = LaguerreL(n,x)\f$
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

///@brief \f$ f(x) = (-x)^n/n! exp(x)\f$
double GyroLagK(int n, double x) { return pow(-x,n)/tgamma(n+1)*exp(x);}

///@brief \f$ f(x) = (-a*x)^n/n! exp(a*x)\f$
template<class T = double>
struct GyrolagKop
{
    GyrolagKop(int n, double a): m_n(n), m_a(a) {}

    DG_DEVICE T  operator()(double x) const { return pow(-x*m_a,m_n)/tgamma(m_n+1)*exp(x*m_a); }
  private:
    double  m_a;
    unsigned m_n;
};

///@brief \f$ f(x) = exp(-x)*LaguerreL(n,x)*BesselJ(0,2*Sqrt(x*t))\f$
template<class T = double>
struct Gyrointegrant
{
    Gyrointegrant(double t, unsigned n):m_t(t), m_n(n) {}

    DG_DEVICE T  operator()(double x) const
    {
        return exp(-x)*boost::math::laguerre(m_n, x)*boost::math::cyl_bessel_j(0, 2.0*sqrt(x*m_t));
    }
  private:
    double  m_t;
    unsigned m_n;
};
using Matrix = dg::DMatrix;
using Container = dg::DVec;

int main(int argc, char * argv[])
{
    dg::Timer t;

    unsigned n, Nx, Ny;
    std::cout << "# Type n, Nx and Ny! \n";
    std::cin >> n >> Nx >> Ny;
    std::cout <<"# You typed\n"
              <<"n:  "<<n<<"\n"
              <<"Nx: "<<Nx<<"\n"
              <<"Ny: "<<Ny<<std::endl;
    unsigned iter = 0;

    unsigned max_iter = 1;
    std::cout << "# Type max_iter of tridiagonalization (500) ?\n";
    std::cin >> max_iter ;
    std::cout << "# Type in eps of tridiagonalization (1e-7)\n";
    double eps = 1e-7; //# of pcg iter increases very much if
    std::cin >> eps;
    std::cout <<"# You typed\n"
              <<"max_iter: "<<max_iter<<"\n"
              <<"eps: "<<eps<<std::endl;

    double erel = 0;

    dg::Grid2d g( 0, lx, 0, ly,n, Nx, Ny, bcx, bcy);
    const Container w2d = dg::create::weights( g);

    double max_weights = dg::blas1::reduce(w2d, 0., dg::AbsMax<double>() );
    double min_weights = dg::blas1::reduce(w2d, max_weights, dg::AbsMin<double>() );
    std::cout << "#   min(W)  = "<<min_weights <<"  max(W) = "<<max_weights << "\n";
    const double kappa = sqrt(max_weights/min_weights); //condition number
    dg::Elliptic<dg::CartesianGrid2d, Matrix, Container> A( {g, dg::centered, 1.0});
    dg::mat::UniversalLanczos<Container> lanczos( A.weights(), 20);
    auto T = lanczos.tridiag( A, A.weights(), A.weights());
    auto extremeEVs = dg::mat::compute_extreme_EV( T);
    double EVmin = extremeEVs[0];
    double EVmax = extremeEVs[1];


    std::vector< std::function<double (double)>> funcs{
	//Elliptic
        [](double x) { return GyroLagK(0,-alpha*x);},
        [](double x) { return GyroLagK(0,-alpha*x);}
    };
    std::vector<std::string> outs = {
            "K_0_naive(-alpha A)",
            "K_0_fast(-alpha A)"
    };
    for( unsigned u=0; u<funcs.size(); u++)
    {
        std::cout << "\n#Compute x = "<<outs[u]<<" b " << std::endl;

        Container x = dg::evaluate(lhs, g), x_exac(x), x_h(x), b(x), error(x);
        Container d = dg::evaluate(dg::ONE(), g);
        dg::blas1::scal(x_exac, funcs[u](ell_fac));
        

        std::cout << outs[u] << ":\n";

        dg::mat::UniversalLanczos<Container> krylovfunceigen( x, max_iter);

        auto func = dg::mat::make_FuncEigen_Te1( funcs[u]);
        double time = t.diff();
        unsigned iter_sum=0;
	//MLanczos-universal
        
        if (u==0)
        {
            t.tic();
            for( unsigned k=0; k<x.size(); k++)
            {
//                 A.set_chi(lambda_d[k]);
                iter= krylovfunceigen.solve(x_h, func, A, b, w2d, eps, 1., "universal");
//                 std::cout << "    universal-iter[k]: "<<std::setw(3)<<iter <<"[" << k <<"]"<< "\n";
                iter_sum+=iter;
                x[k] = x_h[k];
            }
            t.toc();
            time = t.diff();
            std::cout << "    universal-iter_sum: "<<std::setw(3)<<iter_sum << "\n";
        }
        if (u==1)
        {
            t.tic();
            //Tridiagonalize A first to T with the stopping condition for the function exp(-max(d)*alpha A)
            auto Tf = krylovfunceigen.tridiag(func, A,  b, w2d, eps, 1.,  "universal");
//          cusp::print(Tf);
            iter = krylovfunceigen.get_iter();
            std::cout << "    universal-iter: "<<std::setw(3)<< iter << "\n";
            
            //make eigendecomposition of f(d T) e_1 = E_T f(d eval_T) E_T^T e_1
            cusp::array2d< double, cusp::host_memory> evecs(iter,iter);
            cusp::array1d< double, cusp::host_memory> evals(iter);
            cusp::lapack::stev(Tf.values.column(1), Tf.values.column(2), evals, evecs, 'V');
//             cusp::print(evecs);
            
            //Compute c[l], v[l] and utlize them for x
            std::vector<Container> c{iter,d}, v{iter,d};
            dg::HVec e_l(iter,0.); //unit vector e_l
            Container fd(d); // helper variable
            dg::blas1::scal(x,0.0);

            for( unsigned l=0; l<iter; l++)
            {
//              std::cout << "l " <<l << std::endl;
                dg::blas1::copy( 0, c[l]); // init sum
                dg::blas1::copy( 0, v[l]); // init sum
                //e_l
                e_l[l] = 1.;
                if (l>0) e_l[l-1]=0.;
                //Compute c[l]
                std::cout << "cl \n";
                for( unsigned j=0; j<iter; j++)
                {
                    std::cout << "j " << j <<  "   " << evals[j] << std::endl;
                    dg::blas1::axpby( evals[j], d, 0., fd);                        
                    dg::blas1::transform(fd, fd, GyrolagKop<double>(0,-1.*alpha));
                    dg::blas1::axpby( evecs(0,j)*evecs(l,j), fd, 1., c[l]);
                }
                std::cout << "vl \n";
                //compute v[l]
                krylovfunceigen.normMbVy(A, Tf, e_l, v[l], b, krylovfunceigen.get_bnorm()); //v[l]=  ||b|| V e_l
                std::cout << "x \n";

                //compute x+=v[l] p. c[l]
                dg::blas1::pointwiseDot(1.0, v[l], c[l], 1.0, x);
                std::cout << " after x \n";

            }
            std::cout << "x2 \n";

            t.toc();
            time = t.diff();
                        std::cout << "x3 \n";

        }
                        std::cout << "er \n";

        dg::blas1::axpby(1.0, x, -1.0, x_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, x_exac));

        std::cout << "    universal-time: "<<time<<"s \n";
        std::cout << "    universal-error: "<<erel  << "\n";
        
    }
    return 0;
}
