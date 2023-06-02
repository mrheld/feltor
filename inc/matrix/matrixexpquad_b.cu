// #define DG_DEBUG

#include <iostream>
#include <iomanip>

#include "dg/algorithm.h"
#include "lanczos.h"
#include "mcg.h"
#include "matrixfunction.h"

#include "gl_quadrature.h"

const double lx = 2.*M_PI;
const double ly = 2.*M_PI;
dg::bc bcx = dg::DIR;
dg::bc bcy = dg::PER;
const double m=4.;
const double n=4.;
const double alpha = 0.5;
const double ell_fac = (m*m+n*n);

//gauss Laguerre quadrature parameters
const int GLs=9; //gl quad with 5 for 12 GL points (use maximum 9 for 64 points) 
const double fac=1.; //scales error in GL quadrature. Small value for better accuracy, but works for matrix exponential

//ogata quadrature parameters
const double eps_o = 1e-17;
const double h_o=1e-4;

//Gauss Legendre quadrature parameters with Bessel zeros
const unsigned GLeg_n = 19;
const unsigned GLeg_Nx = 30;
const int GLeg_zeros = 30;
            
double lhs( double x, double y){ return sin(x*m)*sin(y*n);}

// double GyroLagK(int n, double x) { return pow(-x,n)/tgamma(n+1)*exp(x);}

double psi_Ogata(double x) { return x*tanh(M_PI/2. * sinh(x)); }

double psis_Ogata(double x) { return M_PI/2. * x * cosh(x) * pow(cosh(M_PI/2. * sinh(x)),-2.0) + tanh(M_PI/2. * sinh(x)) ; }

double weights_Ogata(double x) { return M_PI *boost::math::cyl_neumann(0,x)/boost::math::cyl_bessel_j(1, x); }

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
    dg::Elliptic<dg::CartesianGrid2d, Matrix, Container> A( {g, dg::centered, 1.0});

    std::function<double (double,double )> func2
    {
	[](double x, double t)
	{
	       	return boost::math::cyl_bessel_j(0, sqrt(4.0*t*alpha*x));
        }
    };

    std::vector< std::function<double (double)>> funcs{
        //Elliptic
        [](double x) { return dg::mat::GyrolagK<double>(0.,-alpha)(x);},
        //exp with elliptic via GLag quadrature
        [](double x) { 
            double integral=0.0;
            //exp(-alpha x) = int_0^inf dt exp(-t) BesselJ0(2*sqrt(t*alpha*x)) \approx w_i BesselJ0(sqrt(4*t_i*alpha*x))
            for (int i=0; i<GLo[GLs]; i++) integral+=GLw[GLs][i]*boost::math::cyl_bessel_j(0, sqrt(4.*GLx[GLs][i]*alpha*x*fac));
            return pow(integral,1./fac);
        },
        [](double x) {     
            double integral=0.0;
            //exp(-alpha x) = int_0^inf dt exp(-t) BesselJ0(2*sqrt(t*alpha*x)) \approx w_i BesselJ0(sqrt(4*t_i*alpha*x))
            for (int i=0; i<GLo[GLs]; i++) integral+=GLw[GLs][i]*boost::math::cyl_bessel_j(0, sqrt(4.*GLx[GLs][i]))*exp(-GLx[GLs][i]*(1.0/(alpha*x*fac)-1.0))/(alpha*x*fac);
            return pow(integral,1./fac);
        },
        //exp via Ogata quadrature 
        [](double x) { 
            double integral=0.0;
            int i=0;
            dg::mat::Gyrointegranttrans2<double> fg(x*alpha,0);
            while (fabs(psis_Ogata(J0Z[i]*h_o) * fg(psi_Ogata(J0Z[i]*h_o)/h_o)) >eps_o) {                
                integral+= weights_Ogata(J0Z[i]) * psis_Ogata(J0Z[i]*h_o) * fg(psi_Ogata(J0Z[i]*h_o)/h_o);
                i++;
            }
//             std::cout << "#    ogata-iter: "<<std::setw(3)<<i << "   for EV: " << x <<"\n"; //!sometimes negative eigenvalues appear!
            return integral;
        },
        //exp with elliptic via GLeg quadrature between Bessel zeros
        [](double x) {            
            double integral=0.0;
            for (int i=0; i<GLeg_zeros; i++)
            {
                double x0=0;
                double x1=pow(J0Z[i],2.0)/(4.*abs(x)*alpha); //!sometimes negative eigenvalues appear!
          	    if (i>0) x0 = pow(J0Z[i-1],2.0)/(4.*abs(x)*alpha); //!sometimes negative eigenvalues appear!
//                 std::cout << "i: " << i << "   x0: " << x0 << "   x1: " <<x1 << "   x:" << x << std::endl; //!sometimes negative eigenvalues appear!
                dg::Grid1d g1d( x0, x1, GLeg_n, GLeg_Nx);
                dg::mat::Gyrointegrant<double> fg(abs(x)*alpha,0); //!sometimes negative eigenvalues appear!
                const dg::HVec f = dg::evaluate( fg, g1d); 
                const dg::HVec w1d = dg::create::weights( g1d);
                integral+= dg::blas1::dot( w1d, f);
            }
            return integral;
        },
        //exp with elliptiva via GLeg quadrature with log transform
        [](double x) {            
            dg::Grid1d g1d( 0, 1, GLeg_n, GLeg_Nx);
            dg::mat::Gyrointegranttranslog<double> fg(abs(x)*alpha); //!sometimes negative eigenvalues appear!
            const dg::HVec f = dg::evaluate( fg, g1d); 
            const dg::HVec w1d = dg::create::weights( g1d);
            double integral= dg::blas1::dot(w1d, f);
            return integral;
        }
    };
    std::vector<std::string> outs = {
            "K_0(-alpha A)",
            //exp via quadrature rules
            "K_0_GLag(-alpha A)",
            "K_0_GLagalt(-alpha A)",
            "K_0_Ogata(-alpha A)",
            "K_0_GLegZ(-alpha A)",
            "K_0_GLeglog(-alpha A)"
    };
    for( unsigned u=0; u<funcs.size(); u++)
    {
        std::cout << "\n#Compute x = "<<outs[u]<<" b " << std::endl;

        Container x = dg::evaluate(lhs, g), x_exac(x), b(x), error(x);
        if (u>0 ) dg::blas1::scal(x_exac, funcs[0](ell_fac)); //use exact exp(-alpha A) for quadrature schemes
        else dg::blas1::scal(x_exac, funcs[u](ell_fac));

        std::cout << outs[u] << ":\n";
        
        dg::mat::UniversalLanczos<Container> krylovfunceigen( x, max_iter);
        auto func = dg::mat::make_FuncEigen_Te1( funcs[u]);
        //MLanczos-universal
        t.tic();
        iter = krylovfunceigen.solve(x, func, A, b, w2d, eps, 1., "universal", 1);
        t.toc();
        double time = t.diff();

        dg::blas1::axpby(1.0, x, -1.0, x_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, x_exac));

        std::cout << "    universal-time: "<<time<<"s \n";
        std::cout << "    universal-error: "<<erel  << "\n";
//         std::cout << "    universal-abserror: "<<sqrt(dg::blas2::dot( w2d, error) ) << "\n";
        std::cout << "    universal-iter: "<<std::setw(3)<<iter << "\n";
    }
    return 0;
}
