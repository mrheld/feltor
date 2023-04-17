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

std::vector<double> BesselJ0Zeros = 
{
	0.0, //not a zero but for grid
	2.404825557695772768621631879326454643124,
	5.520078110286310649596604112813027425222,
	8.653727912911012216954198712660946685566,
	11.79153443901428161374304491192545892202,
	14.93091770848778594776259399738868220792,
	18.07106396791092254314788297561817656025,
	21.21163662987925895907839335052630683618,
	24.35247153074930273705794476317890718457,
	27.49347913204025479587728823460741454653,
	30.63460646843197511754957892685423273727
};

double lhs( double x, double y){ return sin(x*m)*sin(y*n);}

double GyroLagK(int n, double x) { return pow(-x,n)/tgamma(n+1)*exp(x);}

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

///@brief \f$ f(x) = x^power\f$
template< class T = double >
struct POW
{
    POW(double power): m_power(power) {}
    DG_DEVICE T operator() (T x) const
    {
	    return pow(x, m_power);
    }
	private:
        double m_power;
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
//    double GLeg_gmax = 1000;
//    unsigned GLeg_n = 5;
//    unsigned GLeg_Nx = 5;
//    std::cout << "# Type g_max, n, Nx of Gauss-Legendre quadrature \n";
//    std::cin >> GLeg_gmax, GLeg_n, GLeg_Nx;

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

    std::function<double (double,double )> func2
    {
	[](double x, double t)
	{
	       	return boost::math::cyl_bessel_j(0, sqrt(4.0*t*alpha*x));
        }
    };

    std::vector< std::function<double (double)>> funcs{
	//Elliptic
        [](double x) { return GyroLagK(0,-alpha*x);},
        [](double x) { return GyroLagK(1,-alpha*x);},
        [](double x) { return GyroLagK(5,-alpha*x);},
        [](double x) { return GyroLagK(10,-alpha*x);},
	//Helmholtz
        [](double x) { return 1./(1.+alpha*x);},
	//exp with elliptic via GLag quadrature
        [](double x) { 
            int GLs=4; //gl quad with 5 (use 9 for 64) points
            double fac=1e-10; //scales error constant
            double integral=0.0;
            //exp(-alpha x) = int_0^inf dt exp(-t) BesselJ0(2*sqrt(t*alpha*x)) \approx w_i BesselJ0(sqrt(4*t_i*alpha*x))
            for (unsigned i=0; i<GLo[GLs]; i++) integral+=GLw[GLs][i]*boost::math::cyl_bessel_j(0, sqrt(4.*GLx[GLs][i]*alpha*x*fac));
            return pow(integral,1./fac);
        },
	[](double x) {
            return boost::math::cyl_bessel_j(0, sqrt(4.0*alpha*x));
	},
       //exp with elliptic via GLeg quadrature
        [](double x) {            
            const unsigned GLeg_n = 3;
            const unsigned GLeg_Nx = 3;
	    double integral=0.0;
            for (unsigned i=0; i<5; i++)
	    {
          	    dg::Grid1d g1d( BesselJ0Zeros[i]*sqrt(x*alpha), BesselJ0Zeros[i+1]*sqrt(x*alpha), GLeg_n, GLeg_Nx);
                    Gyrointegrant<double> gyroint(x*alpha,0);
                    const dg::HVec f = dg::evaluate( gyroint, g1d); //f=exp(-t) BesselJ0(2*sqrt(t*alpha*x))
                    const dg::HVec w1d = dg::create::weights( g1d);
                    integral+= dg::blas1::dot( w1d, f);
	    }
            return integral;
        }
        //exp with elliptic via GLeg quadrature
/*        [](double x) {
            const double GLeg_gmax = 50;
            const unsigned GLeg_n = 19;
            const unsigned GLeg_Nx = 500;

            dg::Grid1d g1d( 0, GLeg_gmax, GLeg_n, GLeg_Nx);
            dg::HVec w1d_ref, a1d_ref; 
	    dg::ExponentialRefinement expref(0 , g1d.N());
            expref.generate( g1d, w1d_ref, a1d_ref);	    

	    Gyrointegrant<double> gyroint(x*alpha,0); 
            const dg::HVec f = dg::evaluate( gyroint, g1d); //f=exp(-t) BesselJ0(2*sqrt(t*alpha*x))
            const dg::HVec w1d = dg::create::weights( g1d);

            dg::HVec v(f);
	    dg::blas1::transform(a1d_ref, v, Gyrointegrant<double>(x*alpha,0));
	    dg::blas1::pointwiseDot(w1d, w1d_ref, w1d_ref); //multiply with volume of refinement

            return dg::blas1::dot( w1d_ref, v);
        }
 */
    };
    std::vector<std::string> outs = {
            "K_0(-alpha A)",
            "K_1(-alpha A)",
            "K_5(-alpha A)",
            "K_10(-alpha A)",
            "Inv(1+alpha A)",
            "K_0_GLag(-alpha A)",
            "K_0_GLagsep(-alpha A)",
            "K_0_GLeg(-alpha A)"  
    };
    for( unsigned u=0; u<funcs.size(); u++)
    {
        std::cout << "\n#Compute x = "<<outs[u]<<" b " << std::endl;

        Container x = dg::evaluate(lhs, g), x_exac(x), b(x), error(x);
        if (u==5 || u==6 ) dg::blas1::scal(x_exac, funcs[0](ell_fac)); //use exact exp(-alpha A) for GLag and GLeg
        else dg::blas1::scal(x_exac, funcs[u](ell_fac));

        double res_fac = kappa*funcs[u](EVmin);
        std::cout << "#   min(EV) = "<<EVmin <<"  max(EV) = "<<EVmax << "\n";
        std::cout << "#   kappa   = "<<kappa <<"\n";
        std::cout << "#   res_fac = "<<res_fac<< "\n";
        std::cout << outs[u] << ":\n";

	//MLanczos-residual
        dg::mat::UniversalLanczos<Container> krylovfunceigen( x, max_iter);
        t.tic();
        auto func = dg::mat::make_FuncEigen_Te1( funcs[u]);
	if (u==6){
	        int GLs=4; //gl quad with 5 (use 9 for 64) points
		dg::blas1::scal(x,0.0);
		Container y(x);
                for (int i=0; i<GLo[GLs]; i++) 
		{
                   A.set_chi(GLx[GLs][i]);
		   iter = krylovfunceigen.solve(y, func, A, b, w2d, eps, 1., "residual", res_fac);
		   dg::blas1::axpby(GLw[GLs][i], y, 1.0, x); //sum of integral
                }
		A.set_chi(1.0);
	}
	else{
	        iter = krylovfunceigen.solve(x, func, A, b, w2d, eps, 1.,
                "residual", res_fac);
	}
        t.toc();
        double time = t.diff();

        dg::blas1::axpby(1.0, x, -1.0, x_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, x_exac));

        std::cout << "    residual-time: "<<time<<"s \n";
        std::cout << "    residual-error: "<<erel  << "\n";
        std::cout << "    residual-iter: "<<std::setw(3)<<iter << "\n";

	//MCG
        dg::mat::MCGFuncEigen<Container> mcgfunceigen( x, max_iter);
        t.tic();
        iter = mcgfunceigen(x, funcs[u], A, b, w2d, eps, 1.,
                res_fac);
        t.toc();
        time = t.diff();

        dg::blas1::axpby(1.0, x, -1.0, x_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, x_exac));

        std::cout << "    mcg-time: "<<time<<"s \n";
        std::cout << "    mcg-error: "<<erel  << "\n";
        std::cout << "    mcg-iter: "<<std::setw(3)<<iter << "\n";

	//MLanczos-universal
        t.tic();
        if (u==6){
                int GLs=4; //gl quad with 5 (use 9 for 64) points
		dg::blas1::scal(x,0.0);
                Container y(x);
                for (int i=0; i<GLo[GLs]; i++)
                {
                   A.set_chi(GLx[GLs][i]);
                   iter = krylovfunceigen.solve(y, func, A, b, w2d, eps, 1., "universal");
                   dg::blas1::axpby(GLw[GLs][i], y, 1.0, x); //sum of quadrature integral
                }
                A.set_chi(1.0);
        }
        else{
            iter = krylovfunceigen.solve(x, func, A, b, w2d, eps, 1.,
                "universal");
	}
        t.toc();
        time = t.diff();

        dg::blas1::axpby(1.0, x, -1.0, x_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, x_exac));

        std::cout << "    universal-time: "<<time<<"s \n";
        std::cout << "    universal-error: "<<erel  << "\n";
        std::cout << "    universal-iter: "<<std::setw(3)<<iter << "\n";
    }
    return 0;
}
