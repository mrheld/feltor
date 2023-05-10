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

    std::vector< std::function<double (double)>> funcs{
        [](double x) { return dg::mat::GyrolagK<double>(0.,-alpha)(x);},
        [](double x) { return dg::mat::GyrolagK<double>(0.,-alpha)(x);},
        [](double x) { return dg::mat::GyrolagK<double>(0.,-alpha)(x);}        
    };
    std::vector<std::string> outs = {
            "K_0(-alpha A)",
            "K_0_prod(-alpha A)",
            "K_0_prodadj(-alpha A)"
//             "K_0_naive(-alpha A)"
    };
    for( unsigned u=0; u<funcs.size(); u++)
    {
        std::cout << "\n#Compute x = "<<outs[u]<<" b " << std::endl;

        Container x = dg::evaluate(lhs, g), x_exac(x), x_h(x), b(x), error(x);
        Container d = dg::evaluate(dg::ONE(), g);
//         Container d = dg::evaluate(dg::SinXSinY(0.99, 1.0, 1, 1), g);
        
        

        std::cout << outs[u] << ":\n";

        dg::mat::UniversalLanczos<Container> krylovfunceigen( x, max_iter);
        dg::mat::UniversalLanczos<Container> krylovfunceigend( x, max_iter);

        auto func = dg::mat::make_FuncEigen_Te1( funcs[u]);
        double time = t.diff();
        unsigned iter_sum=0;
	//MLanczos-universal
        

        if (u==0)
        {
            t.tic();
            iter= krylovfunceigen.solve(x, func, A, b, w2d, eps, 1., "universal");
            t.toc();
            time = t.diff();
            std::cout << "    universal-iter: "<<std::setw(3)<<iter << "\n";
            
            dg::blas1::scal(x_exac, funcs[u](ell_fac));

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
                dg::blas1::copy( 0, c[l]); // init sum
                dg::blas1::copy( 0, v[l]); // init sum
                //e_l
                e_l[l] = 1.;
                if (l>0) e_l[l-1]=0.;
                //Compute c[l]
                for( unsigned j=0; j<iter; j++)
                {
                    dg::blas1::axpby( evals[j], d, 0., fd);
                    dg::blas1::transform(fd, fd, dg::mat::GyrolagK<double>(0.,-alpha));
                    dg::blas1::axpby( evecs(0,j)*evecs(l,j), fd, 1., c[l]);
                }
                //compute v[l]
                krylovfunceigen.normMbVy(A, Tf, e_l, v[l], b, krylovfunceigen.get_bnorm()); //v[l]=  ||b|| V e_l

                //compute x+=v[l] p. c[l]
                dg::blas1::pointwiseDot(1.0, v[l], c[l], 1.0, x);
            }
            t.toc();
            time = t.diff();
            
            //Compute exact error for product exponential
            dg::blas1::axpby(ell_fac, d, 0.0, fd);
            dg::blas1::transform(fd, fd, dg::mat::GyrolagK<double>(0.,-alpha));
            dg::blas1::pointwiseDot(fd, x_exac, x_exac); //f(-alpha*(m^2+n^2) d) sin(m x) cos(n y)
        }
        if (u==2)
        {
            t.tic();
            //Tridiagonalize diagonal matrix D
            auto Rf = krylovfunceigend.tridiag(func, d,  b, w2d, eps, 1.,  "universal");
            unsigned iter_Rf = krylovfunceigend.get_iter();
            std::cout << "    universal-iter-Rf: "<<std::setw(3)<< iter_Rf << "\n";
//             cusp::print(Rf);
            
            //make eigendecomposition of Rf = E_Rf  eval_Rf E_Rf^T 
            cusp::array2d< double, cusp::host_memory> evecs_Rf(iter_Rf,iter_Rf);
            cusp::array1d< double, cusp::host_memory> evals_Rf(iter_Rf);
            cusp::lapack::stev(Rf.values.column(1), Rf.values.column(2), evals_Rf, evecs_Rf, 'V');
            cusp::coo_matrix<int, double, cusp::host_memory> E_Rf, E_Rf_t;
            cusp::convert(evecs_Rf, E_Rf);
            cusp::transpose(E_Rf, E_Rf_t);
            
            
            //Compute h_k
            dg::HVec e_1(iter_Rf,0.), e_k(e_1), y(e_1); //unit vector e_1
            std::vector<dg::HVec> h{iter_Rf, e_1};
            std::vector<Container> v{iter_Rf, d}; 
            e_1[0] = 1.;
            
            dg::blas2::symv(E_Rf_t, e_1, y); //y = E_Rf^T e_1
            dg::blas1::scal(x,0.0);
            for( unsigned k=0; k<iter_Rf; k++)
            {
                e_k[k] = 1.;
                if (k>0) e_k[k-1]=0.;
                
                dg::blas1::pointwiseDot(e_k, y, y); //y = e_k * (E_Rf^T e_1)
                dg::blas2::symv(E_Rf, y, h[k]);
                
                krylovfunceigend.normMbVy(d, Rf, h[k], v[k], b, krylovfunceigend.get_bnorm()); //v[k]=  ||b|| V_Rf h[k]
                
                //Solve 
                iter= krylovfunceigen.solve(x_h, func, A, v[k], w2d, eps, 1., "universal"); // x_h = ||v_k|| V_Tf f(Tf lambda_Rf,k) v[k]
                dg::blas1::axpby(1.0, x_h, 1.0, x);
                std::cout << "    universal-iter-Tf: "<<std::setw(3)<< krylovfunceigen.get_iter() << "\n";

            }
            
            //Tridiagonalize A first to T with the stopping condition for the function exp(-max(d)*alpha A)
            t.toc();
            time = t.diff();
            
            x_exac = dg::evaluate(lhs, g);
            dg::blas1::scal(x_exac, funcs[u](ell_fac));

        }
//         if (u==3)
//         {
//             t.tic();
//             for( unsigned k=0; k<x.size(); k++)
//             {
// //                 A.set_chi(lambda_d[k]);
//                 iter= krylovfunceigen.solve(x_h, func, A, b, w2d, eps, 1., "universal");
// //                 std::cout << "    universal-iter[k]: "<<std::setw(3)<<iter <<"[" << k <<"]"<< "\n";
//                 iter_sum+=iter;
//                 x[k] = x_h[k];
//             }
//             t.toc();
//             time = t.diff();
//             std::cout << "    universal-iter_sum: "<<std::setw(3)<<iter_sum << "\n";
//         }
        dg::blas1::axpby(1.0, x, -1.0, x_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, x_exac));

        std::cout << "    universal-time: "<<time<<"s \n";
        std::cout << "    universal-error: "<<erel  << "\n";
        
    }
    return 0;
}
