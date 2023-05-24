// #define DG_DEBUG

#include <iostream>
#include <iomanip>

#include "dg/algorithm.h"
#include "dg/file/file.h"

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
const double ms=2.;
const double ns=2.;
const double alpha = 1./2.;
const double ell_fac = (m*m+n*n);
const double ell_facs = (ms*ms+ns*ns);

double lhs( double x, double y){ return sin(x*m)*sin(y*n);}
double lhss( double x, double y){ return sin(x*ms)*sin(y*ns);}

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
        [](double x) { return dg::mat::GyrolagK<double>(0.,-alpha)(x);},
        [](double x) { return dg::mat::GyrolagK<double>(0.,-alpha)(x);},
        [](double x) { return dg::mat::GyrolagK<double>(0.,-alpha)(x);}        
    };
    std::vector<std::string> outs = {
            "K_0(-alpha A)",
            "K_0(d, -alpha A)",
            "K_0(-alpha A, d)",
            "K_0_naive(d, -alpha A)",
            "K_0_naive(-alpha A, d)"
    };
    
    //Plot into netcdf file
    size_t start = 0;
    dg::file::NC_Error_Handle err;
    int ncid;
    err = nc_create( "visual.nc", NC_NETCDF4|NC_CLOBBER, &ncid);
    int dim_ids[5], tvarID;
    err = dg::file::define_dimensions( ncid, dim_ids, &tvarID, g);

    std::string names[5] = {"K0","K0_prod","K0_prodadj","K0_prod_naive","K0_prodadj_naive"};
    int dataIDs[5];
    for( unsigned i=0; i<5; i++){
    err = nc_def_var( ncid, names[i].data(), NC_DOUBLE, 3, dim_ids, &dataIDs[i]);}

    dg::HVec transferH(dg::evaluate(dg::zero, g));
        
    for( unsigned u=0; u<funcs.size(); u++)
    {
        std::cout << "\n#Compute x = "<<outs[u]<<" b " << std::endl;

        Container x = dg::evaluate(lhs, g), x_exac(x), x_h(x), b(x), error(x);
        Container one = dg::evaluate(dg::ONE(), g);
        
//         Container d = dg::evaluate(dg::ONE(), g);
//         Container d = dg::evaluate(dg::SinXSinY(0.5, 1.0, 1, 1), g);
        Container d = dg::evaluate(dg::Cauchy(lx/2., ly/2., 3./2., 3./2., 5.0), g); //bump function
//         dg::blas1::plus(d, 1.0);
        
        Container b_h(b);
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
        }
        if (u==1)
        {
            t.tic();
            //Tridiagonalize A first to T with the stopping condition for the function exp(-max(d)*alpha A)
            auto Tf = krylovfunceigen.tridiag(func, A,  b, w2d, eps, 1.,  "universal");
            iter = krylovfunceigen.get_iter();
            
            //make eigendecomposition of f(d T) e_1 = E_T f(d eval_T) E_T^T e_1
            cusp::array2d< double, cusp::host_memory> evecs(iter,iter);
            cusp::array1d< double, cusp::host_memory> evals(iter);
            cusp::lapack::stev(Tf.values.column(1), Tf.values.column(2), evals, evecs, 'V');
            
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
        }
        if (u==2)
        {   
            t.tic();
             //Tridiagonalize A first to T with the stopping condition for the function exp(-max(d)*alpha A)
            auto Tf = krylovfunceigen.tridiag(func, A,  b, w2d, eps, 1.,  "universal");
            iter = krylovfunceigen.get_iter();
            
            //make eigendecomposition of f(d T) e_1 = E_T f(d eval_T) E_T^T e_1
            cusp::array2d< double, cusp::host_memory> evecs(iter,iter);
            cusp::array1d< double, cusp::host_memory> evals(iter);
            cusp::lapack::stev(Tf.values.column(1), Tf.values.column(2), evals, evecs, 'V');
            
            //Compute c[l], v[l] and utlize them for x
            std::vector<Container> v{iter,d}, c{iter,d};
            dg::HVec e_k(iter, 0.); //unit vector e_k
            Container fd(d); // helper variable
            dg::blas1::scal(x, 0.0);
            
            //precompute v_l
            for( unsigned k=0; k<iter; k++)
            {
                dg::blas1::copy( 0, v[k]); // init sum
                dg::blas1::copy( 0, c[k]); // init sum
                //e_l
                e_k[k] = 1.;
                if (k>0) e_k[k-1]=0.;
                //compute v[l]
                krylovfunceigen.normMbVy(A, Tf, e_k, v[k], b, 1.0); //v_k=  V e_k
            }
            //Compute v[k]
            for( unsigned l=0; l<iter; l++)
            {
                for( unsigned i=0; i<iter; i++)
                {        
                    dg::blas1::axpby( evals[i], d, 0., fd); //fd = lambda_i d
                    dg::blas1::transform(fd, fd, dg::mat::GyrolagK<double>(0.,-alpha)); //fd =  f(lambda_i d)
                    for( unsigned k=0; k<iter; k++)
                    {
                        dg::blas1::pointwiseDot(evecs(i,l)*evecs(k,i), fd, v[k], 1.0, c[l]); //c_l += (eps_{i,l} eps_{k,i}) f(lambda_i d) * v_k
                    }
                }
                dg::blas1::axpby(dg::blas2::dot(c[l], w2d, b),  v[l], 1., x); //x += (c_l.M b) v_l 
            }
            dg::blas1::scal(x, 1./krylovfunceigen.get_bnorm()/krylovfunceigen.get_bnorm()); // x= x/||b||_M^2
            t.toc();
            time = t.diff();
            
        }
        if (u==3) 
        {
            t.tic();
            double lambda_d = 0.;
            for( unsigned k=0; k<x.size(); k++)
            {
                lambda_d = d[k];
                A.set_chi(lambda_d);
                iter = krylovfunceigen.solve(x_h, func, A, b, w2d, eps, 1., "universal");
                iter_sum+=iter;
                x[k] = x_h[k];
            }
            t.toc();
            time = t.diff();
        }
        if (u==4) 
        {
            dg::blas1::scal(x, 0.0);
            double lambda_d = 0.;
            iter_sum=0;
            t.tic();
            for( unsigned k=0; k<x.size(); k++)
            {
                lambda_d = d[k];
                A.set_chi(lambda_d);
                dg::blas1::scal(b_h, 0.0);
                b_h[k] = b[k];
                iter = krylovfunceigen.solve(x_h, func, A, b_h, w2d, eps, 1., "universal");
                iter_sum+=iter;
                dg::blas1::axpby(1.0, x_h, 1.0, x);
            }
            t.toc();
            time = t.diff();
        }
        //write solution into file
        dg::assign( x, transferH);
        dg::file::put_vara_double( ncid, dataIDs[u], start, g, transferH);
      
        //Compute errors
        if (u==0)
        {
            dg::blas1::scal(x_exac, funcs[u](ell_fac));
        }
        else 
        {
            Container fd(d); // helper variable
            //Compute absolute and relative error in adjointness //not useful if the operator is self-adjoint! use general g!
            if (u==2 || u==4)
            {
                x_h = dg::evaluate(lhss, g); // -> g
                dg::blas1::axpby(ell_facs, d, 0.0, fd);
                dg::blas1::transform(fd, fd, dg::mat::GyrolagK<double>(0.,-alpha));
                dg::blas1::pointwiseDot(fd, x_h, x_exac); //x_exac = f(-alpha*(ms^2+ns^2) d) sin(x*ms) cos(y*ms) \equiv exp(d,-alpha A) g
                x_h = dg::evaluate(lhs, g); // -> f
                double fOg = dg::blas2::dot( x_h, w2d, x_exac); //<f,exp(d,-alpha A) g>
                std::cout << "<f, exp(d,-alpha A) g> = " << fOg << std::endl;
                x_h = dg::evaluate(lhss, g); // -> g
                double gOadjf = dg::blas2::dot( x, w2d, x_h); //<exp(-alpha A, d)f, g>
                std::cout << "<exp(-alpha A, d)f, g> = " << gOadjf << std::endl;

                double eabs_adj = fOg-gOadjf; // <f,exp(d,-alpha A) g> -<exp(-alpha A, d)f, g>
                std::cout << "    universal-abserror-adjointness: "<< eabs_adj  << "\n"; 
                fOg = eabs_adj/fOg; //(<f,exp(d,-alpha A) g> -<exp(-alpha A, d)f, g>)/<f,exp(d,-alpha A) g> //does a relative error make sense here?
                std::cout << "    universal-relerror-adjointness: "<< fOg  << "\n";
            }
            
            //Compute exact error for product exponential (is used also for adjoint product exponential since we have no analytical solution there)
            x_h = dg::evaluate(lhs, g);
            dg::blas1::axpby(ell_fac, d, 0.0, fd);
            dg::blas1::transform(fd, fd, dg::mat::GyrolagK<double>(0.,-alpha));
            dg::blas1::pointwiseDot(fd, x_h, x_exac); //x_exac = f(-alpha*(m^2+n^2) d) sin(m x) cos(n y)
        }        
        dg::blas1::axpby(1.0, x, -1.0, x_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, x_exac));
        std::cout << "    universal-iter: "<<std::setw(3)<< iter << "\n";
        if (u==3 || u==4) std::cout << "    universal-iter_sum: "<<std::setw(3)<<iter_sum << "\n";
        std::cout << "    universal-time: "<<time<<"s \n";
        std::cout << "    universal-error: "<<erel  << "\n";
    }
    err = nc_close(ncid);

    return 0;
}
