#include <iostream>
#include <iomanip>

#include <mpi.h>
#ifdef _OPENMP
#include <omp.h>
#endif//_OPENMP
#include "algorithm.h"
#include "../geometries/geometries.h"


const double lx = 2*M_PI;
const double ly = 2*M_PI;
const double lz = 1.;

dg::bc bcx = dg::PER;
dg::bc bcy = dg::PER;
dg::bc bcz = dg::PER;
double left( double x, double y, double z) {return sin(x)*cos(y)*z;}
double right( double x, double y, double z) {return cos(x)*sin(y)*z;} 
//double right2( double x, double y) {return sin(y);}
double jacobian( double x, double y, double z) 
{
    return z*z*cos(x)*sin(y)*2*sin(2*x)*cos(2*y)-sin(x)*cos(y)*2*cos(2*x)*sin(2*y);
}

const double R_0 = 1000;
double fct(double x, double y, double z){ return sin(x-R_0)*sin(y);}
double derivative( double x, double y, double z){return cos(x-R_0)*sin(y);}
double laplace_fct( double x, double y, double z) { return -1./x*cos(x-R_0)*sin(y) + 2.*sin(y)*sin(x-R_0);}
double initial( double x, double y, double z) {return sin(0);}

typedef dg::MDMatrix Matrix;
typedef dg::MIDMatrix IMatrix;
typedef dg::MDVec Vector;


/*******************************************************************************
program expects npx, npy, npz, n, Nx, Ny, Nz from std::cin
outputs one line to std::cout 
# npx npy npz #procs #threads n Nx Ny Nz t_AXPBY t_DOT t_DX_per t_DY_per t_DZ_per t_ARAKAWA #iterations t_1xELLIPTIC_CG_dir_centered t_DS EXBLASCHECK
if Nz == 1, ds is not executed
Run with: 
>$ echo npx npy npz n Nx Ny Nz | mpirun -n#procs ./cluster_mpib 
 
 *******************************************************************************/

int main(int argc, char* argv[])
{
    MPI_Init( &argc, &argv);
    unsigned n, Nx, Ny, Nz; 
    ////////////////////////////////////////////////////////////////
    mpi_init3d( bcx, bcy, bcz, n, Nx, Ny, Nz, comm, std::cin, false);
    int rank;
    MPI_Comm_rank( comm, &rank);
    if(rank==0)
    {
        int dims[3], periods[3], coords[3];
        MPI_Cart_get( comm, 3, dims, periods, coords);
        std::cout<< dims[0] <<" "<<dims[1]<<" "<<dims[2]<<" "<<dims[0]*dims[1]*dims[2]<<" ";
        int num_threads = 1;
#ifdef _OPENMP
        num_threads = omp_get_max_threads( );
#endif //omp
        std::cout << num_threads<<" ";
        std::cout<< n <<" "<<Nx<<" "<<Ny<<" "<<Nz<<" ";
    }


    dg::CartesianMPIGrid3d grid( 0, lx, 0, ly, 0,lz, n, Nx, Ny, Nz, bcx, bcy, dg::PER, comm);
    dg::Timer t;
    Vector w3d = dg::create::weights( grid);
    Vector lhs = dg::evaluate ( left, grid), jac(lhs);
    Vector rhs = dg::evaluate ( right,grid);
    const Vector sol = dg::evaluate( jacobian, grid );
    Vector eins = dg::evaluate( dg::one, grid );
    std::cout<< std::setprecision(6);
    unsigned multi=100;

    dg::blas1::axpby( 1., lhs, -1., jac);
    //AXPBY
    t.tic();
    for( unsigned i=0; i<multi; i++)
        dg::blas1::axpby( 1., lhs, -1., jac);
    t.toc();
    if(rank==0)std::cout<<t.diff()/(double)multi<<" ";
    //DOT
    t.tic();
    double norm;
    for( unsigned i=0; i<multi; i++)
        norm = dg::blas1::dot( lhs, rhs);
    t.toc();
    if(rank==0)std::cout<<t.diff()/(double)multi<<" ";
    norm++;//avoid compiler warning
    //Matrix-Vector product
    Matrix dx = dg::create::dx( grid, dg::centered);
    t.tic(); 
    for( unsigned i=0; i<multi; i++)
        dg::blas2::symv( dx, rhs, jac);
    t.toc();
    if(rank==0)std::cout<<t.diff()/(double)multi<<" ";
    //Matrix-Vector product
    Matrix dy = dg::create::dy( grid, dg::centered);
    t.tic(); 
    for( unsigned i=0; i<multi; i++)
        dg::blas2::symv( dy, rhs, jac);
    t.toc();
    if(rank==0)std::cout<<t.diff()/(double)multi<<" ";
    if( Nz > 2)
    {
        //Matrix-Vector product
        Matrix dz = dg::create::dz( grid, dg::centered);
        t.tic(); 
        for( unsigned i=0; i<multi; i++)
            dg::blas2::symv( dz, rhs, jac);
        t.toc();
        if(rank==0)std::cout<<t.diff()/(double)multi<<" ";
    }

    //The Arakawa scheme
    dg::ArakawaX<dg::CartesianMPIGrid3d, Matrix, Vector> arakawa( grid);
    t.tic(); 
    for( unsigned i=0; i<multi; i++)
        arakawa( lhs, rhs, jac);
    t.toc();
    if(rank==0)std::cout<<t.diff()/(double)multi<<" ";
    //The Elliptic scheme
    periods[0] = false, periods[1] = false;
    MPI_Comm commEll;
    MPI_Cart_create( MPI_COMM_WORLD, 3, np, periods, true, &commEll);
    dg::CylindricalMPIGrid3d gridEll( R_0, R_0+lx, 0., ly, 0.,lz, n, Nx, Ny,Nz, dg::DIR, dg::DIR, dg::PER, commEll);
    const Vector ellw3d = dg::create::volume(gridEll);
    const Vector ellv3d = dg::create::inv_volume(gridEll);
    dg::Elliptic<dg::CylindricalMPIGrid3d, Matrix, Vector> laplace(gridEll, dg::not_normed, dg::centered);
    const Vector solution = dg::evaluate ( fct, gridEll);
    const Vector deriv = dg::evaluate( derivative, gridEll);
    Vector x = dg::evaluate( initial, gridEll);
    Vector b = dg::evaluate ( laplace_fct, gridEll);
    dg::blas2::symv( ellw3d, b, b);
    dg::CG< Vector > pcg( x, n*n*Nx*Ny);
    t.tic();
    unsigned number = pcg(laplace, x, b, ellv3d, 1e-6);
    t.toc();
    if(rank==0)std::cout << number << " "<<t.diff()/(double)number<<" "<<std::flush;
    dg::blas1::axpby( 1., solution, -1., x);
    exblas::udouble res;
    res.d = dg::blas2::dot( x, ellw3d, x);
    if( Nz > 1)
    {
        //Application of ds
        double gpR0  =  10, gpI0=20;
        double inv_aspect_ratio =  0.1;
        double gpa = gpR0*inv_aspect_ratio;
        double Rmin=gpR0-1.0*gpa;
        double Zmin=-1.0*gpa*1.00;
        double Rmax=gpR0+1.0*gpa; 
        double Zmax=1.0*gpa*1.00;
        dg::CylindricalMPIGrid3d g3d( Rmin,Rmax, Zmin,Zmax, 0, 2.*M_PI, n, Nx ,Ny, Nz,dg::DIR, dg::DIR, dg::PER,commEll);
        dg::geo::TokamakMagneticField magfield = dg::geo::createGuentherField(gpR0, gpI0);
        dg::geo::Fieldaligned<dg::aProductMPIGeometry3d, IMatrix, Vector> dsFA( magfield, g3d, dg::NEU, dg::NEU, dg::geo::FullLimiter());
        dg::geo::DS<dg::aProductMPIGeometry3d, IMatrix, Matrix, Vector>  ds ( dsFA, dg::not_normed, dg::centered);
        dg::geo::guenther::FuncNeu funcNEU(gpR0,gpI0);
        Vector function = dg::evaluate( funcNEU, g3d) , dsTdsfb(function);

        t.tic(); 
        for( unsigned i=0; i<multi; i++)
            ds.symv(function,dsTdsfb);
        t.toc();
        if(rank==0)std::cout<<t.diff()/(double)multi<<" ";
    }
    if(rank==0)std::cout << res.i<<" ";


    if(rank==0)std::cout <<std::endl;
    MPI_Finalize();
    return 0;
}
