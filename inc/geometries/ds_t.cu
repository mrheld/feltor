#include <iostream>

#include <cusp/print.h>

#include "dg/backend/functions.h"
#include "dg/backend/timer.cuh"
#include "dg/blas.h"
#include "dg/functors.h"
#include "dg/geometry/geometry.h"
#include "ds.h"
#include "solovev.h"
#include "flux.h"
#include "toroidal.h"


const double R_0 = 10;
const double I_0 = 2;
double func(double R, double Z, double phi)
{
    double r2 = (R-R_0)*(R-R_0)+Z*Z;
    return r2*sin(phi);
}
double deri(double R, double Z, double phi)
{
    double r2 = (R-R_0)*(R-R_0)+Z*Z;
    return I_0/R/sqrt(I_0*I_0 + r2)* r2*cos(phi);
}

int main(int argc, char * argv[])
{
    std::cout << "First test the cylindrical version\n";
    std::cout << "Type n, Nx, Ny, Nz\n";
    unsigned n, Nx, Ny, Nz;
    std::cin >> n>> Nx>>Ny>>Nz;
    std::cout << "You typed "<<n<<" "<<Nx<<" "<<Ny<<" "<<Nz<<std::endl;
    dg::CylindricalGrid3d g3d( R_0 - 1, R_0+1, -1, 1, 0, 2.*M_PI, n, Nx, Ny, Nz, dg::NEU, dg::NEU, dg::PER);
    const dg::DVec vol3d = dg::create::volume( g3d);
    dg::Timer t;
    t.tic();
    dg::geo::TokamakMagneticField mag = dg::geo::createCircularField( R_0, I_0);
    dg::geo::BinaryVectorLvl0 bhat( (dg::geo::BHatR)(mag), (dg::geo::BHatZ)(mag), (dg::geo::BHatP)(mag));
    dg::geo::Fieldaligned<dg::aProductGeometry3d,dg::IDMatrix,dg::DVec>  dsFA( bhat, g3d, 2,2,true,true,1e-10, dg::NEU, dg::NEU, dg::geo::NoLimiter());

    dg::geo::DS<dg::aProductGeometry3d, dg::IDMatrix, dg::DMatrix, dg::DVec> ds( dsFA, dg::not_normed, dg::centered);
    t.toc();
    std::cout << "Creation of parallel Derivative took     "<<t.diff()<<"s\n";

    dg::DVec function = dg::evaluate( func, g3d), derivative(function);
    const dg::DVec solution = dg::evaluate( deri, g3d);
    t.tic();
    ds( function, derivative);
    t.toc();
    std::cout << "Application of parallel Derivative took  "<<t.diff()<<"s\n";
    dg::blas1::axpby( 1., solution, -1., derivative);
    double norm = dg::blas2::dot( vol3d, solution);
    std::cout << "Norm Solution "<<sqrt( norm)<<"\n";
    std::cout << "Relative Difference Is "<< sqrt( dg::blas2::dot( derivative, vol3d, derivative)/norm )<<"\n";
    std::cout << "Error is from the parallel derivative only if n>2\n"; //since the function is a parabola
    dg::Gaussian init0(R_0+0.5, 0, 0.2, 0.2, 1);
    dg::GaussianZ modulate(0., M_PI/3., 1);
    t.tic();
    function = ds.fieldaligned().evaluate( init0, modulate, Nz/2, 2);
    t.toc();
    std::cout << "Fieldaligned initialization took "<<t.diff()<<"s\n";
    ds( function, derivative);
    norm = dg::blas2::dot(vol3d, derivative);
    std::cout << "Norm Centered Derivative "<<sqrt( norm)<<" (compare with that of ds_mpib)\n";
    ds.forward( 1., function, 0., derivative);
    norm = dg::blas2::dot(vol3d, derivative);
    std::cout << "Norm Forward  Derivative "<<sqrt( norm)<<" (compare with that of ds_mpib)\n";
    ds.backward( 1., function, 0., derivative);
    norm = dg::blas2::dot(vol3d, derivative);
    std::cout << "Norm Backward Derivative "<<sqrt( norm)<<" (compare with that of ds_mpib)\n";

    return 0;
}