#include <iostream>

#include <cusp/print.h>

#include "dg/backend/functions.h"
#include "dg/backend/timer.cuh"
#include "dg/blas.h"
#include "dg/functors.h"
#include "dg/geometry/geometry.h"
#include "testfunctors.h"
#include "ds.h"
#include "solovev.h"
#include "flux.h"
#include "toroidal.h"


int main(int argc, char * argv[])
{
    std::cout << "Start DS test on flux grid!"<<std::endl;
    Json::Reader reader;
    Json::Value js;
    if( argc==1) {
        std::ifstream is("geometry_params_Xpoint.js");
        reader.parse(is,js,false);
    }
    else {
        std::ifstream is(argv[1]);
        reader.parse(is,js,false);
    }
    dg::geo::solovev::Parameters gp(js);
    dg::geo::TokamakMagneticField mag = dg::geo::createSolovevField( gp);
    dg::Timer t;
    t.tic();
    std::cout << "Type n(3), Nx(8), Ny(80), Nz(20)\n";
    unsigned n,Nx,Ny,Nz;
    std::cin >> n>> Nx>>Ny>>Nz;   
    std::cout << "Type multipleX (1) and multipleY (10)!\n";
    unsigned mx, my;
    std::cin >> mx >> my;

    double psi_0 = -20, psi_1 = -4;
    dg::geo::FluxGenerator flux( mag.get_psip(), mag.get_ipol(), psi_0, psi_1, gp.R_0, 0., 1);
    dg::geo::CurvilinearProductGrid3d g3d(flux, n, Nx, Ny,Nz, dg::DIR);
    //dg::geo::Fieldaligned<dg::aGeometry3d, dg::IHMatrix, dg::HVec> fieldaligned( bhat, g3d, 1, 4, gp.rk4eps, dg::NoLimiter() ); 
    dg::geo::DS<dg::aProductGeometry3d, dg::IHMatrix, dg::HMatrix, dg::HVec> ds( mag, g3d, mx, my, false, true, 1e-8, dg::normed, dg::centered);
    
    t.toc();
    std::cout << "Construction took "<<t.diff()<<"s\n";
    dg::HVec B = dg::pullback( dg::geo::InvB(mag), g3d), divB(B);
    dg::HVec lnB = dg::pullback( dg::geo::LnB(mag), g3d), gradB(B);
    dg::HVec gradLnB = dg::pullback( dg::geo::GradLnB(mag), g3d);
    dg::HVec ones3d = dg::evaluate( dg::one, g3d);
    dg::HVec vol3d = dg::create::volume( g3d);
    dg::blas1::pointwiseDivide( ones3d, B, B);
    dg::HVec function = dg::pullback( dg::geo::FuncNeu(mag), g3d), derivative(function);
    ds( function, derivative);

    ds.centeredAdj( 1., B, 0., divB);
    double norm =  sqrt( dg::blas2::dot(divB, vol3d, divB));
    std::cout << "Divergence of B is "<<norm<<"\n";

    ds.centered( 1., lnB, 0., gradB);
    std::cout << "num. norm of gradLnB is "<<sqrt( dg::blas2::dot( gradB,vol3d, gradB))<<"\n";
    norm = sqrt( dg::blas2::dot( gradLnB, vol3d, gradLnB) );
    std::cout << "ana. norm of gradLnB is "<<norm<<"\n";
    dg::blas1::axpby( 1., gradB, -1., gradLnB, gradLnB);
    double norm2 = sqrt(dg::blas2::dot(gradLnB, vol3d, gradLnB));
    std::cout << "rel. error of lnB is    "<<norm2/norm<<"\n";
    
    return 0;
}