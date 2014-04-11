#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <omp.h>

#include "toefl/toefl.h"
#include "toefl/timer.h"
#include "file/read_input.h"
#include "file/file.h"
//#include "utility.h"
#include "particle_density.h"
#include "dft_dft_solver.h"

#include "energetics.h"
//#include "drt_dft_solver.h"
#include "blueprint.h"

using namespace std;
using namespace toefl;
typedef std::complex<double> Complex;

const unsigned n = 3;
typedef DFT_DFT_Solver<n> Sol;
typedef Sol::Matrix_Type Mat;
    
unsigned itstp; //initialized by init function
unsigned max_out;
double amp, imp_amp; //
double blob_width, posX, posY;
unsigned energy_interval;

void write_probe( const Mat& field, std::vector<double>* total, std::vector<double>* fluct)
{
    unsigned nx = field.cols();
    unsigned ny = field.rows();
    std::vector<double> average(8,0);
    for( unsigned l=0; l<8;l++)
    {
        unsigned posX = nx/16+nx*l/8;
        for( unsigned i=0; i<ny; i++)
            average[l] += field( i, posX);
        average[l] /= ny;
    }
    for( unsigned k=0; k<8; k++)
    {
        unsigned posY = ny/16+ny*k/8;
        for( unsigned l=0; l<8;l++)
        {
            unsigned posX = nx/16+nx*l/8;
            total[k*8+l].push_back(field( posY, posX ));
            fluct[k*8+l].push_back(field( posY, posX ) - average[l]);
        }
    }
}
void write_vx( const Mat& phi, std::vector<double>* v, double h)
{
    unsigned nx = phi.cols();
    unsigned ny = phi.rows();
    for( unsigned k=0; k<8; k++)
    {
        unsigned posY = ny/16+ny*k/8;
        for( unsigned l=0; l<8;l++)
        {
            unsigned posX = nx/16+nx*l/8;
            v[k*8+l].push_back( -(phi( posY +1, posX ) - phi( posY-1, posX))/2./h);//-dy phi
        }
    }
}
void write_vy( const Mat& phi, std::vector<double>* vy, std::vector<double>* vy_fluc, double h)
{
    unsigned nx = phi.cols();
    unsigned ny = phi.rows();
    std::vector<double> average(24,0);
    for( unsigned l=0; l<8;l++)
    {
        unsigned posX = nx/16+nx*l/8;
        for( unsigned j=0; j<ny; j++)
            average[l] += (phi( j, posX+1)-phi(j, posX-1))/2./h; //dx phi
        average[l] /= ny;
    }
    for( unsigned k=0; k<8; k++)
    {
        unsigned posY = ny/16+ny*k/8;
        for( unsigned l=0; l<8;l++)
        {
            unsigned posX = nx/16+nx*l/8;
            double dxphi = (phi( posY, posX+1 )-phi(posY, posX-1))/2./h;
            vy[k*8+l].push_back(   dxphi    );
            vy_fluc[k*8+l].push_back(  dxphi - average[l]);
        }
    }
}


Blueprint read( char const * file)
{
    std::cout << "Reading from "<<file<<"\n";
    std::vector<double> para;
    try{ para = file::read_input( file); }
    catch (Message& m) 
    {  
        m.display(); 
        throw m;
    }
    Blueprint bp( para);
    amp = para[10];
    imp_amp = para[14];
    itstp = para[19];
    omp_set_num_threads( para[20]);
    blob_width = para[21];
    max_out = para[22];
    posX = para[23];
    posY = para[24];
    energy_interval = para[25];
    std::cout<< "With "<<omp_get_max_threads()<<" threads\n";
    return bp;
}

double integral( const Mat& src, double h)
{
    double sum=0;
    for( unsigned i=0; i<src.rows(); i++)
        for( unsigned j=0; j<src.cols(); j++)
            sum+=h*h*src(i,j);
    return sum;
}

void xpa( std::vector<double>& x, double a)
{
    for( unsigned i =0; i<x.size(); i++)
        x[i] += a;
}

int main( int argc, char* argv[])
{
    //Parameter initialisation
    
    Blueprint bp_mod;
    std::vector<double> v;
    std::string input;
    if( argc != 5)
    {
        cerr << "ERROR: Wrong number of arguments!\nUsage: "<< argv[0]<<" [inputfile] [fields.h5] [probe.h5] [energies.dat]\n";
        return -1;
    }
    else 
    {
        bp_mod = read(argv[1]);
        input = file::read_file( argv[1]);
    }
    const Blueprint bp = bp_mod;
    bp.display( );
    if( bp.boundary().bc_x != TL_PERIODIC)
    {
        cerr << "ERROR: Only periodic boundaries allowed!\n";
        return -1;
    }
    if( !bp.isEnabled( TL_IMPURITY) )
    {
        cerr << "ERROR: Only allowed with impurities!\n";
        return -1;
    }
    //construct solvers 
    try{ Sol solver( bp); }catch( Message& m){m.display();}
    Sol solver (bp);
    unsigned rows = bp.algorithmic().ny, cols = bp.algorithmic().nx;
    DFT_DFT dft_dft(rows, cols);
    unsigned crows = rows, ccols = cols/2+1;
    Matrix<Complex > cphi(crows, ccols), cne(cphi), cni( cne), cnz( cne);

    const Algorithmic& alg = bp.algorithmic();
    Mat ne{ alg.ny, alg.nx, 0.}, ni(ne), nz(ne), phi{ ne};
    const Boundary& bound = bp.boundary();
    // init ne and nz! 
    try{
        init_gaussian( ne, posX, posY, blob_width/bound.lx, blob_width/bound.ly, amp);
        init_gaussian_column( nz, posX, blob_width/bound.lx, imp_amp);
        std::array< Mat, n> arr{{ ne, nz, phi}};
        //now set the field to be computed
        solver.init( arr, TL_IONS);
    }catch( Message& m){m.display();}
    double meanMassE = integral( ne, alg.h)/bound.lx/bound.ly;
    std::cout << setprecision(6) <<meanMassE<<std::endl;
    //Mat elec = solver.getField(TL_ELECTRONS);
    //Mat ion = solver.getField(TL_IONS);
    //for( unsigned i=0; i<elec.rows(); i++)
    //    for( unsigned j=0; j<elec.cols(); j++)
    //    {
    //        phi(i,j) = elec(i,j)-ion(i,j);
    //        if( fabs(phi(i,j))>  1e-13)
    //            std::cout <<phi(i,j)<<" ";
    //    }
    //std::cout<< std::endl;

    
    Energetics<n> energetics(bp);

    /////////////////////////////////////////////////////////////////////////
    file::T5trunc t5file( argv[2], input);
    std::vector<double> times;
    std::vector<double> probe_ne[64];
    std::vector<double> probe_ne_fluc[64];
    std::vector<double> probe_ni[64];
    std::vector<double> probe_ni_fluc[64];
    std::vector<double> probe_nz[64];
    std::vector<double> probe_nz_fluc[64];
    std::vector<double> probe_phi[64];
    std::vector<double> probe_phi_fluc[64];
    std::vector<double> probe_vy[64];
    std::vector<double> probe_vy_fluc[64];
    std::vector<double> probe_vx[64];
    std::ofstream  os( argv[4]);
    os << "#Time(1) Ue(2) Ui(3) Uj(4) Ei(5) Ej(6) M(Ei)(7) M(Ej)(8) F_e(9) F_i(10) F_j(11) R_i(12) R_j(13) Diff(14) M(Diff)(15) A(16) J(17)\n";
    os << std::setprecision(14);
    double time = 0.0;
    std::vector<double> probe_array( 64), probe_fluct( 64);
    std::vector<double> average(8,0);
    std::vector<double> out( alg.nx*alg.ny);
    std::vector<double> output[n+1] = {out, out, out, out};
    toefl::Timer t, t2;
    t.tic();
    for( unsigned i=0; i<max_out; i++)
    {
        output[0] = solver.getField( TL_ELECTRONS).copy();
        //xpa( output[0], meanMassE);
        output[1] = solver.getField( TL_IONS).copy();
        output[2] = solver.getField( TL_IMPURITIES).copy();
        output[3] = solver.getField( TL_POTENTIAL).copy();
        t5file.write( output[0], output[1], output[2],output[3], time, alg.nx, alg.ny);
        t2.tic();
        for( unsigned j=0; j<itstp; j++)
        {
            times.push_back(time);
            const Mat& electrons = solver.getField( TL_ELECTRONS);
            write_probe( electrons, probe_ne, probe_ne_fluc);
            const Mat& ions = solver.getField( TL_IONS);
            write_probe( ions, probe_ni, probe_ni_fluc);
            const Mat& imp = solver.getField( TL_IMPURITIES);
            write_probe( imp, probe_nz, probe_nz_fluc);
            const Mat& potential = solver.getField( TL_POTENTIAL);
            write_probe( potential, probe_phi, probe_phi_fluc);
            write_vx( potential, probe_vx, alg.h);
            write_vy( potential, probe_vy, probe_vy_fluc, alg.h);
            if( !(j%energy_interval))
            {
                os << time<<" ";
                std::vector<double> thermal = energetics.thermal_energies( solver.getDensity());
                std::vector<double> exb = energetics.exb_energies( solver.getField(TL_POTENTIAL));
                std::vector<double> gradient_flux = energetics.gradient_flux( solver.getDensity(), solver.getPotential() );
                std::vector<double> diffusion = energetics.diffusion( solver.getDensity(), solver.getPotential() );
                double capital_a = energetics.capital_a( solver.getField( TL_ELECTRONS), solver.getField(TL_POTENTIAL));
                double capital_jot = energetics.capital_jot( solver.getField( TL_ELECTRONS), solver.getField(TL_POTENTIAL));

                for( unsigned k=0; k<thermal.size(); k++)
                    os << thermal[k]<<" ";
                for( unsigned k=0; k<exb.size(); k++)
                    os << exb[k]<<" ";
                for( unsigned k=0; k<gradient_flux.size(); k++)
                    os << gradient_flux[k]<<" ";
                for( unsigned k=0; k<diffusion.size(); k++)
                    os << diffusion[k]<<" ";
                os << capital_a<<" ";
                os << capital_jot;
                os << std::endl;

            }
            if( i==0 && j==0)
                solver.first_step();
            else if( i==0 && j==1)
                solver.second_step();
            else
                solver.step();
            time += alg.dt;
        }
        t2.toc();
        std::cout << "\n\t Time "<<time <<" / "<<alg.dt*itstp*max_out;
        std::cout << "\n\t Average time for one step: "<<t2.diff()/(double)itstp<<"s\n\n"<<std::flush;
    }
    output[0] = solver.getField( TL_ELECTRONS).copy();
    //xpa( output[0], meanMassE);
    output[1] = solver.getField( TL_IONS).copy();
    output[2] = solver.getField( TL_IMPURITIES).copy();
    output[3] = solver.getField( TL_POTENTIAL).copy();
    t2.tic();
    t5file.write( output[0], output[1], output[2], output[3], time, alg.nx, alg.ny);

    times.push_back(time);
    const Mat& electrons = solver.getField( TL_ELECTRONS);
    write_probe( electrons, probe_ne, probe_ne_fluc);
    const Mat& ions = solver.getField( TL_IONS);
    write_probe( ions, probe_ni, probe_ni_fluc);
    const Mat& imp = solver.getField( TL_IMPURITIES);
    write_probe( imp, probe_nz, probe_nz_fluc);
    const Mat& potential = solver.getField( TL_POTENTIAL);
    write_probe( potential, probe_phi, probe_phi_fluc);
    write_vx( potential, probe_vx, alg.h);
    write_vy( potential, probe_vy, probe_vy_fluc, alg.h);
    //
    {
    os << time<<" ";
    std::vector<double> thermal = energetics.thermal_energies( solver.getDensity());
    std::vector<double> exb = energetics.exb_energies( solver.getField(TL_POTENTIAL));
    std::vector<double> gradient_flux = energetics.gradient_flux( solver.getDensity(), solver.getPotential() );
    std::vector<double> diffusion = energetics.diffusion( solver.getDensity(), solver.getPotential() );
    double capital_a = energetics.capital_a( solver.getField( TL_ELECTRONS), solver.getField(TL_POTENTIAL));
    double capital_jot = energetics.capital_jot( solver.getField( TL_ELECTRONS), solver.getField(TL_POTENTIAL));

    for( unsigned k=0; k<thermal.size(); k++)
        os << thermal[k]<<" ";
    for( unsigned k=0; k<exb.size(); k++)
        os << exb[k]<<" ";
    for( unsigned k=0; k<gradient_flux.size(); k++)
        os << gradient_flux[k]<<" ";
    for( unsigned k=0; k<diffusion.size(); k++)
        os << diffusion[k]<<" ";
    os << capital_a<<" ";
    os << capital_jot;
    os << std::endl;
    }

    //write Probe file
    {
        file::Probe probe( argv[3], input, times);
        probe.createGroup("ne");
        for( unsigned i=0; i<8; i++)
            for( unsigned j=0; j<8; j++)
                probe.write( probe_ne[i*8+j], i+1, j+1, max_out*itstp+1);
        probe.closeGroup();
        probe.createGroup("ne_fluc");
        for( unsigned i=0; i<8; i++)
            for( unsigned j=0; j<8; j++)
                probe.write( probe_ne_fluc[i*8+j], i+1, j+1, max_out*itstp+1);
        probe.closeGroup();

        probe.createGroup("ni");
        for( unsigned i=0; i<8; i++)
            for( unsigned j=0; j<8; j++)
                probe.write( probe_ni[i*8+j], i+1, j+1, max_out*itstp+1);
        probe.closeGroup();
        probe.createGroup("ni_fluc");
        for( unsigned i=0; i<8; i++)
            for( unsigned j=0; j<8; j++)
                probe.write( probe_ni_fluc[i*8+j], i+1, j+1, max_out*itstp+1);
        probe.closeGroup();

        probe.createGroup("nz");
        for( unsigned i=0; i<8; i++)
            for( unsigned j=0; j<8; j++)
                probe.write( probe_ni[i*8+j], i+1, j+1, max_out*itstp+1);
        probe.closeGroup();
        probe.createGroup("nz_fluc");
        for( unsigned i=0; i<8; i++)
            for( unsigned j=0; j<8; j++)
                probe.write( probe_ni_fluc[i*8+j], i+1, j+1, max_out*itstp+1);
        probe.closeGroup();

        probe.createGroup("phi");
        for( unsigned i=0; i<8; i++)
            for( unsigned j=0; j<8; j++)
                probe.write( probe_phi[i*8+j], i+1, j+1, max_out*itstp+1);
        probe.closeGroup();
        probe.createGroup("phi_fluc");
        for( unsigned i=0; i<8; i++)
            for( unsigned j=0; j<8; j++)
                probe.write( probe_phi_fluc[i*8+j], i+1, j+1, max_out*itstp+1);
        probe.closeGroup();

        probe.createGroup("vx");
        for( unsigned i=0; i<8; i++)
            for( unsigned j=0; j<8; j++)
                probe.write( probe_vx[i*8+j], i+1, j+1, max_out*itstp+1);
        probe.closeGroup();

        probe.createGroup("vy");
        for( unsigned i=0; i<8; i++)
            for( unsigned j=0; j<8; j++)
                probe.write( probe_phi[i*8+j], i+1, j+1, max_out*itstp+1);
        probe.closeGroup();
        probe.createGroup("vy_fluc");
        for( unsigned i=0; i<8; i++)
            for( unsigned j=0; j<8; j++)
                probe.write( probe_phi_fluc[i*8+j], i+1, j+1, max_out*itstp+1);
        probe.closeGroup();
    }

    t2.toc();
    std::cout << "Probe took "<<t2.diff()<< "s\n";
    //////////////////////////////////////////////////////////////////
    os.close();
    fftw_cleanup();
    t.toc();
    std::cout << "Total simulation time for "<<max_out*itstp<<" steps: "<<t.diff()<<"s\n";
    std::cout << "Which is "<<t.diff()/(double)(max_out*itstp)<<"s/step\n";
    //std::cout << "Times size: "<<times.size()<<"\n";
    //std::cout << "Probes size: "<<probe_ne[0].size()<<"\n";
    //for( unsigned i=0; i<probe_ne[16].size(); i++)
    //    std::cout << probe_ne[16][i]<<"\n";
    return 0;

}