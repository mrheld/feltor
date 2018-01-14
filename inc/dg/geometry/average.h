#pragma once

#include "grid.h"
#include "weights.cuh"
#include "dg/blas1.h"

/*! @file 
  @brief contains classes for poloidal and toroidal average computations.
  */
namespace dg{

/**
 * @brief MPI specialized class for y average computations
 *
 * @snippet backend/average_mpit.cu doxygen
 * @ingroup utilities
 */
template< class container>
struct Average
{
    Average( const aTopology2d& g, enum Coordinate direction): m_dir(direction), m_dim3(false)
    {
        m_nx = g.Nx()*g.n(), m_ny = g.Ny()*g.n();
        m_w=dg::transfer<container>(dg::detail::create_weights(g, diretion));
        if( direction == dg::x)
            dg::blas1::scal( m_w, 1./g.lx());
        else if ( direction == dg::y)
            dg::blas1::scal( m_w, 1./g.ly());
        else
            std::cerr "Warning: attempting to average wrong direction in 2d\n";
        m_temp1d = m_temp = m_w;
    }

    Average( const aTopology3d& g, enum Coordinate direction): m_dir(direction), m_dim3(true)
    {
        m_w = dg::transfer<container>(dg::detail::create_weights(g, diretion));
        m_temp2 = m_temp = m_w;
        unsigned nx = g.n()*g.Nx(), ny = g.n()*g.Ny(), nz = g.Nz();
        if( direction == dg::x) {
            dg::blas1::scal( m_w, 1./g.lx());
            m_nx = nx, m_ny = ny*nz;
        }
        else if( direction == dg::z) {
            dg::blas1::scal( m_w, 1./g.lz());
            m_nx = nx*ny, m_ny = nz;
        }
        else if( direction == dg::xy) {
            dg::blas1::scal( m_w, 1./g.lx()/g.ly());
            m_nx = nx*ny, m_ny = nz;
        }
        else if( direction == dg::yz) {
            dg::blas1::scal( m_w, 1./g.ly()/g.lz());
            m_nx = nx, m_ny = ny*nz;
        }
        else 
            std::cerr << "Warning: this direction is not implemented\n";
    }
    /**
     * @brief Compute the average 
     *
     * @param src 2D Source Vector (must have the same size as the grid given in the constructor)
     * @param res 2D result Vector (may alias src), every line contains the x-dependent average over
     the y-direction of src 
     */
    void operator() (const container& src, container& res)
    {
        if( m_dir == dg::x || m_dir == dg::xy)
        {
            dg::average( m_nx, m_ny, m_src, m_w2d, m_temp);
            dg::extend_column( m_nx, m_ny, m_temp, res);
        }
        else if( (m_dir == dg::y && m_dim3 == false) || m_dir == dg::z || m_dir == dg::yz)
        {
            dg::transpose( m_nx, m_ny, src, m_temp);
            dg::average( m_ny, m_nx, m_temp, m_w2d, m_temp1d);
            dg::extend_line( m_nx, m_ny, m_temp1d, res);
        }
        else if( m_dir == dg::y && m_dim3 == true) 
            std::cerr << "Warning: average y direction in 3d is not implemented\n";
        else
            std::cerr << "Warning: average direction is not implemented\n";

    }
  private:
    unsigned m_nx, m_ny;
    container m_w, m_temp, m_temp2; 
    enum Coordinate m_dir;
    bool m_dim3;

};


}//namespace dg
