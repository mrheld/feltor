#pragma once

#include "gridX.h"
#include "dx.h"

/*! @file 
  
  Convenience functions to create 2D derivatives
  */
namespace dg{

template<class Matrix>
struct Composite
{
    template<class Matrix2>
    Composite( const Composite<Matrix2>& src):m1(src.m1), m2(src.m2), dual(src.dual){}
    Composite( const Matrix& m):m1(m), m2(m), dual(false){ }
    Composite( const Matrix& m1, const Matrix& m2):m1(m1), m2(m2), dual(true){ }
    template<class Matrix2>
    Composite& operator=( const Composite<Matrix2>& src){ Composite c(src); 
        *this = c; return *this;}
    Matrix m1, m2;
    bool dual;
    template< class container>
    void symv( const  container& v1, container& v2)
    {
        m1.symv( v1, v2);
        if( dual)
            m2.symv( v1, v2);
    }
    void display( std::ostream& os = std::cout) const
    {
        if( dual)
        {
            os << " dual matrix: \n";
            os << " INNER MATRIX\n";
            m1.display( os);
            os << " OUTER MATRIX\n";
            m2.display( os);
        }
        else
        {
            os << "single matrix: \n";
            m1.display(os);
        }
    }
};
///@cond
template <class Matrix>
struct MatrixTraits<Composite<Matrix> >
{
    typedef double value_type;
    typedef SelfMadeMatrixTag matrix_category;
};
template <class Matrix>
struct MatrixTraits<const Composite<Matrix> >
{
    typedef double value_type;
    typedef SelfMadeMatrixTag matrix_category;
};
///@endcond


/**
 * @brief Contains functions used for matrix creation
 */
namespace create{

///@addtogroup highlevel
///@{

//dx, dy, jumpX, jumpY

/**
 * @brief Create 2d derivative in x-direction
 *
 * @param g The grid on which to create dx
 * @param bcx The boundary condition
 * @param dir The direction of the first derivative
 *
 * @return A host matrix 
 */
Composite<EllSparseBlockMat> dx( const GridX2d& g, bc bcx, direction dir = centered)
{
    EllSparseBlockMat dx;
    dx = dx_normed( g.n(), g.Nx(), g.hx(), bcx, dir);
    dx.left = g.n()*g.Ny();
    dx.set_defaults_left_right();
    return dx;
}

/**
 * @brief Create 2d derivative in x-direction
 *
 * @param g The grid on which to create dx (boundary condition is taken from here)
 * @param dir The direction of the first derivative
 *
 * @return A host matrix
 */
Composite<EllSparseBlockMat> dx( const GridX2d& g, direction dir = centered) { return dx( g, g.bcx(), dir);}

/**
 * @brief Create 2d derivative in y-direction
 *
 * @param g The grid on which to create dy
 * @param bcy The boundary condition
 * @param dir The direction of the first derivative
 *
 * @return A host matrix
 */
Composite<EllSparseBlockMat> dy( const GridX2d& g, bc bcy, direction dir = centered)
{
    EllSparseBlockMat dy_inner, dy_outer;
    GridX1d g1d_inner( g.y0(), g.y1(), g.fy(), g.n(), g.Ny(), bcy);
    Grid1d<double> g1d_outer( g.y0(), g.y1(), g.n(), g.Ny(), bcy);
    dy_inner = dx( g1d_inner, bcy, dir);
    dy_outer = dx( g1d_outer, bcy, dir);
    dy_inner.right = g.n()*g.Nx();
    dy_inner.right_[0] = 0;
    dy_inner.right_[1] = g.n()*g.inner_Nx();
    dy_outer.right_[0] = g.n()*g.inner_Nx();
    dy_outer.right_[1] = g.n()*g.Nx();
    dy_outer.right = g.n()*g.Nx();

    Composite<EllSparseBlockMat> c( dy_inner, dy_outer);
    return c;
}

/**
 * @brief Create 2d derivative in y-direction
 *
 * @param g The grid on which to create dy (boundary condition is taken from here)
 * @param dir The direction of the first derivative
 *
 * @return A host matrix 
 */
Composite<EllSparseBlockMat> dy( const GridX2d& g, direction dir = centered){ return dy( g, g.bcy(), dir);}

/**
 * @brief Matrix that contains 2d jump terms in X direction
 *
 * @param g grid
 * @param bcx boundary condition in x
 *
 * @return A host matrix 
 */
Composite<EllSparseBlockMat> jumpX( const GridX2d& g, bc bcx)
{
    EllSparseBlockMat jx;
    jx = jump( g.n(), g.Nx(), g.hx(), bcx);
    jx.left = g.n()*g.Ny();
    jx.set_defaults_left_right();
    return jx;
}

/**
 * @brief Matrix that contains 2d jump terms in Y direction
 *
 * @param g grid
 * @param bcy boundary condition in y
 *
 * @return A host matrix 
 */
Composite<EllSparseBlockMat> jumpY( const GridX2d& g, bc bcy)
{
    EllSparseBlockMat jy_inner, jy_outer;
    GridX1d g1d_inner( g.y0(), g.y1(), g.fy(), g.n(), g.Ny(), bcy);
    Grid1d<double> g1d_outer( g.y0(), g.y1(), g.n(), g.Ny(), bcy);
    jy_inner = jump( g1d_inner, bcy);
    jy_outer = jump( g1d_outer, bcy);
    jy_inner.right = g.n()*g.Nx();
    jy_inner.right_[0] = 0;
    jy_inner.right_[1] = g.n()*g.inner_Nx();
    jy_outer.right_[0] = g.n()*g.inner_Nx();
    jy_outer.right_[1] = g.n()*g.Nx();
    jy_outer.right = g.n()*g.Nx();

    Composite<EllSparseBlockMat> c( jy_inner, jy_outer);
    return c;
}

/**
 * @brief Matrix that contains 2d jump terms in X direction taking boundary conditions from the grid
 *
 * @param g grid
 *
 * @return A host matrix 
 */
Composite<EllSparseBlockMat> jumpX( const GridX2d& g)
{
    return jumpX( g, g.bcx());
}

/**
 * @brief Matrix that contains 2d jump terms in Y direction taking boundary conditions from the grid
 *
 * @param g grid
 *
 * @return A host matrix 
 */
Composite<EllSparseBlockMat> jumpY( const GridX2d& g)
{
    return jumpY( g, g.bcy());
}

///////////////////////////////////////////3D VERSIONS//////////////////////
//jumpX, jumpY, jumpZ, dx, dy, dz
/**
 * @brief Matrix that contains jump terms in X direction in 3D
 *
 * @param g The 3D grid
 * @param bcx boundary condition in x
 *
 * @return A host matrix 
 */
Composite<EllSparseBlockMat> jumpX( const GridX3d& g, bc bcx)
{
    EllSparseBlockMat jx;
    jx = jump( g.n(), g.Nx(), g.hx(), bcx);
    jx.left = g.n()*g.Ny()*g.Nz();
    jx.set_defaults_left_right();
    return jx;
}

/**
 * @brief Matrix that contains jump terms in Y direction in 3D
 *
 * @param g The 3D grid
 * @param bcy boundary condition in y
 *
 * @return A host matrix 
 */
Composite<EllSparseBlockMat> jumpY( const GridX3d& g, bc bcy)
{
    EllSparseBlockMat jy_inner, jy_outer;
    GridX1d g1d_inner( g.y0(), g.y1(), g.fy(), g.n(), g.Ny(), bcy);
    Grid1d<double> g1d_outer( g.y0(), g.y1(), g.n(), g.Ny(), bcy);
    jy_inner = jump( g1d_inner, bcy);
    jy_outer = jump( g1d_outer, bcy);
    jy_inner.right = g.n()*g.Nx();
    jy_inner.right_[0] = 0;
    jy_inner.right_[1] = g.n()*g.inner_Nx();
    jy_outer.right_[0] = g.n()*g.inner_Nx();
    jy_outer.right_[1] = g.n()*g.Nx();
    jy_outer.right = g.n()*g.Nx();
    jy_inner.left = jy_inner.left_[1] = g.Nz();
    jy_outer.left = jy_outer.left_[1] = g.Nz();

    Composite<EllSparseBlockMat> c( jy_inner, jy_outer);
    return c;
}

/**
 * @brief Matrix that contains jump terms in Z direction in 3D
 *
 * @param g The 3D grid
 * @param bcz boundary condition in z
 *
 * @return A host matrix 
 */
Composite<EllSparseBlockMat> jumpZ( const GridX3d& g, bc bcz)
{
    EllSparseBlockMat jz;
    jz = jump( 1, g.Nz(), g.hz(), bcz);
    jz.right = g.n()*g.Nx()*g.n()*g.Ny();
    jz.set_defaults_left_right();
    return jz;
}

/**
 * @brief Matrix that contains 3d jump terms in X direction taking boundary conditions from the grid
 *
 * @param g grid
 *
 * @return A host matrix
 */
Composite<EllSparseBlockMat> jumpX( const GridX3d& g)
{
    return jumpX( g, g.bcx());
}

/**
 * @brief Matrix that contains 3d jump terms in Y direction taking boundary conditions from the grid
 *
 * @param g grid
 *
 * @return A host matrix
 */
Composite<EllSparseBlockMat> jumpY( const GridX3d& g)
{
    return jumpY( g, g.bcy());
}

/**
 * @brief Matrix that contains 3d jump terms in Z direction taking boundary conditions from the grid
 *
 * @param g grid
 *
 * @return A host matrix
 */
Composite<EllSparseBlockMat> jumpZ( const GridX3d& g)
{
    return jumpZ( g, g.bcz());
}


/**
 * @brief Create 3d derivative in x-direction
 *
 * @param g The grid on which to create dx
 * @param bcx The boundary condition
 * @param dir The direction of the first derivative
 *
 * @return A host matrix 
 */
Composite<EllSparseBlockMat> dx( const GridX3d& g, bc bcx, direction dir = centered)
{
    EllSparseBlockMat dx;
    dx = dx_normed( g.n(), g.Nx(), g.hx(), bcx, dir);
    dx.left = g.n()*g.Ny()*g.Nz();
    dx.set_defaults_left_right();
    return dx;
}

/**
 * @brief Create 3d derivative in x-direction
 *
 * @param g The grid on which to create dx (boundary condition is taken from here)
 * @param dir The direction of the first derivative
 *
 * @return A host matrix 
 */
Composite<EllSparseBlockMat> dx( const GridX3d& g, direction dir = centered) { return dx( g, g.bcx(), dir);}

/**
 * @brief Create 3d derivative in y-direction
 *
 * @param g The grid on which to create dy
 * @param bcy The boundary condition
 * @param dir The direction of the first derivative
 *
 * @return A host matrix 
 */
Composite<EllSparseBlockMat> dy( const GridX3d& g, bc bcy, direction dir = centered)
{
    EllSparseBlockMat dy_inner, dy_outer;
    GridX1d g1d_inner( g.y0(), g.y1(), g.fy(), g.n(), g.Ny(), bcy);
    Grid1d<double> g1d_outer( g.y0(), g.y1(), g.n(), g.Ny(), bcy);
    dy_inner = dx( g1d_inner, bcy, dir);
    dy_outer = dx( g1d_outer, bcy, dir);
    dy_inner.right = g.n()*g.Nx();
    dy_inner.right_[0] = 0;
    dy_inner.right_[1] = g.n()*g.inner_Nx();
    dy_outer.right_[0] = g.n()*g.inner_Nx();
    dy_outer.right_[1] = g.n()*g.Nx();
    dy_outer.right = g.n()*g.Nx();
    dy_inner.left = dy_inner.left_[1] = g.Nz();
    dy_outer.left = dy_outer.left_[1] = g.Nz();

    Composite<EllSparseBlockMat> c( dy_inner, dy_outer);
    return c;
}

/**
 * @brief Create 3d derivative in y-direction
 *
 * @param g The grid on which to create dy (boundary condition is taken from here)
 * @param dir The direction of the first derivative
 *
 * @return A host matrix 
 */
Composite<EllSparseBlockMat> dy( const GridX3d& g, direction dir = centered){ return dy( g, g.bcy(), dir);}

/**
 * @brief Create 3d derivative in z-direction
 *
 * @param g The grid on which to create dz
 * @param bcz The boundary condition
 * @param dir The direction of the stencil
 *
 * @return A host matrix 
 */
Composite<EllSparseBlockMat> dz( const GridX3d& g, bc bcz, direction dir = centered)
{
    EllSparseBlockMat dz;
    dz = dx_normed( 1, g.Nz(), g.hz(), bcz, dir);
    dz.right = g.n()*g.n()*g.Nx()*g.Ny();
    dz.set_defaults_left_right();
    return dz;

}

/**
 * @brief Create 3d derivative in z-direction
 *
 * @param g The grid on which to create dz (boundary condition is taken from here)
 * @param dir The direction of the stencil
 *
 * @return A host matrix 
 */
Composite<EllSparseBlockMat> dz( const GridX3d& g, direction dir = centered){ return dz( g, g.bcz(), dir);}



///@}

} //namespace create

} //namespace dg

