//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file disk.cpp
//  \brief Initializes stratified Keplerian accretion disk in both cylindrical and
//  spherical polar coordinates.  Initial conditions are in vertical hydrostatic eqm.

// C headers

// C++ headers
#include <algorithm>  // min
#include <cmath>      // sqrt
#include <cstdlib>    // srand
#include <cstring>    // strcmp()
#include <fstream>
#include <iostream>   // endl
#include <limits>
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <complex>
#include <chrono>
using namespace std::chrono;

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../hydro/hydro_diffusion/hydro_diffusion.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../orbital_advection/orbital_advection.hpp"

void GetCylCoord(Coordinates *pco,Real &rad,Real &phi,Real &z,int i,int j,int k);
Real DenProfileCyl(const Real rad, const Real phi, const Real z);
Real PoverR(const Real rad, const Real phi, const Real z);
//Real PreessProfile(const Real rad, const Real phi, const Real z);
void VelProfileCyl(const Real rad, const Real phi, const Real z,
                   Real &v1, Real &v2, Real &v3);
Real RotOrbitalVelocity(OrbitalAdvection *porb, Real x_, Real y_, Real z_);
Real RotOrbitalVelocity_r(OrbitalAdvection *porb, Real x_, Real y_, Real z_);
Real RotOrbitalVelocity_t(OrbitalAdvection *porb, Real x_, Real y_, Real z_);
void SubtractionOrbitalVelocity(OrbitalAdvection *porb, Coordinates *pco,
                                Real &v1, Real &v2, Real &v3, int i, int j, int k);

void AlphaVis(HydroDiffusion *phdif, MeshBlock *pmb,const  AthenaArray<Real> &w, const AthenaArray<Real> &bc,int is, int ie, int js, int je, int ks, int ke);
Real compute_newton_raphson(int N_it, Real l, Real e);
// problem parameters which are useful to make global to this file
static Real gm0, r0, rho0, dslope, p0_over_r0, pslope, gamma_gas, omegarot, Omega0, sound_speed, luminosity, r_s;
static Real alpha, alphaa, aslope;
static int visflag;
static Real tcool;
static Real dfloor;
static Real rindamp, routdamp;
static Real insert_time;
static Real gauss_width, a_bin, eccent, nmax, mass_sink, f;
// static Real f_acc_x, f_acc_y, f_acc_z; // need these as global variables in order to access them in BinaryForce
//Real voluume;
//----------------------------------------
// class for planetary system including mass, position, velocity

class BinarySystem
{
public:
  int np;
  int ind;
  Real rsoft2;
  double *mass;
  double *massset;
  double *xp, *yp, *zp, *vp;
  double *f_acc_x, *f_acc_y, *f_acc_z;         // position in Cartesian coord.
  double *f_acc_x_user, *f_acc_y_user, *f_acc_z_user;
  BinarySystem(int np);
  ~BinarySystem();
public:
  void orbit(double dt);      // circular planetary orbit
  void Position(double dt, int obj); //position of each object; for a binary obj is 0 or 1
  void binary_force(Real x1obj, Real y1obj, Real z1obj,Real x2obj, Real y2obj, Real z2obj, Real* fx, Real* fy, Real* fz);
};

//------------------------------------------
// constructor for planetary system for np planets

BinarySystem::BinarySystem(int np0)
{
  np   = np0;
  ind  = 1;
  rsoft2 = 0.0;
  mass = new double[np];
  massset = new double[np];
  xp   = new double[np];
  yp   = new double[np];
  zp   = new double[np];
  vp   = new double[np]; //orbital velocity of an object in a binary
  f_acc_x = new double[np];
  f_acc_y = new double[np];
  f_acc_z = new double[np];
  f_acc_x_user = new double[np];
  f_acc_y_user = new double[np];
  f_acc_z_user = new double[np];
};

//---------------------------------------------
// destructor for planetary system

BinarySystem::~BinarySystem()
{
  delete[] mass;
  delete[] massset;
  delete[] xp;
  delete[] yp;
  delete[] zp;
  delete[] vp;
  delete[] f_acc_x;
  delete[] f_acc_y;
  delete[] f_acc_z;
  delete[] f_acc_x_user;
  delete[] f_acc_y_user;
  delete[] f_acc_z_user;
};


// Planet Potential
Real grav_pot_car_btoa(const Real xca, const Real yca, const Real zca,
        const Real xcb, const Real ycb, const Real zcb, const Real gb, const Real rsoft2);
Real grav_pot_car_ind(const Real xca, const Real yca, const Real zca,
        const Real xpp, const Real ypp, const Real zpp, const Real gmp);
// Force on the planet
Real BinaryForce(MeshBlock *pmb, int iout);
Real eccentricity(double dt);
Real sink_fraction(Real d_over_rs);

// planetary system
static BinarySystem *psys;

/*void AllSourceTerms(MeshBlock *pmb, const Real time, const Real dt,
        const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);*/

void AllSourceTerms(MeshBlock *pmb, const Real time, const Real dt,
        const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons, AthenaArray<Real> &cons_scalar);

void Cooling(MeshBlock *pmb, const Real time, const Real dt,
        const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);

// Functions for Planetary Source terms
void BinarySourceTerms(MeshBlock *pmb, const Real time, const Real dt,
        const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);

// User-defined boundary conditions for disk simulations

void DiskInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // Get parameters for gravitatonal potential of central point mass
  gm0 = pin->GetOrAddReal("problem","GM",0.0);
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    gm0 = pin->GetOrAddReal("problem","GMS",0.0);
  }
  r0 = pin->GetOrAddReal("problem","r0",1.0);
  omegarot = pin->GetOrAddReal("problem","omegarot",0.0);
  // Omega0 = pin->GetOrAddReal("problem","Omega0",0.0); 
  Omega0 = pin->GetOrAddReal("orbital_advection","Omega0",0.0);// - commented by Sabina

  if(Omega0!=0.0&&omegarot!=0.0){
    std::stringstream msg;
    msg << "omegarot and Omega0 cannot be non-zero at the same tiime"<<std::endl;
    ATHENA_ERROR(msg);
  }

  // Get parameters for initial density and velocity
  rho0 = pin->GetReal("problem","rho0");
  dslope = pin->GetOrAddReal("problem","dslope",0.0);

  // viscosity
  alpha = pin->GetOrAddReal("problem","nu_iso",0.0);
  alphaa = pin->GetOrAddReal("problem","nu_aniso",0.0);
  aslope = pin->GetOrAddReal("problem","aslope",0.0); 
  visflag = pin->GetOrAddInteger("problem","visflag",0);

  // initial Gaussian
  gauss_width =  pin->GetOrAddReal("problem","gauss_width",0.0);

  // Get parameters of initial pressure and cooling parameters
  if (NON_BAROTROPIC_EOS) {
    p0_over_r0 = pin->GetOrAddReal("problem","p0_over_r0",0.0025);
    pslope = pin->GetOrAddReal("problem","pslope",0.0);
    gamma_gas = pin->GetReal("hydro","gamma");
    tcool = pin->GetOrAddReal("problem","tcool",0.0);
    sound_speed = pin->GetReal("hydro","iso_sound_speed"); //added by Sabina
    //std::cout<<"Sabina, IT IS NON-BAROTROPIC"<<std::endl;  // Sabina needs to print
  } else {
    p0_over_r0=SQR(pin->GetReal("hydro","iso_sound_speed"));
    pslope = 0.0;
    std::cout << "Sabina, it is BAROTROPIC"<<std::endl;
  }
  Real float_min = std::numeric_limits<float>::min();
  dfloor=pin->GetOrAddReal("hydro","dfloor",(1024*(float_min)));

  // Get boundary condition parameters
  rindamp = pin->GetOrAddReal("problem","rindamp",0.0);
  routdamp = pin->GetOrAddReal("problem","routdamp",HUGE_NUMBER);

  insert_time= pin->GetOrAddReal("problem","insert_time",5.0);
  luminosity = pin->GetOrAddReal("problem", "luminosity", 0.0); // luminosity 
  r_s = pin->GetOrAddReal("problem", "r_s", 0.0); // sink radius
  // set up the planetary system
  Real np = pin->GetOrAddInteger("planets","np",0);
  psys = new BinarySystem(np);
  psys->ind = pin->GetOrAddInteger("planets","ind",1);
  psys->rsoft2 = pin->GetOrAddReal("planets","rsoft2",0.0);
  a_bin = pin->GetOrAddReal("planets","a_bin",0.05);
  eccent = pin->GetOrAddReal("planets","eccent",0.0);
  //nmax = pin->GetOrAddReal("planets","nmax",1.0); //number of orbits

  // set initial planet properties
  for(int ip=0; ip<psys->np; ++ip){
    char pname[10];
    sprintf(pname,"mass%d",ip);
    // psys->massset[ip]=pin->GetOrAddReal("planets",pname,0.0);
    // psys->mass[ip]=0.0;
    psys->mass[ip]=pin->GetOrAddReal("planets",pname,0.0);
    sprintf(pname,"x%d",ip);
    psys->xp[ip]=pin->GetOrAddReal("planets",pname,0.0);
    sprintf(pname,"y%d",ip);
    psys->yp[ip]=pin->GetOrAddReal("planets",pname,0.0);
    sprintf(pname,"z%d",ip);
    psys->zp[ip]=pin->GetOrAddReal("planets",pname,0.0);
    sprintf(pname,"v%d",ip);
    psys->vp[ip]=pin->GetOrAddReal("planets",pname,0.0);
    /*std::cout<<pname<<std::endl;
    std::cout<<"ip is "<<ip<<std::endl;
    std::cout<<"xp[ip] is "<<psys->xp[ip]<<std::endl;
    std::cout<<"yp[ip] is "<<psys->yp[ip]<<std::endl;
    std::cout<<"zp[ip] is "<<psys->zp[ip]<<std::endl;*/
  }

  // Enroll Fargo function
  if(omegarot!=0){
    EnrollOrbitalVelocity(RotOrbitalVelocity);
    EnrollOrbitalVelocityDerivative(0, RotOrbitalVelocity_r);
    EnrollOrbitalVelocityDerivative(1, RotOrbitalVelocity_t);
  }

  // enroll planetary potential
  EnrollUserExplicitSourceFunction(AllSourceTerms);
  AllocateUserHistoryOutput(47);
  EnrollUserHistoryOutput(0, BinaryForce, "fr1");
  EnrollUserHistoryOutput(1, BinaryForce, "ft1");
  EnrollUserHistoryOutput(2, BinaryForce, "fp1");
  EnrollUserHistoryOutput(3, BinaryForce, "fr2");
  EnrollUserHistoryOutput(4, BinaryForce, "ft2");
  EnrollUserHistoryOutput(5, BinaryForce, "fp2");
  EnrollUserHistoryOutput(6, BinaryForce, "fxpp1");
  EnrollUserHistoryOutput(7, BinaryForce, "fypp1");
  EnrollUserHistoryOutput(8, BinaryForce, "fzpp1");
  EnrollUserHistoryOutput(9, BinaryForce, "fxpp2");
  EnrollUserHistoryOutput(10, BinaryForce, "fypp2");
  EnrollUserHistoryOutput(11, BinaryForce, "fzpp2");
  EnrollUserHistoryOutput(12, BinaryForce, "torque1");
  EnrollUserHistoryOutput(13, BinaryForce, "torque2");
  EnrollUserHistoryOutput(14, BinaryForce, "xpp1");
  EnrollUserHistoryOutput(15, BinaryForce, "ypp1");
  EnrollUserHistoryOutput(16, BinaryForce, "zpp1");
  EnrollUserHistoryOutput(17, BinaryForce, "xpp2");
  EnrollUserHistoryOutput(18, BinaryForce, "ypp2");
  EnrollUserHistoryOutput(19, BinaryForce, "zpp2");
  EnrollUserHistoryOutput(20, BinaryForce, "rpp1");
  EnrollUserHistoryOutput(21, BinaryForce, "tpp1");
  EnrollUserHistoryOutput(22, BinaryForce, "ppp1");
  EnrollUserHistoryOutput(23, BinaryForce, "rpp2");
  EnrollUserHistoryOutput(24, BinaryForce, "tpp2");
  EnrollUserHistoryOutput(25, BinaryForce, "ppp2");
  EnrollUserHistoryOutput(26, BinaryForce, "mp1");
  EnrollUserHistoryOutput(27, BinaryForce, "mp2");
  EnrollUserHistoryOutput(28, BinaryForce, "Eb1");
  EnrollUserHistoryOutput(29, BinaryForce, "Eb2");
  EnrollUserHistoryOutput(30, BinaryForce, "vol");
  EnrollUserHistoryOutput(31, BinaryForce, "vp1");
  EnrollUserHistoryOutput(32, BinaryForce, "vp2");
  EnrollUserHistoryOutput(33, BinaryForce, "mass_sink1");
  EnrollUserHistoryOutput(34, BinaryForce, "f_acc_x1");
  EnrollUserHistoryOutput(35, BinaryForce, "f_acc_x2");
  EnrollUserHistoryOutput(36, BinaryForce, "f_acc_y1");
  EnrollUserHistoryOutput(37, BinaryForce, "f_acc_y2");
  EnrollUserHistoryOutput(38, BinaryForce, "f_acc_z1");
  EnrollUserHistoryOutput(39, BinaryForce, "f_acc_z2");
  EnrollUserHistoryOutput(40, BinaryForce, "f_grav_x1");
  EnrollUserHistoryOutput(41, BinaryForce, "f_grav_x2");
  EnrollUserHistoryOutput(42, BinaryForce, "f_grav_y1");
  EnrollUserHistoryOutput(43, BinaryForce, "f_grav_y2");
  EnrollUserHistoryOutput(44, BinaryForce, "f_grav_z1");
  EnrollUserHistoryOutput(45, BinaryForce, "f_grav_z2");
  EnrollUserHistoryOutput(46, BinaryForce, "sink_frac");

  EnrollViscosityCoefficient(AlphaVis);

  // enroll user-defined boundary condition
  if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, DiskInnerX1);
  }
  if (mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x1, DiskOuterX1);
  }
  if (mesh_bcs[BoundaryFace::inner_x2] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x2, DiskInnerX2);
  }
  if (mesh_bcs[BoundaryFace::outer_x2] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x2, DiskOuterX2);
  }
  if (mesh_bcs[BoundaryFace::inner_x3] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x3, DiskInnerX3);
  }
  if (mesh_bcs[BoundaryFace::outer_x3] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x3, DiskOuterX3);
  }

  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Initializes Keplerian accretion disk.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  Real rad(0.0), phi(0.0), z(0.0);
  Real v1(0.0), v2(0.0), v3(0.0);
  //std::cout << "Sabina, haven't reached if yet"<<std::endl; // Sabina needs to print
  //  Initialize density and momenta
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        GetCylCoord(pcoord,rad,phi,z,i,j,k); // convert to cylindrical coordinates
        // compute initial conditions in cylindrical coordinates
        phydro->u(IDN,k,j,i) = DenProfileCyl(rad,phi,z);
        VelProfileCyl(rad,phi,z,v1,v2,v3);
        
        if(porb->orbital_advection_defined) {
          SubtractionOrbitalVelocity(porb,pcoord,v1,v2,v3,i,j,k);
          // std::cout << "Sabina, ORBITAL ADVECTION IS DEFINED"<<std::endl; // Sabina needs to print
        }

        phydro->u(IM1,k,j,i) = phydro->u(IDN,k,j,i)*v1;
        phydro->u(IM2,k,j,i) = phydro->u(IDN,k,j,i)*v2;
        phydro->u(IM3,k,j,i) = phydro->u(IDN,k,j,i)*v3;
        if (NON_BAROTROPIC_EOS) {
          Real p_over_r = PoverR(rad,phi,z);
          //Real p_over_r = sound_speed*sound_speed;
          phydro->u(IEN,k,j,i) = p_over_r*phydro->u(IDN,k,j,i)/(gamma_gas - 1.0);
          phydro->u(IEN,k,j,i) += 0.5*(SQR(phydro->u(IM1,k,j,i))+SQR(phydro->u(IM2,k,j,i))
                                       + SQR(phydro->u(IM3,k,j,i)))/phydro->u(IDN,k,j,i);
          /*std::cout << "the energy is: " <<phydro->u(IEN,k,PI,i) <<std::endl; // Sabina needs to print
          std::cout << "the density is: " <<phydro->u(IDN,k,PI,i) <<std::endl; // Sabina needs to print
          std::cout << "the adiabatic pressure is: " <<phydro->u(IEN,k,PI,i)*(gamma_gas - 1.0) <<std::endl; // Sabina needs to print
          std::cout << "the isothermal pressure is: " <<phydro->u(IDN,k,PI,i)*iso_sound_speed*iso_sound_speed <<std::endl; // Sabina needs to print*/
        }
      }
    }
  }


  return;
}

//----------------------------------------------------------------------------------------
//!\f transform to cylindrical coordinate

void GetCylCoord(Coordinates *pco,Real &rad,Real &phi,Real &z,int i,int j,int k) {
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    rad=pco->x1v(i);
    phi=pco->x2v(j);
    z=pco->x3v(k);
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    rad=std::fabs(pco->x1v(i)*std::sin(pco->x2v(j)));
    phi=pco->x3v(i);
    z=pco->x1v(i)*std::cos(pco->x2v(j));
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \f  computes density in cylindrical coordinates

Real DenProfileCyl(const Real rad, const Real phi, const Real z) {
  Real den;
  /*Real p_over_r = p0_over_r0;
  if (NON_BAROTROPIC_EOS) p_over_r = PoverR(rad, phi, z);
  Real denmid = rho0*std::pow(rad/r0,dslope);
  if (gauss_width>0.) denmid = rho0*exp(-(rad-r0)*(rad-r0)/gauss_width/gauss_width);
  Real dentem = denmid*std::exp(gm0/p_over_r*(1./std::sqrt(SQR(rad)+SQR(z))-1./rad));
  den = dentem;*/
  Real diff = std::pow(rad/r0,dslope*(gamma_gas-1))-gm0*(gamma_gas-1)/(sound_speed*sound_speed)*(1./rad-1./std::sqrt(SQR(rad)+SQR(z)));
  den = rho0*std::pow(diff,(1/(gamma_gas-1)));
  if(std::isnan(den) == 1){
    den = dfloor;
  }

  //std::cout << "the density is "<< std::max(den,dfloor) <<std::endl;
  return std::max(den,dfloor);
}


//----------------------------------------------------------------------------------------
//! \f  computes isentropic pressure

/*Real PreessProfile(const Real rad, const Real phi, const Real z) {
  Real pre;
  Real den = DenProfileCyl(rad,phi,z);
  pre = sound_speed*sound_speed*rho0*std::pow(den,gamma_gas)/std::pow(rho0,gamma_gas)/gamma_gas;
  return pre;
}*/

//----------------------------------------------------------------------------------------
//! \f  computes pressure/density in cylindrical coordinates

Real PoverR(const Real rad, const Real phi, const Real z) {
  Real poverr;
  Real den = DenProfileCyl(rad,phi,z);
  //poverr = p0_over_r0*std::pow(rad/r0, pslope);
  poverr = sound_speed*sound_speed*std::pow(den,gamma_gas-1)/gamma_gas;
  //std::cout << "the pres_over_r is "<< poverr <<std::endl;

  return poverr;
}

//----------------------------------------------------------------------------------------
//! \f  computes rotational velocity in cylindrical coordinates

void VelProfileCyl(const Real rad, const Real phi, const Real z,
                   Real &v1, Real &v2, Real &v3) {
  Real p_over_r = PoverR(rad, phi, z);
  /*Real vel = (dslope+pslope)*p_over_r/(gm0/rad) + (1.0+pslope)
             - pslope*rad/std::sqrt(rad*rad+z*z);
  vel = std::sqrt(gm0/rad)*std::sqrt(vel);*/

  Real den, vel, ome, dis;
  den = DenProfileCyl(rad,phi,z);
  Real gradpre = sound_speed*sound_speed*dslope*std::pow(rho0,1+gamma_gas-1*(dslope-1))*std::pow(rad,dslope*gamma_gas-1);
  Real diff = gradpre+den*gm0/(rad*rad);
  vel = std::sqrt(diff/(den*rad));

  if (std::isnan(vel) == 1) {
    vel = std::sqrt((den*gm0/(rad*rad))/(den*rad));
  } 
  

  vel = vel*rad;

  //std::cout << "the velocity is "<< vel <<std::endl;
  //Real vx = 0;
  //Real vy = 1/2*std::sqrt(2*psys->mass[0]/0.05);

  /*if(obj==0){
    vy = 1/2*std::sqrt(2*psys->mass[0]/0.05);
  } else if(obj==1){
    vy = -1/2*std::sqrt(2*psys->mass[1]/0.05);
  }*/
  //std::cout << "the vy is: " <<vy<<std::endl;
  /*Real vz = 0;

  Real x = rad*std::cos(phi);
  Real y = rad*std::sin(phi);*/
  
  //std::cout << "the x coord is: " <<x<<std::endl;
  //std::cout << "the y coord is: " <<y<<std::endl;


  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    v1=0.0;
    v2=vel;
    v3=0.0;
    if(omegarot!=0.0) v2-=omegarot*rad;
    if(Omega0!=0.0) v2-=Omega0*rad;
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    //v1=(y*vy)/rad;
    v1=0.0;
    //std::cout << "the v1 is: " <<v1<<std::endl;
    v2=0.0;
    //v3=vy*rad/((1+y*y)*z)+vel;
    //v3=std::sqrt(vy*vy-v1*v1)-vel;
    v3=vel;
    //std::cout << "the v3 is: " <<v3<<std::endl;
    if(omegarot!=0.0) v3-=omegarot*rad;
    if(Omega0!=0.0) v3-=Omega0*rad;
  }
  return;
}

//-----------------------------------------------------------------------
//! \f fargo scheme to substract orbital velocity from v2

void SubtractionOrbitalVelocity(OrbitalAdvection *porb, Coordinates *pco,
                                Real &v1, Real &v2, Real &v3, int i, int j, int k) {
  OrbitalVelocityFunc &vK = porb->OrbitalVelocity;
  Real x1 = pco->x1v(i);
  Real x2 = pco->x2v(j);
  Real x3 = pco->x3v(k);
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    v2 -= vK(porb,x1,x2,x3);
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    v3 -= vK(porb,x1,x2,x3);
  }
}

Real RotOrbitalVelocity(OrbitalAdvection *porb, Real x_, Real y_, Real z_){
  return std::sqrt(gm0/(x_*std::sin(y_)))-omegarot*x_*std::sin(y_);
//  return std::sqrt(gm0/x_)-omegarot*std::sin(y_)*x_;
}

Real RotOrbitalVelocity_r(OrbitalAdvection *porb, Real x_, Real y_, Real z_){
  return -0.5*std::sqrt(gm0/(std::sin(y_)*x_*x_*x_))-omegarot*std::sin(y_);
//  return -0.5*std::sqrt(gm0/x_)/x_-omegarot*std::sin(y_);
}

Real RotOrbitalVelocity_t(OrbitalAdvection *porb, Real x_, Real y_, Real z_){
  return -0.5*std::sqrt(gm0/(std::sin(y_)*x_))*std::cos(y_)/std::sin(y_)
         -omegarot*x_*std::cos(y_);
//  return -omegarot*std::cos(y_)*x_;
}

void AlphaVis(HydroDiffusion *phdif, MeshBlock *pmb,const  AthenaArray<Real> &w, const AthenaArray<Real> &bc,int is, int ie, int js, int je, int ks, int ke) {
  Real rad,phi,z;
  Coordinates *pcoord = pmb->pcoord;
  if (phdif->nu_iso > 0.0) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          GetCylCoord(pcoord,rad,phi,z,i,j,k);
          phdif->nu(HydroDiffusion::DiffProcess::iso,k,j,i) = alpha*std::pow(rad/r0,aslope)*PoverR(rad, phi, z)/(sqrt(gm0/rad)/rad);
        }
      }
    }
  }
 if (phdif->nu_aniso > 0.0) {
    for (int ij=0; ij<=2; ij++) {
      for (int ii=0; ii<=2; ii++) {
       //phdif->ani(ij,ii)=0.;
      }
    }
    //phdif->ani(0,2)=1.;
    //phdif->ani(2,0)=1.;  //SABINA COMMENTED

/*    phdif->ani(0,0)=1.;
      phdif->ani(1,1)=1.;
      phdif->ani(2,2)=1.;
      phdif->ani(0,1)=1.;
      phdif->ani(1,0)=1.;
      phdif->ani(1,2)=1.;
      phdif->ani(2,1)=1.;
*/
    if(visflag==0){
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            GetCylCoord(pcoord,rad,phi,z,i,j,k);
            phdif->nu(HydroDiffusion::DiffProcess::aniso,k,j,i) = alphaa*std::pow(rad/r0,aslope)*PoverR(rad, phi, z)/(sqrt(gm0/rad)/rad);
          }
        }
      }
    }
    if(visflag==1){
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            GetCylCoord(pcoord,rad,phi,z,i,j,k);
            Real height=sqrt(PoverR(rad, phi, z))/(sqrt(gm0/rad)/rad);
            Real ch=3.;
            if(std::fabs(z)<=ch*height){
              phdif->nu(HydroDiffusion::DiffProcess::aniso,k,j,i) = 
                alphaa*std::pow(rad/r0,aslope)*PoverR(rad, phi, z)/(sqrt(gm0/rad)/rad)*sqrt(2.*PI)/2./ch*exp(z*z/height/height/2.);
            }           
          }
        }
      }
    }
  }

  return;
}



//--------------------------------------------------------------------------
//!\f: User-work in loop: add damping boundary condition at the inner and outer boundaries
//

void MeshBlock::UserWorkInLoop() {
    Real smooth, tau ;

    if (lid == 0){
      psys->f_acc_x_user[0] = psys->f_acc_x[0];
      psys->f_acc_y_user[0] = psys->f_acc_y[0];
      psys->f_acc_z_user[0] = psys->f_acc_z[0];

      psys->f_acc_x[0] = 0.0;
      psys->f_acc_y[0] = 0.0;
      psys->f_acc_z[0] = 0.0;

      psys->f_acc_x_user[1] = psys->f_acc_x[1];
      psys->f_acc_y_user[1] = psys->f_acc_y[1];
      psys->f_acc_z_user[1] = psys->f_acc_z[1];

      psys->f_acc_x[1] = 0.0;
      psys->f_acc_y[1] = 0.0;
      psys->f_acc_z[1] = 0.0;
    }


    Real rad(0.0), phi(0.0), z(0.0);
    Real den(0.0), v1(0.0), v2(0.0), v3(0.0), pre(0.0), tote(0.0);
    Real x1min = pmy_mesh->mesh_size.x1min;
    Real x1max = pmy_mesh->mesh_size.x1max;
    Real dt = pmy_mesh->dt;
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          if (pcoord->x1v(i)<rindamp||pcoord->x1v(i)>routdamp){
            if (pcoord->x1v(i)<rindamp) {
              smooth = 1.- SQR(std::sin(PI/2.*(pcoord->x1v(i)-x1min)/(rindamp-x1min)));
              tau = 0.1*2.*PI*std::sqrt(x1min/gm0)*x1min;
            }
            if (pcoord->x1v(i)>routdamp){
              smooth = SQR(std::sin(PI/2.*(pcoord->x1v(i)-routdamp)/(x1max-routdamp)));
              tau = 0.1*2.*PI*std::sqrt(x1max/gm0)*x1max;
            } 
            GetCylCoord(pcoord,rad,phi,z,i,j,k);
            den = DenProfileCyl(rad,phi,z);
            VelProfileCyl(rad,phi,z,v1,v2,v3);       
            if(porb->orbital_advection_defined) {
              SubtractionOrbitalVelocity(porb,pcoord,v1,v2,v3,i,j,k);
            }
            if (NON_BAROTROPIC_EOS){
              Real gam = peos->GetGamma();
              

              pre = PoverR(rad, phi, z)*den;
              tote = 0.5*den*(SQR(v1)+SQR(v2)+SQR(v3)) + pre/(gam-1.0);
            }
            Real taud = tau/smooth;
            phydro->u(IDN,k,j,i) = (phydro->u(IDN,k,j,i)*taud + den*dt)/(dt + taud);            
            phydro->u(IM1,k,j,i) = (phydro->u(IM1,k,j,i)*taud + den*v1*dt)/(dt + taud);
            phydro->u(IM2,k,j,i) = (phydro->u(IM2,k,j,i)*taud + den*v2*dt)/(dt + taud);    
            phydro->u(IM3,k,j,i) = (phydro->u(IM3,k,j,i)*taud + den*v3*dt)/(dt + taud);
            if (NON_BAROTROPIC_EOS)
        phydro->u(IEN,k,j,i) = (phydro->u(IEN,k,j,i)*taud + tote*dt)/(dt + taud);         
          }
        }
      }
    }

    

  return;
}

/*void AllSourceTerms(MeshBlock *pmb, const Real time, const Real dt,
  const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons)*/
void AllSourceTerms(MeshBlock *pmb, const Real time, const Real dt,
        const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons, AthenaArray<Real> &cons_scalar)
{
  BinarySourceTerms(pmb,time,dt,prim,bcc,cons);
  if(NON_BAROTROPIC_EOS&&tcool>0.0) Cooling(pmb,time,dt,prim,bcc,cons);
  return;
}

void Cooling(MeshBlock *pmb, const Real time, const Real dt,
  const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons)
{
  if(tcool>0.0) {
    Coordinates *pco = pmb->pcoord;
    Real rad,phi,z;
    for(int k=pmb->ks; k<=pmb->ke; ++k){
      for (int j=pmb->js; j<=pmb->je; ++j) {
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          GetCylCoord(pco,rad,phi,z,i,j,k);
          Real eint = cons(IEN,k,j,i)-0.5*(SQR(cons(IM1,k,j,i))+SQR(cons(IM2,k,j,i))
                                          +SQR(cons(IM3,k,j,i)))/cons(IDN,k,j,i);
          Real pres_over_r=eint*(gamma_gas-1.0)/cons(IDN,k,j,i);
          Real p_over_r = PoverR(rad, phi, z);
          Real dtr = std::max(tcool*2.*PI/sqrt(gm0/rad/rad/rad),dt);
          Real dfrac=dt/dtr;
          Real dE=eint-p_over_r/(gamma_gas-1.0)*cons(IDN,k,j,i);
          cons(IEN,k,j,i) -= dE*dfrac; 
        }
      }
    }
  }
}

Real sink_fraction(Real d_over_rs) {
/*----------------------------------------------------------------------------*/
/*! \fn Real sink_fraction(Real d_over_rs)
 *  \brief returns the fraction f of mass and linear momentum that is to be
 *  drained from each cell based on a smooth spline function (see Springel+01)
 *                   ╱ 1 - 6 *（d/rs)^2 + 6 * (d/rs)^3  if 0<= d/rs <=0.5
 *  f(dist/r_sink) = |
 *                   ╲ 2 * (1 - d/rs)^3                 if 0.5< d/rs <=1
 */
    Real f = 0.0;
    if (d_over_rs <= 0.5) {
        f = 1 - 6 * std::pow(d_over_rs, 2.) + 6 * std::pow(d_over_rs, 3.);
    } else if (d_over_rs > 1) {
        // f = 1.0;
        f = 1.0;
    } else {
        f = 2 * std::pow(1 - d_over_rs, 3.);
    }

    // Sabina: want to avoid 0
    // if (f == 0.0){
    //   f = 1.0;
    // }
    return f;
}

void BinarySourceTerms(MeshBlock *pmb, const Real time, const Real dt,
  const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons)
{
  const AthenaArray<Real> *flux=pmb->phydro->flux;
  Real cosx3, sinx3, x3, cosx2, sinx2, x2;
  Real xcar, ycar, zcar;
  Real EccA, TrA; // eccentric and true anomalies
  Real xpp,ypp,zpp;

  // psys->orbit(time);
 
  Coordinates *pco = pmb->pcoord;
  AthenaArray<Real> vol;
  vol.NewAthenaArray((pmb->ie-pmb->is)+1+2*(NGHOST));


  Real src[NHYDRO];
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    x3=pco->x3v(k);
    if(std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0){  
      cosx3=cos(x3);
      sinx3=sin(x3);
    }
    for (int j=pmb->js; j<=pmb->je; ++j) {
      x2=pco->x2v(j);
      cosx2=cos(x2);
      sinx2=sin(x2);
      Real sm = std::fabs(std::sin(pco->x2f(j  )));
      Real sp = std::fabs(std::sin(pco->x2f(j+1)));
      Real cmmcp = std::fabs(std::cos(pco->x2f(j  )) - std::cos(pco->x2f(j+1)));
      Real coord_src1_j=(sp-sm)/cmmcp;
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real rm = pco->x1f(i  );
        Real rp = pco->x1f(i+1);
        Real coord_src1_i=1.5*(rp*rp-rm*rm)/(rp*rp*rp-rm*rm*rm);
        Real drs = pco->dx1v(i) / 10000.;
        if(std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0){
          xcar = pco->x1v(i)*sinx2*cosx3;
          ycar = pco->x1v(i)*sinx2*sinx3;
          zcar = pco->x1v(i)*cosx2;
        }
        if(std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0){
          xcar = pco->x1v(i)*cosx2;
          ycar = pco->x1v(i)*sinx2;
          zcar = x3;
        }
        Real f_x1 = 0.0;
        Real f_x2 = 0.0;
        Real f_x3 = 0.0;
        Real gravBin_pot;

        for (int ip=0; ip< psys->np; ++ip){

          psys->Position(time,ip);

          // the code below computes keplerian positions:

          // double period = 2.0 * PI * sqrt(pow(a_bin * 0.5, 3) / (psys->mass[0] + psys->mass[1])); // period 

  
          // if(ip==0){
          //   EccA = compute_newton_raphson(10,2 * PI * time / period,eccent); // eccentric anomaly
          //   TrA = 2.0 * atan(sqrt((1.0 - eccent) / (1.0 + eccent)) * tan(EccA / 2.0)); // true anomaly
          //   // f = 2.0 * atan2(sqrt(1.0 - eccent * eccent) * sin(E) / (1.0 + eccent * cos(E)), cos(E) / (1.0 + eccent)); // true anomaly
          //   xpp = 0.5 * a_bin * (cos(TrA) - eccent) - psys->xp[ip];
          //   ypp = 0.5 * a_bin * sqrt(1 - eccent * eccent) * sin(TrA);
          //   zpp = 0.0;
          // } else if(ip==1){
          //   EccA = compute_newton_raphson(10,2 * PI * time / period + PI,eccent);
          //   TrA = 2.0 * atan(sqrt((1.0 - eccent) / (1.0 + eccent)) * tan(EccA / 2.0)); // true anomaly
          //   // f = 2.0 * atan2(sqrt(1.0 - eccent * eccent) * sin(E) / (1.0 + eccent * cos(E)), cos(E) / (1.0 + eccent)); // true anomaly
          //   xpp = 0.5 * a_bin * (cos(TrA) - eccent) - psys->xp[ip];
          //   ypp = 0.5 * a_bin * sqrt(1 - eccent * eccent) * sin(TrA);
          //   zpp = 0.0;
          // }


          // double dis; // separation between the object and the COM
          // dis = sqrt((xpp + 1) * (xpp + 1) + ypp * ypp + zpp * zpp);
          // Real vpp = sqrt((psys->mass[0] + psys->mass[1])*(2 / dis - 1 / a_bin)); //orbital velocity


          Real xpp=psys->xp[ip]; 
          Real ypp=psys->yp[ip];
          Real zpp=psys->zp[ip];
          Real mp=psys->mass[ip];

          // Real f_acc_xp = psys->f_acc_x[ip];
          // Real f_acc_yp = psys->f_acc_y[ip];
          // Real f_acc_zp = psys->f_acc_z[ip];

          Real rsoft2=psys->rsoft2;
          Real vpp=psys->vp[ip];
          Real voluume = 0.0;
          pco->CellVolume(k,j,pmb->is,pmb->ie,vol);

          if (r_s!=0){

            double cmx = -1;
            double cmy = 0;
            double cmz = 0;
            double R; // distance from the COM
            R = sqrt((xpp-cmx)*(xpp-cmx)+(ypp-cmy)*(ypp-cmy)+(zpp-cmz)*(zpp-cmz));
   
            double ome = sqrt(mp/R/R/R)*dt;

            double b = R*R+1.*1.+2*R*1.*cos(ome);
            double cosalpha = (R*R - b*b - 1.*1.)/(2*b*1.);

            Real vppr = vpp*cosalpha; // from the SMBH POV
            Real vppphi = sqrt(mp/b/b/b)*dt*b; // from the SMBH POV
            Real vpptheta = 0; // from the SMBH POV

            Real dist = sqrt((xcar-xpp)*(xcar-xpp) + (ycar-ypp)*(ycar-ypp) + (zcar-zpp)*(zcar-zpp));
            f = sink_fraction(dist/r_s); 

            // cons(IDN,k,j,i)=1;
            // cons(IM1,k,j,i)=1;
            // cons(IM2,k,j,i)=1;
            // cons(IM3,k,j,i)=1;

            // if (dist<r_s){
              // std::cout << "dist/r_s is "<<dist/r_s<<std::endl;
              // std::cout << "f is "<<f<<std::endl;
            if (cons(IDN,k,j,i)>=rho0) {
              cons(IDN,k,j,i) = cons(IDN,k,j,i) * std::max(f, 0.1);
            }
            mass_sink = cons(IDN,k,j,i) * vol(i);

            cons(IEN,k,j,i) = cons(IEN,k,j,i) * std::max(f, 0.1);

            Real Old_mom1 = cons(IM1,k,j,i);
            Real Old_mom2 = cons(IM2,k,j,i);
            Real Old_mom3 = cons(IM3,k,j,i);

            cons(IM1,k,j,i) = (cons(IM1,k,j,i) - vppr*prim(IDN,k,j,i)) * std::max(f, 0.1) + vppr*prim(IDN,k,j,i);
            cons(IM2,k,j,i) = (cons(IM2,k,j,i) - vpptheta*prim(IDN,k,j,i)) * std::max(f, 0.1) + vpptheta*prim(IDN,k,j,i);
            cons(IM3,k,j,i) = (cons(IM3,k,j,i) - vppphi*prim(IDN,k,j,i)) * std::max(f, 0.1) + vppphi*prim(IDN,k,j,i);

            Real New_mom1 = (cons(IM1,k,j,i) - vppr*prim(IDN,k,j,i)) * std::max(f, 0.1) + vppr*prim(IDN,k,j,i);
            Real New_mom2 = (cons(IM2,k,j,i) - vpptheta*prim(IDN,k,j,i)) * std::max(f, 0.1) + vpptheta*prim(IDN,k,j,i);
            Real New_mom3 = (cons(IM3,k,j,i) - vppphi*prim(IDN,k,j,i)) * std::max(f, 0.1) + vppphi*prim(IDN,k,j,i);

            // saving the accretion forces and transforming them into cartesian coordinates

            Real f_acc_r = (Old_mom1 - New_mom1) / dt * vol(i);
            Real f_acc_theta = (Old_mom2 - New_mom2) / dt  * vol(i);
            Real f_acc_phi = (Old_mom3 - New_mom3) / dt  * vol(i);

            psys->f_acc_x[ip] += f_acc_r * sin(pco->x2v(j)) * cos(pco->x3v(k)) + f_acc_theta * cos(pco->x2v(j)) * cos(pco->x3v(k)) - f_acc_phi * sin(pco->x3v(k));
            psys->f_acc_y[ip] += f_acc_r * sin(pco->x2v(j)) * sin(pco->x3v(k)) + f_acc_theta * cos(pco->x2v(j)) * sin(pco->x3v(k)) + f_acc_phi * cos(pco->x3v(k));
            psys->f_acc_z[ip] += f_acc_r * cos(pco->x2v(j)) - f_acc_theta * sin(pco->x2v(j));

          }

          


          Real f_xca = -1.0* (grav_pot_car_btoa(xcar+drs, ycar, zcar,xpp,ypp,zpp,mp,rsoft2)
                                -grav_pot_car_btoa(xcar-drs, ycar, zcar,xpp,ypp,zpp,mp,rsoft2))/(2.0*drs);
          Real f_yca = -1.0* (grav_pot_car_btoa(xcar, ycar+drs, zcar,xpp,ypp,zpp,mp,rsoft2)
                                -grav_pot_car_btoa(xcar, ycar-drs, zcar,xpp,ypp,zpp,mp,rsoft2))/(2.0*drs);
          Real f_zca = -1.0* (grav_pot_car_btoa(xcar, ycar, zcar+drs,xpp,ypp,zpp,mp,rsoft2)
                                -grav_pot_car_btoa(xcar, ycar, zcar-drs,xpp,ypp,zpp,mp,rsoft2))/(2.0*drs);
          if(psys->ind!=0){
            f_xca += -1.0* (grav_pot_car_ind(xcar+drs, ycar, zcar,xpp,ypp,zpp,mp)
                                -grav_pot_car_ind(xcar-drs, ycar, zcar,xpp,ypp,zpp,mp))/(2.0*drs);
            f_yca += -1.0* (grav_pot_car_ind(xcar, ycar+drs, zcar,xpp,ypp,zpp,mp)
                                -grav_pot_car_ind(xcar, ycar-drs, zcar,xpp,ypp,zpp,mp))/(2.0*drs);
            f_zca += -1.0* (grav_pot_car_ind(xcar, ycar, zcar+drs,xpp,ypp,zpp,mp)
                                -grav_pot_car_ind(xcar, ycar, zcar-drs,xpp,ypp,zpp,mp))/(2.0*drs);
          }
          if(std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0){
            f_x1 += f_xca*sinx2*cosx3+f_yca*sinx2*sinx3+f_zca*cosx2;
            f_x2 += f_xca*cosx2*cosx3+f_yca*cosx2*sinx3-f_zca*sinx2;
            f_x3 += f_xca*(-sinx3) + f_yca*cosx3;
          }
          if(std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0){
            f_x1 += f_xca*cosx2 + f_yca*sinx2;
            f_x2 += -f_xca*sinx2 + f_yca*cosx2;
            f_x3 += f_zca;
          }
        }
        if(omegarot!=0.0) {
        /* centrifugal force */
          f_x1 += omegarot*omegarot*pco->x1v(i)*pco->x1v(i)*sinx2*sinx2*coord_src1_i;
          f_x2 += omegarot*omegarot*pco->x1v(i)*pco->x1v(i)*sinx2*sinx2*coord_src1_i*coord_src1_j;
        }

        src[IM1] = dt*prim(IDN,k,j,i)*f_x1;
        src[IM2] = dt*prim(IDN,k,j,i)*f_x2;
        src[IM3] = dt*prim(IDN,k,j,i)*f_x3;

        cons(IM1,k,j,i) += src[IM1];
        cons(IM2,k,j,i) += src[IM2];
        cons(IM3,k,j,i) += src[IM3];
        if(NON_BAROTROPIC_EOS) {
          src[IEN] = f_x1*dt*0.5*(flux[X1DIR](IDN,k,j,i)+flux[X1DIR](IDN,k,j,i+1))+f_x2*dt*0.5*(flux[X2DIR](IDN,k,j,i)+flux[X2DIR](IDN,k,j+1,i))+f_x3*dt*0.5*(flux[X3DIR](IDN,k,j,i)+flux[X3DIR](IDN,k+1,j,i));
//          src[IEN] = src[IM1]*prim(IM1,k,j,i)+ src[IM2]*prim(IM2,k,j,i) + src[IM3]*prim(IM3,k,j,i);
          if(pmb->porb->orbital_advection_defined&&std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0){
            Real vadv=pmb->porb->OrbitalVelocity(pmb->porb,pco->x1v(i),pco->x2v(j),pco->x3v(k));
            src[IEN] += dt*coord_src1_i*(0.5*(flux[X1DIR](IDN,k,j,i)+flux[X1DIR](IDN,k,j,i+1))*2.*omegarot*pco->x1v(i)*sinx2*vadv
                                        +0.5*(flux[X2DIR](IDN,k,j,i)+flux[X2DIR](IDN,k,j+1,i))*2.*omegarot*pco->x1v(i)*sinx2*vadv*coord_src1_j);  
//            src[IEN] += dt*coord_src1_i*(prim(IM1,k,j,i)*prim(IDN,k,j,i)*2.*omegarot*pco->x1v(i)*sinx2*vadv
//                                        +prim(IM2,k,j,i)*prim(IDN,k,j,i)*2.*omegarot*pco->x1v(i)*sinx2*vadv*coord_src1_j);                      
          }


          cons(IEN,k,j,i) += src[IEN];
        }
        if(omegarot!=0.0) {
        /* Coriolis force */
          cons(IM1,k,j,i) += omegarot*sinx2*(flux[X3DIR](IDN,k+1,j,i)+flux[X3DIR](IDN,k,j,i))*pco->x1v(i)*coord_src1_i*dt;
          cons(IM2,k,j,i) += omegarot*sinx2*(flux[X3DIR](IDN,k+1,j,i)+flux[X3DIR](IDN,k,j,i))*pco->x1v(i)*coord_src1_i*coord_src1_j*dt;
          cons(IM3,k,j,i) -= (omegarot*3.*(sp+sm)/(rp+rm)/(rp*rp+rp*rm+rm*rm)*(rp*rp*(3.*rp+rm)/4.*flux[X1DIR](IDN,k,j,i+1)+rm*rm*(3.*rm+rp)/4.*flux[X1DIR](IDN,k,j,i))+omegarot*3.*(rp+rm)*(rp+rm)*(sp-sm)/2./(sp+sm)/(rp*rp+rp*rm+rm*rm)/cmmcp*((3.*sp+sm)/4.*sp*flux[X2DIR](IDN,k,j+1,i)+(3.*sm+sp)/4.*sm*flux[X2DIR](IDN,k,j,i)))*dt;

          if(pmb->porb->orbital_advection_defined&&std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0){
            Real vadv=pmb->porb->OrbitalVelocity(pmb->porb,pco->x1v(i),pco->x2v(j),pco->x3v(k));
            cons(IM1,k,j,i) += omegarot*sinx2*2.*prim(IDN,k,j,i)*vadv*pco->x1v(i)*coord_src1_i*dt;
            cons(IM2,k,j,i) += omegarot*sinx2*2.*prim(IDN,k,j,i)*vadv*pco->x1v(i)*coord_src1_i*coord_src1_j*dt;
          }
/*
          Real dt_den_over_r = dt*prim(IDN,k,j,i)*coord_src1_i;
          Real rv  = pmb->pcoord->x1v(i);
          Real vc  = rv*std::sin(pmb->pcoord->x2v(j))*omegarot;
          Real cv1 = coord_src1_j;
          Real cv3 = coord_src1_j;

          Real src_i1 = 2.0*dt_den_over_r*vc*prim(IVZ,k,j,i);
          if(pmb->porb->orbital_advection_defined){
            Real vadv=pmb->porb->OrbitalVelocity(pmb->porb,pco->x1v(i),pco->x2v(j),pco->x3v(k));
            src_i1 += 2.0*dt_den_over_r*vc*vadv;
          }
          cons(IM1,k,j,i) += src_i1;
          cons(IM2,k,j,i) += src_i1*cv1;
*/
//          cons(IM3,k,j,i) += -2.0*dt_den_over_r*vc*(prim(IVX,k,j,i)+cv3*prim(IVY,k,j,i));

/* the following is identical to the implementation of the fargo scheme */
//          cons(IM3,k,j,i) += -2.0*dt*coord_src1_i*vc*
//                             (0.5*(flux[X1DIR](IDN,k,j,i)+flux[X1DIR](IDN,k,j,i+1))+
//                              cv3*0.5*(flux[X2DIR](IDN,k,j,i)+flux[X2DIR](IDN,k,j+1,i)));
        } 
      }
    }
  }
}

Real BinaryForce(MeshBlock *pmb, int iout)
{
  //std::cout << "I'm in BinaryForce!"<<std::endl;
  const AthenaArray<Real> *flux=pmb->phydro->flux;
  Real f_pres_x, f_pres_y, f_pres_z;
  if (psys->np > 0) {
    // psys->orbit(pmb->pmy_mesh->time);
    Coordinates *pco = pmb->pcoord;
    AthenaArray<Real> vol;
    vol.NewAthenaArray((pmb->ie-pmb->is)+1+2*(NGHOST));
    Real dphi=1.e-3;

    Real time = pmb->pmy_mesh->time;
    Real EccA, TrA;
    Real xpp,ypp,zpp;

    if(std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0){
      for (int ip=0; ip< psys->np; ++ip){
        Real torque_z = 0;

        Real f_xpp = 0.0;
        Real f_ypp = 0.0;
        Real f_zpp = 0.0;

        Real f_x = 0.0;
        Real f_y = 0.0;
        Real f_z = 0.0;

        Real torque = 0.0;
        Real voluume = 0.0;
        Real ecc, mdot, l, Eb;

        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        // psys->Position(time,ip);

        Real xpp=psys->xp[ip]; 
        Real ypp=psys->yp[ip];
        Real zpp=psys->zp[ip];
        Real rpp=sqrt(xpp*xpp+ypp*ypp+zpp*zpp);
        Real thepp=acos(zpp/rpp);
        Real phipp=atan2(ypp,xpp);
        Real mp=psys->mass[ip];
        Real vpp=psys->vp[ip];
        Real rsoft2=psys->rsoft2;
        for (int k=pmb->ks; k<=pmb->ke; ++k) {
          Real x3=pco->x3v(k);
          Real cosx3=cos(x3);
          Real sinx3=sin(x3);
          Real x3p=pco->x3v(k)+dphi;
          Real cosx3p=cos(x3p);
          Real sinx3p=sin(x3p);
          Real x3m=pco->x3v(k)-dphi;
          Real cosx3m=cos(x3m);
          Real sinx3m=sin(x3m);
          for (int j=pmb->js; j<=pmb->je; ++j) {
            pco->CellVolume(k,j,pmb->is,pmb->ie,vol);
            Real x2=pco->x2v(j);
            Real cosx2=cos(x2);
            Real sinx2=sin(x2);
            for (int i=pmb->is; i<=pmb->ie; ++i) {
              Real drs = pco->dx1v(i) / 10000.;
              Real xcar = pco->x1v(i)*sinx2*cosx3;
              Real ycar = pco->x1v(i)*sinx2*sinx3;
              Real zcar = pco->x1v(i)*cosx2;


              f_xpp += vol(i)*pmb->phydro->u(IDN,k,j,i)*(grav_pot_car_btoa(xcar+drs, ycar, zcar,xpp,ypp,zpp,mp,rsoft2)
                                  -grav_pot_car_btoa(xcar-drs, ycar, zcar,xpp,ypp,zpp,mp,rsoft2))/(2.0*drs);
              f_ypp += vol(i)*pmb->phydro->u(IDN,k,j,i)*(grav_pot_car_btoa(xcar, ycar+drs, zcar,xpp,ypp,zpp,mp,rsoft2)
                                  -grav_pot_car_btoa(xcar, ycar-drs, zcar,xpp,ypp,zpp,mp,rsoft2))/(2.0*drs);
              f_zpp += vol(i)*pmb->phydro->u(IDN,k,j,i)*(grav_pot_car_btoa(xcar, ycar, zcar+drs,xpp,ypp,zpp,mp,rsoft2)
                                  -grav_pot_car_btoa(xcar, ycar, zcar-drs,xpp,ypp,zpp,mp,rsoft2))/(2.0*drs);
              Real xpcar = pco->x1v(i)*sinx2*cosx3p;
              Real ypcar = pco->x1v(i)*sinx2*sinx3p;
              Real zpcar = pco->x1v(i)*cosx2;
              Real xmcar = pco->x1v(i)*sinx2*cosx3m;
              Real ymcar = pco->x1v(i)*sinx2*sinx3m;
              Real zmcar = pco->x1v(i)*cosx2;
              torque += -vol(i)*pmb->phydro->u(IDN,k,j,i)*(grav_pot_car_btoa(xpcar,ypcar,zpcar,xpp,ypp,zpp,mp,rsoft2)
                                  -grav_pot_car_btoa(xmcar,ymcar,zmcar,xpp,ypp,zpp,mp,rsoft2))/2.0/dphi;     

              voluume = vol(i);

              double cmx = -1;
              double cmy = 0;
              double cmz = 0;
              double R; // distance from the COM
              R = sqrt((xpp-cmx)*(xpp-cmx)+(ypp-cmy)*(ypp-cmy)+(zpp-cmz)*(zpp-cmz));
              l=R*vpp*sin(phipp);
              //l=(xpp-cmx)*psys->vpy[ip]-(ypp-cmy)*psys->vpx[ip];
              //Real l = dis*(1+ecc*cos(thepp)); //from wikipedia
              //Real a = l/(1-ecc*ecc);
              Eb = (mp*psys->vp[ip]*psys->vp[ip]*R-2*mp)/(2*R);


              // computing the pressure forces for each cell, then transforming them into cartesian coordinates 
              
              Real f_pres_r = flux[X1DIR](IM1,k,j,i) * pco->GetFace1Area(k,j,i) - flux[X1DIR](IM1,k,j,i+1) * pco->GetFace1Area(k,j,i+1) 
              + flux[X2DIR](IM1,k,j,i) * pco->GetFace2Area(k,j,i) - flux[X2DIR](IM1,k,j+1,i) * pco->GetFace2Area(k,j+1,i) 
              + flux[X3DIR](IM1,k,j,i) * pco->GetFace3Area(k,j,i) - flux[X3DIR](IM1,k+1,j,i) * pco->GetFace3Area(k+1,j,i); 

              Real f_pres_theta = flux[X1DIR](IM2,k,j,i) * pco->GetFace1Area(k,j,i) - flux[X1DIR](IM2,k,j,i+1) * pco->GetFace1Area(k,j,i+1) 
              + flux[X2DIR](IM2,k,j,i) * pco->GetFace2Area(k,j,i) - flux[X2DIR](IM2,k,j+1,i) * pco->GetFace2Area(k,j+1,i) 
              + flux[X3DIR](IM2,k,j,i) * pco->GetFace3Area(k,j,i) - flux[X3DIR](IM2,k+1,j,i) * pco->GetFace3Area(k+1,j,i); 

              Real f_pres_phi = flux[X1DIR](IM3,k,j,i) * pco->GetFace1Area(k,j,i) - flux[X1DIR](IM3,k,j,i+1) * pco->GetFace1Area(k,j,i+1) 
              + flux[X2DIR](IM3,k,j,i) * pco->GetFace2Area(k,j,i) - flux[X2DIR](IM3,k,j+1,i) * pco->GetFace2Area(k,j+1,i) 
              + flux[X3DIR](IM3,k,j,i) * pco->GetFace3Area(k,j,i) - flux[X3DIR](IM3,k+1,j,i) * pco->GetFace3Area(k+1,j,i); 

              f_pres_x += f_pres_r * sin(pco->x2v(j)) * cos(pco->x3v(k)) + f_pres_theta * cos(pco->x2v(j)) * cos(pco->x3v(k)) - f_pres_phi * sin(pco->x3v(k));
              f_pres_y += f_pres_r * sin(pco->x2v(j)) * sin(pco->x3v(k)) + f_pres_theta * cos(pco->x2v(j)) * sin(pco->x3v(k)) + f_pres_phi * cos(pco->x3v(k));
              f_pres_z += f_pres_r * cos(pco->x2v(j)) - f_pres_theta * sin(pco->x2v(j));

              // compute the total z-component of a torque

              // Real f_x = f_xpp + f_pres_x + f_acc_x; // this is the total force applied on a black hole
              // Real f_y = f_ypp + f_pres_y + f_acc_y;

              f_x = f_xpp + psys->f_acc_x_user[ip]; // this is the total force applied on a black hole
              f_y = f_ypp + psys->f_acc_y_user[ip];
              f_z = f_zpp + psys->f_acc_z_user[ip];
              torque_z += (xpp-cmx) * f_y - (ypp-cmy) * f_x;

            }
          }
        }
        
        Real f_r = (f_x)*sin(thepp)*cos(phipp) + (f_y)*sin(thepp)*sin(phipp) + (f_z)*cos(thepp);
        Real f_t = (f_x)*cos(thepp)*cos(phipp) + (f_y)*cos(thepp)*sin(phipp) - (f_z)*sin(thepp);
        Real f_p = (f_x)*(-sin(phipp)) + (f_y)*cos(phipp);

        if (iout==0&&ip==0) return f_r;
        if (iout==1&&ip==0) return f_t;
        if (iout==2&&ip==0) return f_p;
        if (iout==3&&ip==1) return f_r;
        if (iout==4&&ip==1) return f_t;
        if (iout==5&&ip==1) return f_p;
        if (iout==6&&ip==0) return f_x;
        if (iout==7&&ip==0) return f_y;
        if (iout==8&&ip==0) return f_z;
        if (iout==9&&ip==1) return f_x;
        if (iout==10&&ip==1) return f_y;
        if (iout==11&&ip==1) return f_z;
        if (iout==12&&ip==0) return torque_z;
        if (iout==13&&ip==1) return torque_z;
        if (rank == 0) {
            if (iout==14&&ip==0) return xpp;
            if (iout==15&&ip==0) return ypp;
            if (iout==16&&ip==0) return zpp;
            if (iout==17&&ip==1) return xpp;
            if (iout==18&&ip==1) return ypp;
            if (iout==19&&ip==1) return zpp;
            if (iout==20&&ip==0) return rpp;
            if (iout==21&&ip==0) return thepp;
            if (iout==22&&ip==0) return phipp;
            if (iout==23&&ip==1) return rpp;
            if (iout==24&&ip==1) return thepp;
            if (iout==25&&ip==1) return phipp;
            if (iout==26&&ip==0) return mp;
            if (iout==27&&ip==1) return mp;
            if (iout==28&&ip==0) return Eb;
            if (iout==29&&ip==1) return Eb;
            if (iout==30&&ip==0) return voluume;
            if (iout==31&&ip==0) return vpp;
            if (iout==32&&ip==1) return vpp;
            if (iout==33&&ip==0) return mass_sink;
          }

        if (pmb->lid == 0) {
            if (iout==34&&ip==0) return psys->f_acc_x_user[ip];
            if (iout==35&&ip==1) return psys->f_acc_x_user[ip];
            if (iout==36&&ip==0) return psys->f_acc_y_user[ip];
            if (iout==37&&ip==1) return psys->f_acc_y_user[ip];
            if (iout==38&&ip==0) return psys->f_acc_z_user[ip];
            if (iout==39&&ip==1) return psys->f_acc_z_user[ip];
        }
        
        if (iout==40&&ip==0) return f_xpp;
        if (iout==41&&ip==1) return f_xpp;
        if (iout==42&&ip==0) return f_ypp;
        if (iout==43&&ip==1) return f_ypp;
        if (iout==44&&ip==0) return f_zpp;
        if (iout==45&&ip==1) return f_zpp;
        if (iout==46&&ip==0) return f;

          
        // if (iout==14&&ip==0) return xpp;
        // if (iout==15&&ip==0) return ypp;
        // if (iout==16&&ip==0) return zpp;
        // if (iout==17&&ip==1) return xpp;
        // if (iout==18&&ip==1) return ypp;
        // if (iout==19&&ip==1) return zpp;
        // if (iout==20&&ip==0) return rpp;
        // if (iout==21&&ip==0) return thepp;
        // if (iout==22&&ip==0) return phipp;
        // if (iout==23&&ip==1) return rpp;
        // if (iout==24&&ip==1) return thepp;
        // if (iout==25&&ip==1) return phipp;
        // if (iout==26&&ip==0) return mp;
        // if (iout==27&&ip==1) return mp;
        // if (iout==28&&ip==0) return Eb;
        // if (iout==29&&ip==1) return Eb;
        // if (iout==30&&ip==0) return voluume;
        // if (iout==31&&ip==0) return vpp;
        // if (iout==32&&ip==1) return vpp;
        // if (iout==33&&ip==0) return mass_sink;

      }
    }
  }
  return 0;
}     


void BinarySystem::Position(double time, int obj){
  double dis = sqrt((xp[obj]+1)*(xp[obj]+1)+(yp[obj]*yp[obj])+(zp[obj]*zp[obj]));

  double period = 2.0 * PI * sqrt(pow(a_bin * 0.5, 3) / (2 * mass[obj])); // period 
  
  Real E, f;

  if(obj==0){
    E = compute_newton_raphson(10,2 * PI * time / period,eccent); //eccentric anomaly
    // f = 2.0 * atan(sqrt((1.0 - eccent) / (1.0 + eccent)) * tan(E / 2.0)); // true anomaly
    f = 2.0 * atan2(sqrt(1.0 - eccent * eccent) * sin(E) / (1.0 + eccent * cos(E)), cos(E) / (1.0 + eccent));
    xp[obj]=0.5 * a_bin * (cos(f) - eccent) - 1.;
    yp[obj]=0.5 * a_bin * sqrt(1 - eccent * eccent) * sin(f);
    zp[obj]=0.0;
  } else if(obj==1){
    E = compute_newton_raphson(10,2 * PI * time / period,eccent);
    // f = 2.0 * atan(sqrt((1.0 - eccent) / (1.0 + eccent)) * tan(E / 2.0)); // true anomaly
    f = 2.0 * atan2(sqrt(1.0 - eccent * eccent) * sin(E) / (1.0 + eccent * cos(E)), cos(E) / (1.0 + eccent));
    xp[obj]=-0.5 * a_bin * (cos(f) - eccent) - 1.;
    yp[obj]=-0.5 * a_bin * sqrt(1 - eccent * eccent) * sin(f);
    zp[obj]=0.0;
  }
  vp[obj]=sqrt((2 * 1.5e-5)*(2/dis-1/a_bin)); //orbital velocity

  
}

Real eccentricity(double time){ //equation 4 in D'orazio2021
  Real e0 = 0;
  Real ef = 0.9;
  Real es;
  es = e0+ef/(2*PI*nmax)*time;
  return(es);
}

Real compute_newton_raphson(int N_it, Real M, Real e){
    /// Compute Newton-Raphson method with given number of steps
    // This has quadratic convergence.
    // The initial step is defined as E_0 = M + sgn(sin(ell))*e*k 

    double k = 0.85;
    Real f_E, fP_E, this_ell, old_E, new_E, E;


    this_ell = M;

    // Define initial estimate
    if((sin(this_ell))<0) old_E = this_ell - k*e;
    else old_E = this_ell + k*e;

    // Perform Newton-Raphson estimate
    for(int j=0;j<N_it;j++) {

    // Compute f(E) and f'(E)
    f_E = old_E - e*sin(old_E)-this_ell;
    fP_E = 1. - e*cos(old_E);

    // Update E
    old_E = old_E - f_E/fP_E;
  }
    // Add to output
    new_E = old_E;

  return(new_E);
  }


void BinarySystem::binary_force(Real x1obj, Real y1obj, Real z1obj,
        Real x2obj, Real y2obj, Real z2obj, Real* fx, Real* fy, Real* fz){

  double dist=sqrt((x1obj-x2obj)*(x1obj-x2obj)+(y1obj-y2obj)*(y1obj-y2obj)+(z1obj-z2obj)*(z1obj-z2obj));

  *fx=-mass[0]*mass[1]*(x1obj-x2obj)/dist/dist/dist;
  *fy=-mass[0]*mass[1]*(y1obj-y2obj)/dist/dist/dist;
  *fz=-mass[0]*mass[1]*(z1obj-z2obj)/dist/dist/dist;

  return;
}

//----------------------------------------------------------------------------------------
//!\f: Use grav potential to calculate forces from b to a
//

Real grav_pot_car_btoa(const Real xca, const Real yca, const Real zca,
        const Real xcb, const Real ycb, const Real zcb, const Real gb, const Real rsoft2)
{
  Real dist=sqrt((xca-xcb)*(xca-xcb) + (yca-ycb)*(yca-ycb) + (zca-zcb)*(zca-zcb));
  Real rsoft=sqrt(rsoft2);
  Real dos=dist/rsoft;
  Real pot;
  if(dist>=rsoft){
     pot=-gb/dist;
  }else{
     pot=-gb/dist*(dos*dos*dos*dos-2.*dos*dos*dos+2*dos);
  }
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    Real dists=sqrt(xca*xca+yca*yca+zca*zca);
    pot+=-1./dists;
  }
//  dist2 = (xca-xcb)*(xca-xcb) + (yca-ycb)*(yca-ycb) + (zca-zcb)*(zca-zcb);
//  Real pot = -gb*(dist2+1.5*rsoft2)/(dist2+rsoft2)/sqrt(dist2+rsoft2);


  return(pot);
}

//----------------------------------------------------------------------------------------
//!\f: Use grav potential to calculate indirect forces due to gmp 
//

Real grav_pot_car_ind(const Real xca, const Real yca, const Real zca,
        const Real xpp, const Real ypp, const Real zpp, const Real gmp)
{
  Real pdist=sqrt(xpp*xpp+ypp*ypp+zpp*zpp);
  Real pot = gmp/pdist/pdist/pdist*(xca*xpp+yca*ypp+zca*zpp);
  return(pot);
}

//----------------------------------------------------------------------------------------
//!\f: Fix planetary orbit
//
void BinarySystem::orbit(double time)
{
  int i;
  for(i=0; i<np; ++i){
    if(time<insert_time*2.*PI) {
      mass[i]=massset[i]*sin(time/insert_time/4.)*sin(time/insert_time/4.);
    }else{
      mass[i]=massset[i];
    }
    double dis=sqrt(xp[i]*xp[i]+yp[i]*yp[i]);
    double ome=(sqrt((gm0+mass[i])/dis/dis/dis)-omegarot-Omega0);
    double ang=acos(xp[i]/dis);
    ang = ome*time;
    // xp[i]=dis*cos(ang);
    // yp[i]=dis*sin(ang);
  }
  return;
}

//----------------------------------------------------------------------------------------
//!\f: User-defined boundary Conditions: sets solution in ghost zones to initial values
//

void DiskInnerX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real v1(0.0), v2(0.0), v3(0.0);
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {
        GetCylCoord(pco,rad,phi,z,il-i,j,k);
        prim(IDN,k,j,il-i) = DenProfileCyl(rad,phi,z);
        VelProfileCyl(rad,phi,z,v1,v2,v3);
        if(pmb->porb->orbital_advection_defined) {
          SubtractionOrbitalVelocity(pmb->porb,pco,v1,v2,v3,il-i,j,k);
        }
        prim(IM1,k,j,il-i) = v1;
        prim(IM2,k,j,il-i) = v2;
        prim(IM3,k,j,il-i) = v3;
        if (NON_BAROTROPIC_EOS)
          prim(IEN,k,j,il-i) = PoverR(rad, phi, z)*prim(IDN,k,j,il-i);
      }
    }
  }
}

void DiskOuterX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real v1(0.0), v2(0.0), v3(0.0);
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {
        GetCylCoord(pco,rad,phi,z,iu+i,j,k);
        prim(IDN,k,j,iu+i) = DenProfileCyl(rad,phi,z);
        VelProfileCyl(rad,phi,z,v1,v2,v3);
        if(pmb->porb->orbital_advection_defined) {
          SubtractionOrbitalVelocity(pmb->porb,pco,v1,v2,v3,iu+i,j,k);
        }
        prim(IM1,k,j,iu+i) = v1;
        prim(IM2,k,j,iu+i) = v2;
        prim(IM3,k,j,iu+i) = v3;
        if (NON_BAROTROPIC_EOS)
          prim(IEN,k,j,iu+i) = PoverR(rad, phi, z)*prim(IDN,k,j,iu+i);
      }
    }
  }
}

void DiskInnerX2(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real v1(0.0), v2(0.0), v3(0.0);
  for (int k=kl; k<=ku; ++k) {
    for (int j=1; j<=ngh; ++j) {
      for (int i=il; i<=iu; ++i) {
        GetCylCoord(pco,rad,phi,z,i,jl-j,k);
        prim(IDN,k,jl-j,i) = DenProfileCyl(rad,phi,z);
        VelProfileCyl(rad,phi,z,v1,v2,v3);
        if(pmb->porb->orbital_advection_defined) {
          SubtractionOrbitalVelocity(pmb->porb,pco,v1,v2,v3,i,jl-j,k);
        }
        prim(IM1,k,jl-j,i) = v1;
        prim(IM2,k,jl-j,i) = v2;
        prim(IM3,k,jl-j,i) = v3;
        if (NON_BAROTROPIC_EOS)
          prim(IEN,k,jl-j,i) = PoverR(rad, phi, z)*prim(IDN,k,jl-j,i);
      }
    }
  }
}

void DiskOuterX2(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real v1(0.0), v2(0.0), v3(0.0);
  for (int k=kl; k<=ku; ++k) {
    for (int j=1; j<=ngh; ++j) {
      for (int i=il; i<=iu; ++i) {
        GetCylCoord(pco,rad,phi,z,i,ju+j,k);
        prim(IDN,k,ju+j,i) = DenProfileCyl(rad,phi,z);
        VelProfileCyl(rad,phi,z,v1,v2,v3);
        if(pmb->porb->orbital_advection_defined) {
          SubtractionOrbitalVelocity(pmb->porb,pco,v1,v2,v3,i,ju+j,k);
        }
        prim(IM1,k,ju+j,i) = v1;
        prim(IM2,k,ju+j,i) = v2;
        prim(IM3,k,ju+j,i) = v3;
        if (NON_BAROTROPIC_EOS)
          prim(IEN,k,ju+j,i) = PoverR(rad, phi, z)*prim(IDN,k,ju+j,i);
      }
    }
  }
}

void DiskInnerX3(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real v1(0.0), v2(0.0), v3(0.0);
  for (int k=1; k<=ngh; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=il; i<=iu; ++i) {
        GetCylCoord(pco,rad,phi,z,i,j,kl-k);
        prim(IDN,kl-k,j,i) = DenProfileCyl(rad,phi,z);
        VelProfileCyl(rad,phi,z,v1,v2,v3);
        if(pmb->porb->orbital_advection_defined) {
          SubtractionOrbitalVelocity(pmb->porb,pco,v1,v2,v3,i,j,kl-k);
        }
        prim(IM1,kl-k,j,i) = v1;
        prim(IM2,kl-k,j,i) = v2;
        prim(IM3,kl-k,j,i) = v3;
        if (NON_BAROTROPIC_EOS)
          prim(IEN,kl-k,j,i) = PoverR(rad, phi, z)*prim(IDN,kl-k,j,i);
      }
    }
  }
}

void DiskOuterX3(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real v1(0.0), v2(0.0), v3(0.0);
  for (int k=1; k<=ngh; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=il; i<=iu; ++i) {
        GetCylCoord(pco,rad,phi,z,i,j,ku+k);
        prim(IDN,ku+k,j,i) = DenProfileCyl(rad,phi,z);
        VelProfileCyl(rad,phi,z,v1,v2,v3);
        if(pmb->porb->orbital_advection_defined) {
          SubtractionOrbitalVelocity(pmb->porb,pco,v1,v2,v3,i,j,ku+k);
        }
        prim(IM1,ku+k,j,i) = v1;
        prim(IM2,ku+k,j,i) = v2;
        prim(IM3,ku+k,j,i) = v3;
        if (NON_BAROTROPIC_EOS)
          prim(IEN,ku+k,j,i) = PoverR(rad, phi, z)*prim(IDN,ku+k,j,i);
      }
    }
  }
}
