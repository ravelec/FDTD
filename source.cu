#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <math.h>
#include <time.h>
#include <cufft.h>
#include <cufftXt.h>
#include <cuComplex.h>
#include <random>
#include "fdtdrav.h"


//using namespace std;
using std::cout;
using std::endl;


int main(int argc, char ** argv) {

    //the second input argument is the number of devices used
  const int numdev = atoi(argv[2]);

	//Measuring total time of execution
	clock_t tStart;
	//initializing hte clock
	tStart = clock();

  //Define problem main constants
	float eps_0 = 8.854187817e-12;		// permittivity of free space
	float pi = 3.1415;					// pi
	float mu_0 = (4 * pi)*1e-7;			// permeability of free space
  float c = 1 / sqrt(mu_0*eps_0);		// speed of light

  //Define space Parameters
	float dx, dy, dz, dt;		// cell size x, y and z dimensions//Time step
	dx = 0.389e-3;
	dy = 0.4e-3;
	dz = 0.265e-3 ;

	float courant_factor;	// courant factor
	courant_factor = 1;

    //time step size
	dt = (1 / (c*sqrt((1 / pow(dx, 2)) + (1 / pow(dy, 2)) + (1 / pow(dz, 2)))));
	dt = courant_factor * dt;

  // number of time steps
	int n_t_steps;
	n_t_steps = atoi(argv[1]);

	// pml size in each direction
  //XXX_n relates to the begining of the simulation space
  //XXX_p relates to the end of the simulation space
	int pml_x_n, pml_y_n, pml_z_n, pml_x_p, pml_y_p, pml_z_p;
	pml_x_n = 10;
	pml_y_n = 10;
	pml_z_n = 10;
	pml_x_p = 10;
	pml_y_p = 10;
	pml_z_p = 10;

	//Air buffer in each direction
  //XXX_n relates to the begining of the simulation space
  //XXX_p relates to the end of the simulation space
	int air_buff_x_n, air_buff_y_n, air_buff_z_n, air_buff_x_p, air_buff_y_p, air_buff_z_p;
	air_buff_x_n = 15;
	air_buff_y_n = 15;
	air_buff_z_n = 15;
	air_buff_x_p = 15;
	air_buff_y_p = 15;
	air_buff_z_p = 15;


  //OBJECTS DEFINITION

  // creat 3d structures such as bricks and spheres
  //number of structures
  const int brick_num = 1;

  //points in the 3d space
  float brick_min_x[brick_num], brick_min_y[brick_num], brick_min_z[brick_num], brick_max_x[brick_num], brick_max_y[brick_num], brick_max_z[brick_num];

  //sigma e(electric conduction) values in each direction
  float brick_sigma_e_x[brick_num], brick_sigma_e_y[brick_num], brick_sigma_e_z[brick_num];

  //epsilon(permissivity ) values in each direction
  float brick_eps_r_x[brick_num], brick_eps_r_y[brick_num], brick_eps_r_z[brick_num];

  //sigma m(magnetic conduction) values in each direction
  float brick_sigma_m_x[brick_num], brick_sigma_m_y[brick_num], brick_sigma_m_z[brick_num];

  //mi(permeability) values in each direction
  float brick_mu_r_x[brick_num], brick_mu_r_y[brick_num], brick_mu_r_z[brick_num];

  //flag to construct the object
  int brick_opt;

  //opt == 0, do not construct object
  brick_opt = 1;

  brick_min_x[0]= 0e-3;
  brick_min_y[0] = 0e-3;
  brick_min_z[0] = 0e-3;
  brick_max_x[0] = 60*dx;
  brick_max_y[0] = 100*dy;
  brick_max_z[0] = 3*dz;
  brick_sigma_e_x[0] = 0.0004;
  brick_sigma_e_y[0] =0.0004;
  brick_sigma_e_z[0] = 0.0004;
  brick_eps_r_x[0] = 2.2;
  brick_eps_r_y[0] = 2.2;
  brick_eps_r_z[0] = 2.2;
  brick_sigma_m_x[0] = 1.2e-38;
  brick_sigma_m_y[0] = 1.2e-38;
  brick_sigma_m_z[0] = 1.2e-38;
  brick_mu_r_x[0] = 1;
  brick_mu_r_y[0] = 1;
  brick_mu_r_z[0] = 1;

  //create 2d structurs (pec plates for the most part)
  //pecs quantity
  const int pec_num =3;

  //pec coordinates in space
  float pec_min_x[pec_num], pec_min_y[pec_num], pec_min_z[pec_num], pec_max_x[pec_num], pec_max_y[pec_num], pec_max_z[pec_num];

  //sigma e(electric conduction) values in each direction
  float pec_sigma_e_x[pec_num], pec_sigma_e_y[pec_num], pec_sigma_e_z[pec_num];

  //flag to construc the object
  //opt == 0, do not construct object
  int pec_opt;

  pec_opt = 1;

    //PEC 1
  pec_min_x[0]= 0;
  pec_min_y[0] = 0;
  pec_min_z[0] = 0;
  pec_max_x[0] = 60*dx;
  pec_max_y[0] = 100*dy;
  pec_max_z[0] = 0;
  pec_sigma_e_x[0] = 1e10;
  pec_sigma_e_y[0] = 1e10;
  pec_sigma_e_z[0] = 1e10;

  //PEC 2
  pec_min_x[1]= 7.535e-3;
  pec_min_y[1] = 0;
  pec_min_z[1] = 3*dz;
  pec_max_x[1] = 9.869e-3;
  pec_max_y[1] = 50*dy;
  pec_max_z[1] = 3*dz;
  pec_sigma_e_x[1] = 1e10;
  pec_sigma_e_y[1] = 1e10;
  pec_sigma_e_z[1] = 1e10;

  //PEC 3
  pec_min_x[2]= 5.445e-3;
  pec_min_y[2] = 50*dy;
  pec_min_z[2] = 3*dz;
  pec_max_x[2] = 17.895e-3;
  pec_max_y[2] = (50*dy) + (16e-3);
  pec_max_z[2] = 3*dz;
  pec_sigma_e_x[2] = 1e10;
  pec_sigma_e_y[2] = 1e10;
  pec_sigma_e_z[2] = 1e10;


  //create source
	//start and end of the source
	//Source coordinates in space
	float source_min_x = 7.535e-3;
	float source_min_y = 0e-3;
	float source_min_z = 0e-3;
	float source_max_x = 9.869e-3;
	float source_max_y = 0e-3;
	float source_max_z = 3*dz;

  //source type : 1 gaussian pulse, 2 gaussian derivative and 3 senoidal
  int source_tp = 1;
  //source frequency for senoidal
  float source_freq = 1e8;
  //source amplitude
  float source_amp = 1;
	//Source resistance
  float rs = 50;
	//Source direction
	int source_direction = 3;
  //Gaussian Pulse parameters
	float nc = 20;
	float tau = (nc*dy) / (2 * c);
	//tau = 15e-12;
	float t_0 = 3 * tau;

  //Create Resistor
  //resistor coordinates in space
  float resistor_min_x, resistor_min_y, resistor_min_z, resistor_max_x, resistor_max_y, resistor_max_z;
  //resistor total resistance
  float resistor_resist;
  //direction of alignment
  int resistor_direction;
  //flag to create object
  //=0 do not create
  int resistor_opt;

  resistor_opt = 0;

  resistor_min_x = 7e-3;
  resistor_min_y = 0;
  resistor_min_z = 0;
  resistor_max_x = 8e-3;
  resistor_max_y = 2e-3;
  resistor_max_z = 4e-3;
  resistor_resist = 50;
  resistor_direction = 1;

	//Define Output variables
	//Sample voltage index
	float sampled_voltage_min_x, sampled_voltage_max_x, sampled_voltage_min_y, sampled_voltage_max_y, sampled_voltage_min_z, sampled_voltage_max_z;
  //sampled voltage direction
  int voltage_direction;

	sampled_voltage_min_x = 7.535e-3;
	sampled_voltage_min_y = 10*dy;
	sampled_voltage_min_z = 0e-3;
	sampled_voltage_max_x = 9.869e-3;
	sampled_voltage_max_y = 10*dy;
	sampled_voltage_max_z = 3*dz;
  voltage_direction = 3;

	//Sampled Current Index
	float sampled_current_min_x, sampled_current_max_x, sampled_current_min_y, sampled_current_max_y, sampled_current_min_z, sampled_current_max_z;
  //sampled current direction
  int current_direction;
	//Sampling current positions
	sampled_current_min_x = 7.535e-3;
	sampled_current_min_y = 10*dy;
	sampled_current_min_z = 3*dz;
	sampled_current_max_x = 9.869e-3;
	sampled_current_max_y = 10*dy;
	sampled_current_max_z = 3*dz;
  current_direction = 2;

  //calculate size of the box
  //box, that involves all the objects, coordiantes
  float box_min_x,box_min_y,box_min_z,box_max_x,box_max_y,box_max_z;
  //number of pecs
  int pec_num2 = ((pec_opt) != 0) ? pec_num : 0;
  //number of bricks
  int brick_num2 = ((brick_opt) != 0) ? brick_num : 0;
  //number of objects in total
  int ob = pec_num2 + brick_num2;
  //auxiliars to get the points of all objects
  float m_min_x[ob],m_min_y[ob],m_min_z[ob],m_max_x[ob],m_max_y[ob],m_max_z[ob];
  float size_x, size_y, size_z;

  for(int i = 0; i < brick_num2; i++){

        m_min_x[i] = brick_min_x[i];
        m_min_y[i] = brick_min_y[i];
        m_min_z[i] = brick_min_z[i];
        m_max_x[i] = brick_max_x[i];
        m_max_y[i] = brick_max_y[i];
        m_max_z[i] = brick_max_z[i];

  }
  for(int i = brick_num2; i < ob; i++){

        m_min_x[i] = pec_min_x[i-brick_num2];
        m_min_y[i] = pec_min_y[i-brick_num2];
        m_min_z[i] = pec_min_z[i-brick_num2];
        m_max_x[i] = pec_max_x[i-brick_num2];
        m_max_y[i] = pec_max_y[i-brick_num2];
        m_max_z[i] = pec_max_z[i-brick_num2];
  }

  box_min_x = m_min_x[0];
  box_min_y = m_min_y[0];
  box_min_z = m_min_z[0];
  box_max_x = m_max_x[0];
  box_max_y  = m_max_y[0];
  box_max_z = m_max_z[0];

  for(int i=0; i < ob; i++){

      if(  m_min_x[i] < box_min_x ){
          box_min_x = m_min_x[i];
      }
      if(  m_min_y[i] < box_min_y ){
          box_min_y = m_min_y[i];
      }
      if(  m_min_z[i] < box_min_z ){
          box_min_z = m_min_z[i];
      }
      if(  m_max_x[i] < box_max_x ){
          box_max_x = m_max_x[i];
      }
      if(  m_max_z[i] < box_max_z ){
          box_max_z = m_max_z[i];
      }
      if(  m_max_z[i] < box_max_z ){
          box_max_z = m_max_z[i];
      }

  }

  size_x = box_max_x - box_min_x;
  size_y = box_max_y - box_min_y;
  size_z = box_max_z - box_min_z;

  int X = round(size_x / dx);
  int Y = round(size_y / dy);
  int Z = round(size_z / dz);


marchingLoop( numdev, eps_0,  pi ,  mu_0,  c,  dx,  dy,  dz,  dt,
     X,  Y,  Z,
     n_t_steps,
     pml_x_n,  pml_y_n,  pml_z_n,  pml_x_p,  pml_y_p,  pml_z_p,
     air_buff_x_n,  air_buff_y_n, air_buff_z_n, air_buff_x_p, air_buff_y_p, air_buff_z_p,
     source_tp, source_freq, source_amp,
     source_min_x , source_min_y ,  source_min_z,
     source_max_x,  source_max_y,  source_max_z,
     source_direction, rs,  nc,  tau,  t_0,
     brick_min_x,brick_min_y,brick_min_z,
     brick_max_x,brick_max_y,brick_max_z,
     brick_sigma_e_x, brick_sigma_e_y, brick_sigma_e_z,
     brick_eps_r_x, brick_eps_r_y, brick_eps_r_z,
     brick_sigma_m_x, brick_sigma_m_y, brick_sigma_m_z,
     brick_mu_r_x, brick_mu_r_y, brick_mu_r_z,
     brick_opt, brick_num,
     pec_min_x, pec_min_y, pec_min_z,
     pec_max_x, pec_max_y, pec_max_z,
     pec_sigma_e_x, pec_sigma_e_y, pec_sigma_e_z,
     pec_opt, pec_num,
     resistor_min_x, resistor_min_y, resistor_min_z,
     resistor_max_x, resistor_max_y, resistor_max_z,
     resistor_resist, resistor_direction, resistor_opt,
     sampled_voltage_min_x,  sampled_voltage_min_y,  sampled_voltage_min_z,
     sampled_voltage_max_x,  sampled_voltage_max_y,  sampled_voltage_max_z,
     voltage_direction,
     sampled_current_min_x,  sampled_current_min_y,  sampled_current_min_z,
     sampled_current_max_x,  sampled_current_max_y,  sampled_current_max_z,
     current_direction);

	cout << "End of Program" << endl;

	// total time of execution
	cout << "\nTotal time elapsed: " << (float)(clock() - tStart) / CLOCKS_PER_SEC << endl;


	getchar();
	return 0;
}
