#ifndef _MYLIB_H_
#define _MYLIB_H_


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

//using namespace std;
using std::cout;
using std::endl;

#define WIDTH 4

//check errors on screen
static void HandleError(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}

//helper to check errors on screen
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

//vectorial product in 2d
__host__ __device__ double pvetorial2d(double ax, double ay, double bx, double by) {

	double result;

	result = (ax*by) - (ay*bx);

	return result;
}

//scalar product in 2d
__host__ __device__ double pescalar(double ax, double ay, double bx, double by) {

	double result = 0;

	result = (ax*bx) + (ay*by);

	return result;
}

//verify if this points create a triangle
__host__ __device__ int tricheck(float pontox, float pontoy,
	float px0, float py0, float px1, float py1, float px2, float py2,
	float raio) {

	double zx1, zy1, zx2, zy2;
	double ax, ay, bx, by, cx, cy, dx, dy;

	double rvetorial1, rvetorial2, rescalar;

	int orienta = 0;

	double dist;

	ax = px1 - px0;
	ay = py1 - py0;

	bx = px2 - px1;
	by = py2 - py1;

	cx = px2 - px0;
	cy = py2 - py0;

	dx = px0 - px1;
	dy = py0 - py1;

	zx1 = pontox - px0;
	zy1 = pontoy - py0;

	zx2 = pontox - px1;
	zy2 = pontoy - py1;


	dist = ((px1 - pontox)*(px1 - pontox)) + ((py1 - pontoy)*(py1 - pontoy));
	dist = sqrt(dist);


	rvetorial1 = pvetorial2d(ax, ay, cx, cy);
	rvetorial2 = pvetorial2d(ax, ay, zx1, zy1);
	rescalar = (rvetorial1 * rvetorial2);

	if (rescalar < 0) {
		//cout << "Ponto X= " << pontox << " e Y= " << pontoy << " fora do triangulo Devido ao p1" << endl;
		orienta = -1;
		//getchar();
		return orienta;
	}

	rvetorial1 = pvetorial2d(bx, by, dx, dy);
	rvetorial2 = pvetorial2d(bx, by, zx2, zy2);
	rescalar = (rvetorial1 * rvetorial2);

	if (rescalar < 0) {
		//cout << "Ponto X= " << pontox << " e Y= " << pontoy << " fora do triangulo Devido ao pv2" << endl;
		orienta = -1;
		//getchar();
		return orienta;

	}


	if (dist > raio) {
		//cout << "Ponto X= " << pontox << " e Y= " << pontoy << " fora do triangulo Devido ao Raio" << endl;
		//cout << "Raio: " << raio << endl;
		orienta = -1;
		//getchar();
		return orienta;
	}

	//cout << "*********" << endl << "Ponto X= " << pontox << " e Y= " << pontoy << " interno ao Triangulo" << endl << "*********" << endl;
	//getchar();
	return orienta;

}

//creat a thickless sector plate
void defineSector(int NX, int NY, int NZ,
	int MAX_X, int MAX_Y,
	int px0, int py0, int px1, int py1, int px2, int py2, int pz0, float raio,
	float sigma_sector_e_x, float sigma_sector_e_y,
	float * h_sigma_e_x, float * h_sigma_e_y,
	float dx, float dy, float dz) {

	int idx;

	int min_x, min_y;
	int max_x, max_y;
	int zcoord;

	min_x = 0;
	max_x = MAX_X;

	min_y = 0;
	max_y = MAX_Y;

	zcoord = pz0;

	int pontox, pontoy;


	//update sigma_e_x
	for (int j = min_y; j <= (max_y); j++) {

		pontoy = j;

		for (int i = min_x; i < (max_x); i++) {

			pontox = i;

			if (tricheck(pontox*dx, pontoy*dy, px0*dx, py0*dy, px1*dx, py1*dy, px2*dx, py2*dy, raio) == 0) {

				//3d to 1d index
				idx = i + j * NX + zcoord * NX * NY;

				h_sigma_e_x[idx] = sigma_sector_e_x;
			}

		}
	}

	//update sigma_e_y
	for (int j = min_y; j < (max_y); j++) {

		pontoy = j;

		for (int i = min_x; i <= (max_x); i++) {

			pontox = i;

			if (tricheck(pontox*dx, pontoy*dy, px0*dx, py0*dy, px1*dx, py1*dy, px2*dx, py2*dy, raio) == 0) {

				//3d to 1d index
				idx = i + j * NX + zcoord * NX * NY;

				h_sigma_e_y[idx] = sigma_sector_e_y;
			}

		}
	}

}

//calculate the index of the point in the grid
int __host__ __device__ calc_index(float coordinate, float dxyz) {

	int index;

	index = (int)round((coordinate - 0) / dxyz);
	return index;


}

//transpose a point
float trans_point(float pt, float delta) {

	float pt_novo;

	pt_novo = pt - delta;

	return pt_novo;

}

//rotate a point
float2 rotate_point(float2 pt, float angle_rad) {

	float2 pt_novo;

	//pt_novo.x = pt.x * cos(angle_rad) - pt.y * sin(angle_rad);
	//pt_novo.y = pt.x * sin(angle_rad) + pt.y * cos(angle_rad);

	pt_novo.x = pt.x * cosf(angle_rad) + pt.y * sinf(angle_rad);
	pt_novo.y = ((-1)*pt.x) * sinf(angle_rad) + pt.y * cosf(angle_rad);

	return pt_novo;
}

//sampling field
__host__ __device__ void sampleField(int i, int j, int k, int k_real, int m, int volt_NX, int volt_NY, int volt_NZ_N, int volt_NZ, int offset, int threadId,
                                     int sampled_is, int sampled_js, int sampled_ks, int sampled_ie, int sampled_je, int sampled_ke,
                                     float * q, float * qSamp){

    if ((i >= (sampled_is - offset)) && (i <= (sampled_ie)) && (j >= (sampled_js - offset)) && (j <= (sampled_je)) && (k_real >= (sampled_ks - offset)) && (k_real <= (sampled_ke)) ) {


        qSamp[(( i - (sampled_is - offset)) + (j- (sampled_js - offset) )*volt_NX + (k * volt_NX * volt_NY) + (m * volt_NX * volt_NY * volt_NZ_N))] = q[threadId];


    }

}

//Kernel to calculate The Magnetic field
__global__ void calc_h(int NX, int NXX, int NY, int NYY, int NZ, int NZ_N,
	int HA, int HB, int gpu_offset,
	int pml_x_n, int pml_x_p, int pml_y_n, int pml_y_p, int pml_z_n, int pml_z_p,
	float * d_Ex, float * d_Jx,
	float * d_Ey, float * d_Jy,
	float * d_Ez, float * d_Jz,
	float * d_gEx, float * d_gEy,
	float * d_Hx, float * d_Mx, float * d_Chxh, float * d_Chxey, float * d_Chxez, float * d_Chxm,
	float * d_Hy, float * d_My, float * d_Chyh, float * d_Chyez, float * d_Chyex, float * d_Chym,
	float * d_Hz, float * d_Mz, float * d_Chzh, float * d_Chzex, float * d_Chzey, float * d_Chzm,
	float * d_cpml_b_mx, float * d_cpml_a_mx,
	float * d_cpml_b_my, float * d_cpml_a_my,
	float * d_cpml_b_mz, float * d_cpml_a_mz,
	float * d_Psi_eyx, float * d_Psi_ezx, float * d_Psi_hyx, float * d_Psi_hzx,
	float * d_cpsi_eyx, float * d_cpsi_ezx, float * d_cpsi_hyx, float * d_cpsi_hzx,
	float * d_Psi_exy, float * d_Psi_ezy, float * d_Psi_hxy, float * d_Psi_hzy,
	float * d_cpsi_exy, float * d_cpsi_ezy, float * d_cpsi_hxy, float * d_cpsi_hzy,
	float * d_Psi_exz, float * d_Psi_eyz, float * d_Psi_hxz, float * d_Psi_hyz,
	float * d_cpsi_exz, float * d_cpsi_eyz, float * d_cpsi_hxz, float * d_cpsi_hyz,
	int sampled_current_is, int sampled_current_js, int sampled_current_ks, int sampled_current_ie, int sampled_current_je, int sampled_current_ke,
	int current_NX, int current_NY, int current_NZ_N,
	int m, float * Hx, float * Hy, float * Hz) {


	//helper to calculate threads global index
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;

	//threads global index
	int threadId = i + j * NX + k * NX * NY;

	int k_real = k + gpu_offset;

	if ((threadId < (NX*NY*NZ_N)) && ((k > HA) && (k < HB) )) {
 


		// (i,j,k) coordinates of the position of the calculated vector in space
		//int i, j, k;

		// auxiliar to calculate the coordinates i,j,k of the vector
		//int aux_threadId;

		//Mapping 3d to 1d the new index reffered
		// and to E(i,j,k+1)
		// and to E(i,j+1,k)
		// and to E(i+1,j,k)
		int e_threadId_k, e_threadId_j, e_threadId_i;
        //index refering to the ghost node
        int e_ghost_k;

		// calculate the i coordinate for Ex
		//k = threadId / (NX * NY);

		// update the auxiliar to calculate j and k for Ex
		//aux_threadId = threadId - (k * NX * NY);

		// calculate the j coordinate for Ex
		//j = aux_threadId / NX;

		// calculate the k coordinate for Ex
		//i = aux_threadId % NX;

		// calculate the index refered to E(i+1,j,k)
		e_threadId_i = threadId + 1;

		// calculate the index refered to E(i,j+1,k)
		e_threadId_j = threadId + NX;

		// calculate the index refered to E(i+1,j,k)
		e_threadId_k = threadId + (NX * NY);

        // calculate the index refered to Eghost(i,j,k = 0)
        e_ghost_k = i + j * NX;


		//**HX UPTADE
		// check for boundaries
		if ((i < NXX) && (j < (NYY - 1)) && (k_real < (NZ - 1))) {
			
			//update Hx
			d_Hx[threadId] = (d_Chxh[threadId] * d_Hx[threadId]) + (d_Chxey[threadId] * (d_Ey[e_threadId_k] - d_Ey[threadId])) + (d_Chxez[threadId] * (d_Ez[e_threadId_j] - d_Ez[threadId]));

			//synchronize threads
            __syncthreads();

			//sample Hx field
            sampleField( i, j, k, k_real, m,current_NX, current_NY, current_NZ_N, NZ, 1, threadId, sampled_current_is, sampled_current_js, sampled_current_ks, sampled_current_ie, sampled_current_je, sampled_current_ke,  d_Hx, Hx);

		}
		//synchronize the threads
		__syncthreads();

		//**HY UPTADE
		// check for boundaries
		if ((i < (NXX - 1)) && (j < (NYY)) && (k_real < (NZ - 1))) {
			
			//update Hy
			d_Hy[threadId] = (d_Chyh[threadId] * d_Hy[threadId]) + (d_Chyez[threadId] * (d_Ez[e_threadId_i] - d_Ez[threadId])) + (d_Chyex[threadId] * (d_Ex[e_threadId_k] - d_Ex[threadId]));
 
			//synchronize threads
            __syncthreads();

            //sampling Hy field
            sampleField( i, j, k, k_real, m,current_NX, current_NY, current_NZ_N, NZ, 1, threadId, sampled_current_is, sampled_current_js, sampled_current_ks, sampled_current_ie, sampled_current_je, sampled_current_ke,  d_Hy, Hy);

		}
		//synchronize the threads
		__syncthreads();


		//**HZ UPTADE
		// check for boundaries
		if ((i < (NXX - 1)) && (j < (NYY - 1)) && (k_real < (NZ))) {

			//uptade Hz
			d_Hz[threadId] = (d_Chzh[threadId] * d_Hz[threadId]) + (d_Chzex[threadId] * (d_Ex[e_threadId_j] - d_Ex[threadId])) + (d_Chzey[threadId] * (d_Ey[e_threadId_i] - d_Ey[threadId]));
			
            //sampling Hz field
            __syncthreads();

			//sampling Hz
            sampleField( i, j, k, k_real, m,current_NX, current_NY, current_NZ_N, NZ, 1, threadId, sampled_current_is, sampled_current_js, sampled_current_ks, sampled_current_ie, sampled_current_je, sampled_current_ke,  d_Hz, Hz);

		}
		//synchronize the threads
		__syncthreads();

		//CPML ADJUST*****************************************************************************

		//CPML at the x_n region. Update Hy and Hz
		if ((i < pml_x_n) && (j < (NYY)) && (k_real < NZ)) {

			//cpml factors
			d_Psi_hyx[threadId] = d_cpml_b_mx[i] * d_Psi_hyx[threadId] + d_cpml_a_mx[i] * (d_Ez[e_threadId_i] - d_Ez[threadId]);

			//cpml factors
			d_Psi_hzx[threadId] = d_cpml_b_mx[i] * d_Psi_hzx[threadId] + d_cpml_a_mx[i] * (d_Ey[e_threadId_i] - d_Ey[threadId]);

			if (k_real < (NZ - 1)) {

				//update Hy
				d_Hy[threadId] = d_Hy[threadId] + d_cpsi_hyx[threadId] * d_Psi_hyx[threadId];

			}
			//synchronize the threads
			__syncthreads();
			if (j < (NYY - 1)) {
			
				//update Hz
				d_Hz[threadId] = d_Hz[threadId] + d_cpsi_hzx[threadId] * d_Psi_hzx[threadId];

			}
		}
		//synchronize the threads
		__syncthreads();

		//CPML at the x_p region. Update Hy and Hz
		if ((i > (NXX - pml_x_p - 1)) && (i < (NXX - 1)) && (j < (NYY)) && (k_real < NZ)) {

			//cpml factors
			d_Psi_hyx[threadId] = d_cpml_b_mx[i] * d_Psi_hyx[threadId] + d_cpml_a_mx[i] * (d_Ez[e_threadId_i] - d_Ez[threadId]);

			//cpml factors
			d_Psi_hzx[threadId] = d_cpml_b_mx[i] * d_Psi_hzx[threadId] + d_cpml_a_mx[i] * (d_Ey[e_threadId_i] - d_Ey[threadId]);

			if (k_real < (NZ - 1)) {

				//update Hy
				d_Hy[threadId] = d_Hy[threadId] + d_cpsi_hyx[threadId] * d_Psi_hyx[threadId];

			}
			//synchronize the threads
			__syncthreads();
			if (j < (NYY - 1)) {

				//update Hz
				d_Hz[threadId] = d_Hz[threadId] + d_cpsi_hzx[threadId] * d_Psi_hzx[threadId];

			}

		}
		//synchronize the threads
		__syncthreads();

		//CPML at the y_n region. Update Hx and Hz
		if ((i < (NXX)) && (j < (pml_y_n)) && (k_real < NZ)) {
			
			//cpml factor
			d_Psi_hxy[threadId] = d_cpml_b_my[j] * d_Psi_hxy[threadId] + d_cpml_a_my[j] * (d_Ez[e_threadId_j] - d_Ez[threadId]);
			
			//cpml factor
			d_Psi_hzy[threadId] = d_cpml_b_my[j] * d_Psi_hzy[threadId] + d_cpml_a_my[j] * (d_Ex[e_threadId_j] - d_Ex[threadId]);

			if (k_real < (NZ - 1)) {

				//calc Hx
				d_Hx[threadId] = d_Hx[threadId] + d_cpsi_hxy[threadId] * d_Psi_hxy[threadId];

			}
			//synchronize the threads
			__syncthreads();
			if (i < (NXX - 1)) {
				
				//calc Hz
				d_Hz[threadId] = d_Hz[threadId] + d_cpsi_hzy[threadId] * d_Psi_hzy[threadId];

			}

		}
		//synchronize the threads
		__syncthreads();

		//CPML at the y_p region. Update Hx and Hz
		if ((j > (NYY - pml_y_p - 1)) && (i < (NXX)) && (j < (NYY - 1)) && (k_real < NZ)) {

			//cpml factor
			d_Psi_hxy[threadId] = d_cpml_b_my[j] * d_Psi_hxy[threadId] + d_cpml_a_my[j] * (d_Ez[e_threadId_j] - d_Ez[threadId]);
			
			//cpml factor
			d_Psi_hzy[threadId] = d_cpml_b_my[j] * d_Psi_hzy[threadId] + d_cpml_a_my[j] * (d_Ex[e_threadId_j] - d_Ex[threadId]);

			if (k_real < (NZ - 1)) {

				//calc Hx
				d_Hx[threadId] = d_Hx[threadId] + d_cpsi_hxy[threadId] * d_Psi_hxy[threadId];

			}
			//synchronize the threads
			__syncthreads();
			if (i < (NXX - 1)) {
				
				//calc Hz
				d_Hz[threadId] = d_Hz[threadId] + d_cpsi_hzy[threadId] * d_Psi_hzy[threadId];

			}
		}
		//synchronize the threads
		__syncthreads();

		//CPML at the z_n region. Update Hx and Hy
		if ((i < (NXX)) && (j < (NYY)) && (k_real < (pml_z_n))) {

			//cpml factor
	        d_Psi_hxz[threadId] = d_cpml_b_mz[k] * d_Psi_hxz[threadId] + d_cpml_a_mz[k] * (d_Ey[e_threadId_k] - d_Ey[threadId]);

            //cpml factor    
			d_Psi_hyz[threadId] = d_cpml_b_mz[k] * d_Psi_hyz[threadId] + d_cpml_a_mz[k] * (d_Ex[e_threadId_k] - d_Ex[threadId]);

			if (j < (NYY - 1)) {

				//calc Hx
				d_Hx[threadId] = d_Hx[threadId] + d_cpsi_hxz[threadId] * d_Psi_hxz[threadId];

			}
			//synchronize the threads
			__syncthreads();
			if (i < (NXX - 1)) {
				
				//calc Hy
				d_Hy[threadId] = d_Hy[threadId] + d_cpsi_hyz[threadId] * d_Psi_hyz[threadId];

			}
		}
		//synchronize the threads
		__syncthreads();

		//CPML at the z_p region. Update Hx and Hy
		if ((k_real > (NZ - pml_z_p - 1)) && (i < (NXX)) && (j < (NYY)) && (k_real < (NZ - 1))) {
			
			//cpml factors
			d_Psi_hxz[threadId] = d_cpml_b_mz[k] * d_Psi_hxz[threadId] + d_cpml_a_mz[k] * (d_Ey[e_threadId_k] - d_Ey[threadId]);

			//cpml factors
            d_Psi_hyz[threadId] = d_cpml_b_mz[k] * d_Psi_hyz[threadId] + d_cpml_a_mz[k] * (d_Ex[e_threadId_k] - d_Ex[threadId]);

			if (j < (NYY - 1)) {

				//calc Hx
				d_Hx[threadId] = d_Hx[threadId] + d_cpsi_hxz[threadId] * d_Psi_hxz[threadId];

			}
			//synchronize the threads
			__syncthreads();
			if (i < (NXX - 1)) {

				//calc Hy
				d_Hy[threadId] = d_Hy[threadId] + d_cpsi_hyz[threadId] * d_Psi_hyz[threadId];

			}

		}
		__syncthreads();
	}

}

//Kernel to Calculate the Magnetic Field on the Critical Points
__global__ void calc_hA(int NX, int NXX, int NY, int NYY, int NZ, int NZ_N, int gpu_offset, int A
	int pml_x_n, int pml_x_p, int pml_y_n, int pml_y_p, int pml_z_n, int pml_z_p,
	float* d_Ex, float* d_Jx,
	float* d_Ey, float* d_Jy,
	float* d_Ez, float* d_Jz,
	float* d_gEx, float* d_gEy,
	float* d_Hx, float* d_Mx, float* d_Chxh, float* d_Chxey, float* d_Chxez, float* d_Chxm,
	float* d_Hy, float* d_My, float* d_Chyh, float* d_Chyez, float* d_Chyex, float* d_Chym,
	float* d_Hz, float* d_Mz, float* d_Chzh, float* d_Chzex, float* d_Chzey, float* d_Chzm,
	float* d_cpml_b_mx, float* d_cpml_a_mx,
	float* d_cpml_b_my, float* d_cpml_a_my,
	float* d_cpml_b_mz, float* d_cpml_a_mz,
	float* d_Psi_eyx, float* d_Psi_ezx, float* d_Psi_hyx, float* d_Psi_hzx,
	float* d_cpsi_eyx, float* d_cpsi_ezx, float* d_cpsi_hyx, float* d_cpsi_hzx,
	float* d_Psi_exy, float* d_Psi_ezy, float* d_Psi_hxy, float* d_Psi_hzy,
	float* d_cpsi_exy, float* d_cpsi_ezy, float* d_cpsi_hxy, float* d_cpsi_hzy,
	float* d_Psi_exz, float* d_Psi_eyz, float* d_Psi_hxz, float* d_Psi_hyz,
	float* d_cpsi_exz, float* d_cpsi_eyz, float* d_cpsi_hxz, float* d_cpsi_hyz,
	int sampled_current_is, int sampled_current_js, int sampled_current_ks, int sampled_current_ie, int sampled_current_je, int sampled_current_ke,
	int current_NX, int current_NY, int current_NZ_N,
	int m, float* Hx, float* Hy, float* Hz) {


	//helper to calculate threads global index
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;

	//threads global index
	int threadId = i + j * NX + k * NX * NY;

	int k_real = k + gpu_offset;

	//if ((threadId < (NX*NY*NZ_N)) && ((k > HA) && (k < HB) )) {
	if (threadId < (NX * NY * NZ_N)) {


		// (i,j,k) coordinates of the position of the calculated vector in space
		//int i, j, k;

		// auxiliar to calculate the coordinates i,j,k of the vector
		//int aux_threadId;

		//Mapping 3d to 1d the new index reffered
		// and to E(i,j,k+1)
		// and to E(i,j+1,k)
		// and to E(i+1,j,k)
		int e_threadId_k, e_threadId_j, e_threadId_i;
		//index refering to the ghost node
		int e_ghost_k;

		// calculate the i coordinate for Ex
		//k = threadId / (NX * NY);

		// update the auxiliar to calculate j and k for Ex
		//aux_threadId = threadId - (k * NX * NY);

		// calculate the j coordinate for Ex
		//j = aux_threadId / NX;

		// calculate the k coordinate for Ex
		//i = aux_threadId % NX;

		// calculate the index refered to E(i+1,j,k)
		e_threadId_i = threadId + 1;

		// calculate the index refered to E(i,j+1,k)
		e_threadId_j = threadId + NX;

		// calculate the index refered to E(i+1,j,k)
		e_threadId_k = threadId + (NX * NY);

		// calculate the index refered to Eghost(i,j,k = 0)
		e_ghost_k = i + j * NX;


		//**HX UPTADE
		// check for boundaries
		//if ((i < NX) && (j < (NY - 1)) && (k < (NZ - 1))) {
		if ((i < NXX) && (j < (NYY - 1)) && (k_real < (NZ - 1))) {

			//update Hx
			if (k == (NZ_N - 1)) {

				//update Hx
				d_Hx[threadId] = (d_Chxh[threadId] * d_Hx[threadId]) + (d_Chxey[threadId] * (d_gEy[e_ghost_k] - d_Ey[threadId])) + (d_Chxez[threadId] * (d_Ez[e_threadId_j] - d_Ez[threadId]));

			}
			else {

				d_Hx[threadId] = (d_Chxh[threadId] * d_Hx[threadId]) + (d_Chxey[threadId] * (d_Ey[e_threadId_k] - d_Ey[threadId])) + (d_Chxez[threadId] * (d_Ez[e_threadId_j] - d_Ez[threadId]));
				//d_Hx[threadId] = (d_Chxh[threadId] * d_Hx[threadId]) + (d_Chxey[threadId] * (d_Ey[e_threadId_k] - s_Ey_Ez[tz][ty][tx].x)) + (d_Chxez[threadId] * (d_Ez[e_threadId_j] - s_Ey_Ez[tz][ty][tx].y));

			}

			//synchronize the threads
			__syncthreads();

			//sample Hx field
			sampleField(i, j, k, k_real, m, current_NX, current_NY, current_NZ_N, NZ, 1, threadId, sampled_current_is, sampled_current_js, sampled_current_ks, sampled_current_ie, sampled_current_je, sampled_current_ke, d_Hx, Hx);


		}
		//synchronize the threads
		__syncthreads();

		//**HY UPTADE
		// check for boundaries
		//if ((i < (NX - 1)) && (j < (NY)) && (k < (NZ - 1))) {
		if ((i < (NXX - 1)) && (j < (NYY)) && (k_real < (NZ - 1))) {

			//update Hy
			if (k == (NZ_N - 1)) {

				d_Hy[threadId] = (d_Chyh[threadId] * d_Hy[threadId]) + (d_Chyez[threadId] * (d_Ez[e_threadId_i] - d_Ez[threadId])) + (d_Chyex[threadId] * (d_gEx[e_ghost_k] - d_Ex[threadId]));

			}
			else {

				d_Hy[threadId] = (d_Chyh[threadId] * d_Hy[threadId]) + (d_Chyez[threadId] * (d_Ez[e_threadId_i] - d_Ez[threadId])) + (d_Chyex[threadId] * (d_Ex[e_threadId_k] - d_Ex[threadId]));
				//d_Hy[threadId] = (d_Chyh[threadId] * d_Hy[threadId]) + (d_Chyez[threadId] * (s_Ey_Ez[tz][ty][tx + 1].y - s_Ey_Ez[tz][ty][tx].y)) + (d_Chyex[threadId] * (d_Ex[e_threadId_k] - d_Ex[threadId]));

			}
			__syncthreads();
			//sampling Hy field
			sampleField(i, j, k, k_real, m, current_NX, current_NY, current_NZ_N, NZ, 1, threadId, sampled_current_is, sampled_current_js, sampled_current_ks, sampled_current_ie, sampled_current_je, sampled_current_ke, d_Hy, Hy);

		}
		//synchronize the threads
		__syncthreads();


		//**HZ UPTADE
		// check for boundaries
		//if ((i < (NX - 1)) && (j < (NY - 1)) && (k < (NZ))) {
		if ((i < (NXX - 1)) && (j < (NYY - 1)) && (k_real < (NZ))) {

			//uptade Hz
			d_Hz[threadId] = (d_Chzh[threadId] * d_Hz[threadId]) + (d_Chzex[threadId] * (d_Ex[e_threadId_j] - d_Ex[threadId])) + (d_Chzey[threadId] * (d_Ey[e_threadId_i] - d_Ey[threadId]));
			//d_Hz[threadId] = (d_Chzh[threadId] * d_Hz[threadId]) + (d_Chzex[threadId] * (d_Ex[e_threadId_j] - d_Ex[threadId])) + (d_Chzey[threadId] * (s_Ey_Ez[tz][ty][tx + 1].x - s_Ey_Ez[tz][ty][tx].x));


			//sampling Hz field
			__syncthreads();
			sampleField(i, j, k, k_real, m, current_NX, current_NY, current_NZ_N, NZ, 1, threadId, sampled_current_is, sampled_current_js, sampled_current_ks, sampled_current_ie, sampled_current_je, sampled_current_ke, d_Hz, Hz);

		}
		//synchronize the threads
		__syncthreads();


		//CPML ADJUST*****************************************************************************

		//CPML at the x_n region. Update Hy and Hz
		if ((i < pml_x_n) && (j < (NYY)) && (k_real < NZ)) {
			//if ((i < pml_x_n) && (j < (NY)) && (k < NZ)) {

			d_Psi_hyx[threadId] = d_cpml_b_mx[i] * d_Psi_hyx[threadId] + d_cpml_a_mx[i] * (d_Ez[e_threadId_i] - d_Ez[threadId]);

			d_Psi_hzx[threadId] = d_cpml_b_mx[i] * d_Psi_hzx[threadId] + d_cpml_a_mx[i] * (d_Ey[e_threadId_i] - d_Ey[threadId]);

			if (k_real < (NZ - 1)) {

				d_Hy[threadId] = d_Hy[threadId] + d_cpsi_hyx[threadId] * d_Psi_hyx[threadId];

			}
			if (j < (NYY - 1)) {
				//if (j < (NY - 1)) {

				d_Hz[threadId] = d_Hz[threadId] + d_cpsi_hzx[threadId] * d_Psi_hzx[threadId];

			}
		}
		//synchronize the threads
		__syncthreads();

		//CPML at the x_p region. Update Hy and Hz
		if ((i > (NXX - pml_x_p - 1)) && (i < (NXX - 1)) && (j < (NYY)) && (k_real < NZ)) {
			//if ((i >(NX - pml_x_p - 1)) && (i < (NX - 1)) && (j < (NY)) && (k < NZ)) {

			d_Psi_hyx[threadId] = d_cpml_b_mx[i] * d_Psi_hyx[threadId] + d_cpml_a_mx[i] * (d_Ez[e_threadId_i] - d_Ez[threadId]);

			d_Psi_hzx[threadId] = d_cpml_b_mx[i] * d_Psi_hzx[threadId] + d_cpml_a_mx[i] * (d_Ey[e_threadId_i] - d_Ey[threadId]);

			if (k_real < (NZ - 1)) {

				d_Hy[threadId] = d_Hy[threadId] + d_cpsi_hyx[threadId] * d_Psi_hyx[threadId];

			}
			if (j < (NYY - 1)) {
				//if (j < (NY - 1)) {
				d_Hz[threadId] = d_Hz[threadId] + d_cpsi_hzx[threadId] * d_Psi_hzx[threadId];

			}

		}
		//synchronize the threads
		__syncthreads();

		//CPML at the y_n region. Update Hx and Hz
		if ((i < (NXX)) && (j < (pml_y_n)) && (k_real < NZ)) {
			//if ((i < (NX)) && (j < (pml_y_n)) && (k < NZ)) {

			d_Psi_hxy[threadId] = d_cpml_b_my[j] * d_Psi_hxy[threadId] + d_cpml_a_my[j] * (d_Ez[e_threadId_j] - d_Ez[threadId]);
			//d_Psi_hxy[threadId] = d_cpml_b_my[j] * d_Psi_hxy[threadId] + d_cpml_a_my[j] * (d_Ez[e_threadId_j] - s_Ey_Ez[tz][ty][tx].y);

			d_Psi_hzy[threadId] = d_cpml_b_my[j] * d_Psi_hzy[threadId] + d_cpml_a_my[j] * (d_Ex[e_threadId_j] - d_Ex[threadId]);


			if (k_real < (NZ - 1)) {

				d_Hx[threadId] = d_Hx[threadId] + d_cpsi_hxy[threadId] * d_Psi_hxy[threadId];

			}
			if (i < (NXX - 1)) {
				//if (i < (NX - 1)) {

				d_Hz[threadId] = d_Hz[threadId] + d_cpsi_hzy[threadId] * d_Psi_hzy[threadId];

			}

		}
		//synchronize the threads
		__syncthreads();

		//CPML at the y_p region. Update Hx and Hz
		if ((j > (NYY - pml_y_p - 1)) && (i < (NXX)) && (j < (NYY - 1)) && (k_real < NZ)) {
			//if ((j > (NY - pml_y_p - 1)) && (i < (NX)) && (j < (NY - 1)) && (k < NZ)) {

			d_Psi_hxy[threadId] = d_cpml_b_my[j] * d_Psi_hxy[threadId] + d_cpml_a_my[j] * (d_Ez[e_threadId_j] - d_Ez[threadId]);
			//d_Psi_hxy[threadId] = d_cpml_b_my[j] * d_Psi_hxy[threadId] + d_cpml_a_my[j] * (d_Ez[e_threadId_j] - s_Ey_Ez[tz][ty][tx].y);

			d_Psi_hzy[threadId] = d_cpml_b_my[j] * d_Psi_hzy[threadId] + d_cpml_a_my[j] * (d_Ex[e_threadId_j] - d_Ex[threadId]);


			if (k_real < (NZ - 1)) {

				d_Hx[threadId] = d_Hx[threadId] + d_cpsi_hxy[threadId] * d_Psi_hxy[threadId];

			}
			if (i < (NXX - 1)) {
				//if (i < (NX - 1)) {

				d_Hz[threadId] = d_Hz[threadId] + d_cpsi_hzy[threadId] * d_Psi_hzy[threadId];

			}
		}
		//synchronize the threads
		__syncthreads();

		//CPML at the z_n region. Update Hx and Hy
		if ((i < (NXX)) && (j < (NYY)) && (k_real < (pml_z_n))) {
			//if ((i < (NX)) && (j < (NY)) && (k < (pml_z_n))) {

			if (k == (NZ_N - 1)) {

				d_Psi_hxz[threadId] = d_cpml_b_mz[k] * d_Psi_hxz[threadId] + d_cpml_a_mz[k] * (d_gEy[e_ghost_k] - d_Ey[threadId]);

				d_Psi_hyz[threadId] = d_cpml_b_mz[k] * d_Psi_hyz[threadId] + d_cpml_a_mz[k] * (d_gEx[e_ghost_k] - d_Ex[threadId]);

			}
			else {

				d_Psi_hxz[threadId] = d_cpml_b_mz[k] * d_Psi_hxz[threadId] + d_cpml_a_mz[k] * (d_Ey[e_threadId_k] - d_Ey[threadId]);

				d_Psi_hyz[threadId] = d_cpml_b_mz[k] * d_Psi_hyz[threadId] + d_cpml_a_mz[k] * (d_Ex[e_threadId_k] - d_Ex[threadId]);

			}

			if (j < (NYY - 1)) {
				//if (j < (NY - 1)) {

				d_Hx[threadId] = d_Hx[threadId] + d_cpsi_hxz[threadId] * d_Psi_hxz[threadId];

			}
			if (i < (NXX - 1)) {
				//if (i < (NX - 1)) {

				d_Hy[threadId] = d_Hy[threadId] + d_cpsi_hyz[threadId] * d_Psi_hyz[threadId];

			}
		}
		//synchronize the threads
		__syncthreads();

		//CPML at the z_p region. Update Hx and Hy
		if ((k_real > (NZ - pml_z_p - 1)) && (i < (NXX)) && (j < (NYY)) && (k_real < (NZ - 1))) {
			//if ((k >(NZ - pml_z_p - 1)) && (i < (NX)) && (j < (NY)) && (k < (NZ - 1))) {

			if (k == (NZ_N - 1)) {

				d_Psi_hxz[threadId] = d_cpml_b_mz[k] * d_Psi_hxz[threadId] + d_cpml_a_mz[k] * (d_gEy[e_ghost_k] - d_Ey[threadId]);

				d_Psi_hyz[threadId] = d_cpml_b_mz[k] * d_Psi_hyz[threadId] + d_cpml_a_mz[k] * (d_gEx[e_ghost_k] - d_Ex[threadId]);

			}
			else {

				d_Psi_hxz[threadId] = d_cpml_b_mz[k] * d_Psi_hxz[threadId] + d_cpml_a_mz[k] * (d_Ey[e_threadId_k] - d_Ey[threadId]);

				d_Psi_hyz[threadId] = d_cpml_b_mz[k] * d_Psi_hyz[threadId] + d_cpml_a_mz[k] * (d_Ex[e_threadId_k] - d_Ex[threadId]);

			}

			if (j < (NYY - 1)) {
				//if (j < (NY - 1)) {

				d_Hx[threadId] = d_Hx[threadId] + d_cpsi_hxz[threadId] * d_Psi_hxz[threadId];

			}
			if (i < (NXX - 1)) {
				//if (i < (NX - 1)) {

				d_Hy[threadId] = d_Hy[threadId] + d_cpsi_hyz[threadId] * d_Psi_hyz[threadId];

			}

		}
		__syncthreads();

		//sampleField( i, j, k, k_real, m,current_NX, current_NY, current_NZ_N, NZ, 1, k_offset, threadId, sampled_current_is, sampled_current_js, sampled_current_ks, sampled_current_ie, sampled_current_je, sampled_curret_ke,  d_Hx, Hx);

		//sampleField( i, j, k, k_real, m,current_NX, current_NY, current_NZ_N, NZ, 1, k_offset, threadId, sampled_current_is, sampled_current_js, sampled_current_ks, sampled_current_ie, sampled_current_je, sampled_current_ke,  d_Hy, Hy);

		//sampleField( i, j, k, k_real, m,current_NX, current_NY, current_NZ_N, NZ, 1, k_offset, threadId, sampled_current_is, sampled_current_js, sampled_current_ks, sampled_current_ie, sampled_current_je, sampled_current_ke,  d_Hz, Hz);

	}

}




//Kernel to calculate The Electric field
__global__ void calc_e(int NX, int NXX, int NY, int NYY, int NZ, int NZ_N, int gpu_offset, int volt_offset,
	int pml_x_n, int pml_x_p, int pml_y_n, int pml_y_p, int pml_z_n, int pml_z_p,
	float * d_Ex, float * d_Jx, float * d_Cexe, float * d_Cexhz, float * d_Cexhy, float * d_Cexj,
	float * d_Ey, float * d_Jy, float * d_Ceye, float * d_Ceyhx, float * d_Ceyhz, float * d_Ceyj,
	float * d_Ez, float * d_Jz, float * d_Ceze, float * d_Cezhy, float * d_Cezhx, float * d_Cezj,
	float * d_Hx, float * d_Mx,
	float * d_Hy, float * d_My,
	float * d_Hz, float * d_Mz,
    float * d_gHx, float * d_gHy,
	float * d_cpml_b_ex, float * d_cpml_a_ex,
	float * d_cpml_b_ey, float * d_cpml_a_ey,
	float * d_cpml_b_ez, float * d_cpml_a_ez,
	float * d_Psi_eyx, float * d_Psi_ezx, float * d_Psi_hyx, float * d_Psi_hzx,
	float * d_cpsi_eyx, float * d_cpsi_ezx, float * d_cpsi_hyx, float * d_cpsi_hzx,
	float * d_Psi_exy, float * d_Psi_ezy, float * d_Psi_hxy, float * d_Psi_hzy,
	float * d_cpsi_exy, float * d_cpsi_ezy, float * d_cpsi_hxy, float * d_cpsi_hzy,
	float * d_Psi_exz, float * d_Psi_eyz, float * d_Psi_hxz, float * d_Psi_hyz,
	float * d_cpsi_exz, float * d_cpsi_eyz, float * d_cpsi_hxz, float * d_cpsi_hyz,
	float * d_signal_per_node, int source_is, int source_js,
	int source_ks, int source_ie, int source_je, int source_ke,
	int sampled_voltage_is, int sampled_voltage_js, int sampled_voltage_ks, int sampled_voltage_ie, int sampled_voltage_je, int sampled_voltage_ke,
	int volt_NX, int volt_NY, int volt_NZ_N,
	int m, float * E) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;

	//index of current thread
	int threadId = i + j * NX + k * NX * NY;

	//real value of k from the simulation space
	int k_real = k + gpu_offset;


    //index of current thread to voltage
    //int threadIdVolt;

	if ( threadId < (NX*NY*NZ_N) ) {


		// (i,j,k) coordinates of the position of the calculated vector in space
		//int i, j, k;

		// auxiliar to calculate the coordinates i,j,k of the vector
		//int aux_threadId;

		//Mapping 3d to 1d the new index reffered to
		// and to H(i-1,j,k)
		// and to H(i,j-1,k)
		// and to H(i,j,k-1)
		int d_threadId_j, d_threadId_k, d_threadId_i;
        // ghost node index
        int d_ghost_k;

		// calculate the i coordinate for Ex
		//k = threadId / (NX * NY);

		// update the auxiliar to calculate j and k for Ex
		//aux_threadId = threadId - (k * NX * NY);

		// calculate the j coordinate for Ex
		//j = aux_threadId / NX;

		// calculate the k coordinate for Ex
		//i = aux_threadId % NX;

		//calculate the index refered to Hz(i -1, j, k)
		d_threadId_i = threadId - 1;

		// calculate the index refered to Hz(i,j-1,k)
		d_threadId_j = threadId - NX;

		// calculate the index refered to H(i-1,j,k)
		d_threadId_k = threadId - (NX * NY);

        //ghost node index
        d_ghost_k =  i + j * NX;


		//synchronize threads
		//__syncthreads();


		//**EX UPTADE
		// check for borders
		// PEC boundaries at the borders
		//if ((i < (NX - 1)) && (j < (NY - 1)) && (k < (NZ - 1)) && (j > 0) && (k > 0)) {
		if ((i < (NXX - 1)) && (j < (NYY - 1)) && (k_real < (NZ - 1)) && (j > 0) && (k_real > 0)) {

			//update Ex
            if(k == (0)){

                 d_Ex[threadId] = (d_Cexe[threadId] * d_Ex[threadId]) + (d_Cexhz[threadId] * (d_Hz[threadId] - d_Hz[d_threadId_j])) + (d_Cexhy[threadId] * (d_Hy[threadId] - d_gHy[d_ghost_k]));

            } else {

                d_Ex[threadId] = (d_Cexe[threadId] * d_Ex[threadId]) + (d_Cexhz[threadId] * (d_Hz[threadId] - d_Hz[d_threadId_j])) + (d_Cexhy[threadId] * (d_Hy[threadId] - d_Hy[d_threadId_k]));
                //d_Ex[threadId] = (d_Cexe[threadId] * d_Ex[threadId]) + (d_Cexhz[threadId] * (s_Hy_Hz[tz][ty][tx + WIDTH*WIDTH].y - d_Hz[d_threadId_j])) + (d_Cexhy[threadId] * (s_Hy_Hz[tz][ty][tx + WIDTH*WIDTH].x - d_Hy[d_threadId_k]));

            }
            //source
			//if ((i >= (source_is)) && (i < (source_ie)) && (j >= (source_js)) && (j <= (source_je)) && (k_real >= (source_ks)) && (k_real <= (source_ke))) {

				//d_Ex[threadId] = d_Ex[threadId] + d_Cexj[threadId] * d_signal_per_node[m];

			//}


		}
		//synchronize threads
		__syncthreads();



		//**EY UPDATE
		// check for borders
		// PEC boundaries at the borders
		//if ((i < (NX - 1)) && (j < (NY - 1)) && (k < (NZ - 1)) && (k > 0) && (i > 0)) {
		if ((i < (NXX - 1)) && (j < (NYY - 1)) && (k_real < (NZ - 1)) && (k_real > 0) && (i > 0)) {

			//update Ey
            if(k == (0)){

                d_Ey[threadId] = (d_Ceye[threadId] * d_Ey[threadId]) + (d_Ceyhx[threadId] * (d_Hx[threadId] - d_gHx[d_ghost_k])) + (d_Ceyhz[threadId] * (d_Hz[threadId] - d_Hz[d_threadId_i]));

            } else {

                d_Ey[threadId] = (d_Ceye[threadId] * d_Ey[threadId]) + (d_Ceyhx[threadId] * (d_Hx[threadId] - d_Hx[d_threadId_k])) + (d_Ceyhz[threadId] * (d_Hz[threadId] - d_Hz[d_threadId_i]));
                //d_Ey[threadId] = (d_Ceye[threadId] * d_Ey[threadId]) + (d_Ceyhx[threadId] * (d_Hx[threadId] - d_Hx[d_threadId_k])) + (d_Ceyhz[threadId] * (s_Hy_Hz[tz][ty][tx + WIDTH*WIDTH].y - s_Hy_Hz[tz][ty][tx + WIDTH*WIDTH - 1].y));

            }

		}
		//synchronize the threads
		__syncthreads();



		//**EZ UPDATE
		// check for borders
		// PEC boundaries at the borders
		//if ((i < (NX - 1)) && (j < (NY - 1)) && (k < (NZ - 1)) && (j > 0) && (i > 0)) {
		if ((i < (NXX - 1)) && (j < (NYY - 1)) && (k_real < (NZ - 1)) && (j > 0) && (i > 0)) {

			//update Ez
			//d_Ez[threadId] = (d_Ceze[threadId] * d_Ez[threadId]) + (d_Cezhy[threadId] * (d_Hy[threadId] - d_Hy[d_threadId_i])) + (d_Cezhx[threadId] * (d_Hx[threadId] - d_Hx[d_threadId_j]));
			//d_Ez[threadId] = (d_Ceze[threadId] * d_Ez[threadId]) + (d_Cezhy[threadId] * (s_Hy_Hz[tz][ty][tx + WIDTH*WIDTH].x - s_Hy_Hz[tz][ty][tx + WIDTH*WIDTH - 1].x)) + (d_Cezhx[threadId] * (d_Hx[threadId] - d_Hx[d_threadId_j]));

            if ((i >= (source_is)) && (i <= (source_ie)) && (j >= (source_js)) && (j <= (source_je)) && (k_real >= (source_ks)) && (k_real < (source_ke))) {

				d_Ez[threadId] = (d_Ceze[threadId] * d_Ez[threadId]) + (d_Cezhy[threadId] * (d_Hy[threadId] - d_Hy[d_threadId_i])) + (d_Cezhx[threadId] * (d_Hx[threadId] - d_Hx[d_threadId_j])) + d_Cezj[threadId] * (d_signal_per_node[m]);

            } else {

                d_Ez[threadId] = (d_Ceze[threadId] * d_Ez[threadId]) + (d_Cezhy[threadId] * (d_Hy[threadId] - d_Hy[d_threadId_i])) + (d_Cezhx[threadId] * (d_Hx[threadId] - d_Hx[d_threadId_j]));


            }

            //synchronize the threads
            //__syncthreads();
            //sample Ez
			sampleField( i, j, k, k_real, m,volt_NX, volt_NY, volt_NZ_N, NZ, 0, threadId, sampled_voltage_is, sampled_voltage_js, sampled_voltage_ks, sampled_voltage_ie, sampled_voltage_je, sampled_voltage_ke,  d_Ez, E);


		}
		//synchronize the threads
		__syncthreads();


		//CPML ADJUST*********************************************************************************

		//CPML at the x_n region. Update Ey and Ez
		//if ((i > 0) && (i < pml_x_n) && (j < (NY - 1)) && (k < NZ - 1)) {
		if ((i > 0) && (i < pml_x_n) && (j < (NYY - 1)) && (k_real < NZ - 1)) {

			d_Psi_eyx[threadId] = d_cpml_b_ex[i] * d_Psi_eyx[threadId] + d_cpml_a_ex[i] * (d_Hz[threadId] - d_Hz[d_threadId_i]);

			d_Psi_ezx[threadId] = d_cpml_b_ex[i] * d_Psi_ezx[threadId] + d_cpml_a_ex[i] * (d_Hy[threadId] - d_Hy[d_threadId_i]);

			if (k_real > 0) {

				d_Ey[threadId] = d_Ey[threadId] + d_cpsi_eyx[threadId] * d_Psi_eyx[threadId];

			}
			//synchronize the threads
			//__syncthreads();
			if (j > 0) {

				d_Ez[threadId] = d_Ez[threadId] + d_cpsi_ezx[threadId] * d_Psi_ezx[threadId];

			}

		}
		//synchronize the threads
		__syncthreads();

		//CPML at the x_p region. Update Ey and Ez
		//if ((i > (NX - pml_x_p - 1)) && (i < (NX - 1)) && (j < (NY - 1)) && (k < NZ - 1)) {
		if ((i > (NXX - pml_x_p - 1)) && (i < (NXX - 1)) && (j < (NYY - 1)) && (k_real < NZ - 1)) {

			d_Psi_eyx[threadId] = d_cpml_b_ex[i] * d_Psi_eyx[threadId] + d_cpml_a_ex[i] * (d_Hz[threadId] - d_Hz[d_threadId_i]);

			d_Psi_ezx[threadId] = d_cpml_b_ex[i] * d_Psi_ezx[threadId] + d_cpml_a_ex[i] * (d_Hy[threadId] - d_Hy[d_threadId_i]);

			if (k_real > 0) {

				d_Ey[threadId] = d_Ey[threadId] + d_cpsi_eyx[threadId] * d_Psi_eyx[threadId];

			}
			//synchronize the threads
			//__syncthreads();
			if (j > 0) {

				d_Ez[threadId] = d_Ez[threadId] + d_cpsi_ezx[threadId] * d_Psi_ezx[threadId];

			}
		}
		//synchronize the threads
		__syncthreads();

		//CPML at the y_n region. Update Ex and Ez
		//if ((i < (NX - 1)) && (j > 0) && (j < (pml_y_n)) && (k < (NZ - 1))) {
		if ((i < (NXX - 1)) && (j > 0) && (j < (pml_y_n)) && (k_real < (NZ - 1))) {

			d_Psi_exy[threadId] = d_cpml_b_ey[j] * d_Psi_exy[threadId] + d_cpml_a_ey[j] * (d_Hz[threadId] - d_Hz[d_threadId_j]);
			//d_Psi_exy[threadId] = d_cpml_b_ey[j] * d_Psi_exy[threadId] + d_cpml_a_ey[j] * (s_Hy_Hz[tz][ty][tx + WIDTH*WIDTH].y - d_Hz[d_threadId_j]);

			d_Psi_ezy[threadId] = d_cpml_b_ey[j] * d_Psi_ezy[threadId] + d_cpml_a_ey[j] * (d_Hx[threadId] - d_Hx[d_threadId_j]);


			if (k_real > 0) {

				d_Ex[threadId] = d_Ex[threadId] + d_cpsi_exy[threadId] * d_Psi_exy[threadId];

			}
			//synchronize the threads
			//__syncthreads();
			if (i > 0) {

				d_Ez[threadId] = d_Ez[threadId] + d_cpsi_ezy[threadId] * d_Psi_ezy[threadId];

			}
		}
		//synchronize the threads
		__syncthreads();

		//CPML at the y_p region. Update Ex and Ez
		//if ((i < (NX - 1)) && (j >(NY - pml_y_p - 1)) && (j < (NY - 1)) && (k < (NZ - 1))) {
		if ((i < (NXX - 1)) && (j >(NYY - pml_y_p - 1)) && (j < (NYY - 1)) && (k_real < (NZ - 1))) {

			d_Psi_exy[threadId] = d_cpml_b_ey[j] * d_Psi_exy[threadId] + d_cpml_a_ey[j] * (d_Hz[threadId] - d_Hz[d_threadId_j]);

			d_Psi_ezy[threadId] = d_cpml_b_ey[j] * d_Psi_ezy[threadId] + d_cpml_a_ey[j] * (d_Hx[threadId] - d_Hx[d_threadId_j]);

			if (k_real > 0) {

				d_Ex[threadId] = d_Ex[threadId] + d_cpsi_exy[threadId] * d_Psi_exy[threadId];

			}
			//synchronize the threads
			//__syncthreads();
			if (i > 0) {

				d_Ez[threadId] = d_Ez[threadId] + d_cpsi_ezy[threadId] * d_Psi_ezy[threadId];

			}
		}
		//synchronize the threads
		__syncthreads();

		//CPML at the z_n region. Update Ex and Ey
		//if ((i < (NX - 1)) && (j < (NY - 1)) && (k > 0) && (k < (pml_z_n))) {
		if ((i < (NXX - 1)) && (j < (NYY - 1)) && (k_real > 0) && (k_real < (pml_z_n))) {

            if(k == (0)) {

                 d_Psi_exz[threadId] = d_cpml_b_ez[k] * d_Psi_exz[threadId] + d_cpml_a_ez[k] * (d_Hy[threadId] - d_gHy[d_ghost_k]);

                d_Psi_eyz[threadId] = d_cpml_b_ez[k] * d_Psi_eyz[threadId] + d_cpml_a_ez[k] * (d_Hx[threadId] - d_gHx[d_ghost_k]);

            } else {

                d_Psi_exz[threadId] = d_cpml_b_ez[k] * d_Psi_exz[threadId] + d_cpml_a_ez[k] * (d_Hy[threadId] - d_Hy[d_threadId_k]);

                d_Psi_eyz[threadId] = d_cpml_b_ez[k] * d_Psi_eyz[threadId] + d_cpml_a_ez[k] * (d_Hx[threadId] - d_Hx[d_threadId_k]);

            }
			if (j > 0) {

				d_Ex[threadId] = d_Ex[threadId] + d_cpsi_exz[threadId] * d_Psi_exz[threadId];

			}
			//synchronize the threads
			//__syncthreads();
			if (i > 0) {

				d_Ey[threadId] = d_Ey[threadId] + d_cpsi_eyz[threadId] * d_Psi_eyz[threadId];

			}
		}
		//synchronize the threads
		__syncthreads();

		//CPML at the z_p region. Update Ex and Ey
		//if ((i < (NX - 1)) && (j < (NY - 1)) && (k >(NZ - pml_z_p - 1)) && (k < (NZ - 1))) {
		if ((i < (NXX - 1)) && (j < (NYY - 1)) && (k_real >(NZ - pml_z_p - 1)) && (k_real < (NZ - 1))) {

            if(k == (0)){

                d_Psi_exz[threadId] = d_cpml_b_ez[k] * d_Psi_exz[threadId] + d_cpml_a_ez[k] * (d_Hy[threadId] - d_gHy[d_ghost_k]);

                d_Psi_eyz[threadId] = d_cpml_b_ez[k] * d_Psi_eyz[threadId] + d_cpml_a_ez[k] * (d_Hx[threadId] - d_gHx[d_ghost_k]);

            } else {

                d_Psi_exz[threadId] = d_cpml_b_ez[k] * d_Psi_exz[threadId] + d_cpml_a_ez[k] * (d_Hy[threadId] - d_Hy[d_threadId_k]);

                d_Psi_eyz[threadId] = d_cpml_b_ez[k] * d_Psi_eyz[threadId] + d_cpml_a_ez[k] * (d_Hx[threadId] - d_Hx[d_threadId_k]);
            }

			if (j > 0) {

				d_Ex[threadId] = d_Ex[threadId] + d_cpsi_exz[threadId] * d_Psi_exz[threadId];

			}
			//synchronize the threads
			//__syncthreads();
			if (i > 0) {

				d_Ey[threadId] = d_Ey[threadId] + d_cpsi_eyz[threadId] * d_Psi_eyz[threadId];

			}
		}


	}
}


//Initialize the device variables. Setting everything as the medium air
__global__ void setZero(float * d_Ex, float * d_Jx, float * d_Cexe, float * d_Cexhz, float * d_Cexhy, float * d_Cexj, float * d_eps_r_x, float * d_sigma_e_x,
	float * d_Ey, float * d_Jy, float * d_Ceye, float * d_Ceyhx, float * d_Ceyhz, float * d_Ceyj, float * d_eps_r_y, float * d_sigma_e_y,
	float * d_Ez, float * d_Jz, float * d_Ceze, float * d_Cezhy, float * d_Cezhx, float * d_Cezj, float * d_eps_r_z, float * d_sigma_e_z,
	float * d_Hx, float * d_Mx, float * d_Chxh, float * d_Chxey, float * d_Chxez, float * d_Chxm, float * d_mu_r_x, float * d_sigma_m_x,
	float * d_Hy, float * d_My, float * d_Chyh, float * d_Chyez, float * d_Chyex, float * d_Chym, float * d_mu_r_y, float * d_sigma_m_y,
	float * d_Hz, float * d_Mz, float * d_Chzh, float * d_Chzex, float * d_Chzey, float * d_Chzm, float * d_mu_r_z, float * d_sigma_m_z,
	float *d_material_3d_space_eps_x, float *d_material_3d_space_eps_y, float *d_material_3d_space_eps_z,
	float *d_material_3d_space_sigma_e_x, float *d_material_3d_space_sigma_e_y, float *d_material_3d_space_sigma_e_z,
	float *d_material_3d_space_mu_x, float *d_material_3d_space_mu_y, float *d_material_3d_space_mu_z,
	float *d_material_3d_space_sigma_m_x, float *d_material_3d_space_sigma_m_y, float *d_material_3d_space_sigma_m_z,
	float * d_cpml_b_ex, float * d_cpml_a_ex, float * d_cpml_b_mx, float * d_cpml_a_mx,
	float * d_cpml_b_ey, float * d_cpml_a_ey, float * d_cpml_b_my, float * d_cpml_a_my,
	float * d_cpml_b_ez, float * d_cpml_a_ez, float * d_cpml_b_mz, float * d_cpml_a_mz,
	float * d_Psi_eyx, float * d_Psi_ezx, float * d_Psi_hyx, float * d_Psi_hzx,
	float * d_cpsi_eyx, float * d_cpsi_ezx, float * d_cpsi_hyx, float * d_cpsi_hzx,
	float * d_Psi_exy, float * d_Psi_ezy, float * d_Psi_hxy, float * d_Psi_hzy,
	float * d_cpsi_exy, float * d_cpsi_ezy, float * d_cpsi_hxy, float * d_cpsi_hzy,
	float * d_Psi_exz, float * d_Psi_eyz, float * d_Psi_hxz, float * d_Psi_hyz,
	float * d_cpsi_exz, float * d_cpsi_eyz, float * d_cpsi_hxz, float * d_cpsi_hyz,
	int NX, int NY, int NZ_N) {

	int a = blockIdx.x * blockDim.x + threadIdx.x;
	int b = blockIdx.y * blockDim.y + threadIdx.y;
	int c = blockIdx.z * blockDim.z + threadIdx.z;

	//index of current thread
	int idx = a + b * NX + c * NX * NY;

	if (idx < (NX*NY*NZ_N)) {

		d_material_3d_space_sigma_e_x[idx] = 1.2e-38;
		d_material_3d_space_eps_x[idx] = 1;
		d_material_3d_space_sigma_e_y[idx] = 1.2e-38;
		d_material_3d_space_eps_y[idx] = 1;
		d_material_3d_space_sigma_e_z[idx] = 1.2e-38;
		d_material_3d_space_eps_z[idx] = 1;
		d_material_3d_space_sigma_m_x[idx] = 1.2e-38;
		d_material_3d_space_mu_x[idx] = 1;
		d_material_3d_space_sigma_m_y[idx] = 1.2e-38;
		d_material_3d_space_mu_y[idx] = 1;
		d_material_3d_space_sigma_m_z[idx] = 1.2e-38;
		d_material_3d_space_mu_z[idx] = 1;

		//elecric conductivity x direction
		d_sigma_e_x[idx] = 1.2e-38;
		//elecric conductivity y direction
		d_sigma_e_y[idx] = 1.2e-38;
		//elecric conductivity z direction
		d_sigma_e_z[idx] = 1.2e-38;

		//relative electric permissivity in x direction
		d_eps_r_x[idx] = 1;
		//relative electric permissivity in y direction
		d_eps_r_y[idx] = 1;
		//relative electric permissivity in z direction
		d_eps_r_z[idx] = 1;

		//magnetic conductivity x direction
		d_sigma_m_x[idx] = 1.2e-38;
		//magnetic conductivity y direction
		d_sigma_m_y[idx] = 1.2e-38;
		//magnetic conductivity z direction
		d_sigma_m_z[idx] = 1.2e-38;

		//relative magnetic permeability in z direction
		d_mu_r_x[idx] = 1;
		//relative magnetic permeability in z direction
		d_mu_r_y[idx] = 1;
		//relative magnetic permeability in z direction
		d_mu_r_z[idx] = 1;

		d_Ez[idx] = 1.2e-38;
		d_Jz[idx] = 1.2e-38;
		d_Ey[idx] = 1.2e-38;
		d_Jy[idx] = 1.2e-38;
		d_Ex[idx] = 1.2e-38;
		d_Jx[idx] = 1.2e-38;

		d_Hx[idx] = 1.2e-38;
		d_Mx[idx] = 1.2e-38;
		d_Hy[idx] = 1.2e-38;
		d_My[idx] = 1.2e-38;
		d_Hz[idx] = 1.2e-38;
		d_Mz[idx] = 1.2e-38;

		//Coeficients
		d_Cexe[idx] = 1.2e-38;
		d_Cexhz[idx] = 1.2e-38;
		d_Cexhy[idx] = 1.2e-38;
		d_Cexj[idx] = 1.2e-38;
		d_Ceye[idx] = 1.2e-38;
		d_Ceyhx[idx] = 1.2e-38;
		d_Ceyhz[idx] = 1.2e-38;
		d_Ceyj[idx] = 1.2e-38;
		d_Ceze[idx] = 1.2e-38;
		d_Cezhy[idx] = 1.2e-38;
		d_Cezhx[idx] = 1.2e-38;
		d_Cezj[idx] = 1.2e-38;


		//Coeficients
		d_Chxh[idx] = 1.2e-38;
		d_Chxey[idx] = 1.2e-38;
		d_Chxez[idx] = 1.2e-38;
		d_Chxm[idx] = 1.2e-38;
		d_Chyh[idx] = 1.2e-38;
		d_Chyez[idx] = 1.2e-38;
		d_Chyex[idx] = 1.2e-38;
		d_Chym[idx] = 1.2e-38;
		d_Chzh[idx] = 1.2e-38;
		d_Chzex[idx] = 1.2e-38;
		d_Chzey[idx] = 1.2e-38;
		d_Chzm[idx] = 1.2e-38;

		//xn and xp coefficients
		d_Psi_eyx[idx] = 1.2e-38;
		d_Psi_ezx[idx] = 1.2e-38;
		d_Psi_hyx[idx] = 1.2e-38;
		d_Psi_hzx[idx] = 1.2e-38;
		d_cpsi_eyx[idx] = 1.2e-38;
		d_cpsi_ezx[idx] = 1.2e-38;
		d_cpsi_hyx[idx] = 1.2e-38;
		d_cpsi_hzx[idx] = 1.2e-38;
		//yn and yp coefficients
		d_Psi_exy[idx] = 1.2e-38;
		d_Psi_ezy[idx] = 1.2e-38;
		d_Psi_hxy[idx] = 1.2e-38;
		d_Psi_hzy[idx] = 1.2e-38;
		d_cpsi_exy[idx] = 1.2e-38;
		d_cpsi_ezy[idx] = 1.2e-38;
		d_cpsi_hxy[idx] = 1.2e-38;
		d_cpsi_hzy[idx] = 1.2e-38;
		//zn and zp coefficients
		d_Psi_exz[idx] = 1.2e-38;
		d_Psi_eyz[idx] = 1.2e-38;
		d_Psi_hxz[idx] = 1.2e-38;
		d_Psi_hyz[idx] = 1.2e-38;
		d_cpsi_exz[idx] = 1.2e-38;
		d_cpsi_eyz[idx] = 1.2e-38;
		d_cpsi_hxz[idx] = 1.2e-38;
		d_cpsi_hyz[idx] = 1.2e-38;

		//Setting the CPML xn and xp coefficients
		if (a < NX)
		{

			d_cpml_b_ex[a] = 1.2e-38;
			d_cpml_a_ex[a] = 1.2e-38;
			d_cpml_b_mx[a] = 1.2e-38;
			d_cpml_a_mx[a] = 1.2e-38;
		}


		//Setting the CPML yn and yp coefficients
		if (b < NY)
		{
			d_cpml_b_ey[b] = 1.2e-38;
			d_cpml_a_ey[b] = 1.2e-38;
			d_cpml_b_my[b] = 1.2e-38;
			d_cpml_a_my[b] = 1.2e-38;
		}



		//Setting the CPML zn and zp coefficients
		if (c < NZ_N)
		{
			d_cpml_b_ez[c] = 1.2e-38;
			d_cpml_a_ez[c] = 1.2e-38;
			d_cpml_b_mz[c] = 1.2e-38;
			d_cpml_a_mz[c] = 1.2e-38;
		}


	}
}

//Determine the sigma and epislon based on the material grid
__global__ void setSigmaEpsSigmaMu(int NX, int NY, int NZ, int NZ_N, int gpu_offset,
	float *d_material_3d_space_eps_x, float *d_material_3d_space_eps_y, float *d_material_3d_space_eps_z,
	float *d_material_3d_space_sigma_e_x, float *d_material_3d_space_sigma_e_y, float *d_material_3d_space_sigma_e_z,
	float *d_material_3d_space_mu_x, float *d_material_3d_space_mu_y, float *d_material_3d_space_mu_z,
	float *d_material_3d_space_sigma_m_x, float *d_material_3d_space_sigma_m_y, float *d_material_3d_space_sigma_m_z,
	float * d_eps_r_x, float * d_sigma_e_x, float * d_eps_r_y, float * d_sigma_e_y, float * d_eps_r_z, float * d_sigma_e_z,
	float * d_mu_r_x, float * d_sigma_m_x, float * d_mu_r_y, float * d_sigma_m_y, float * d_mu_r_z, float * d_sigma_m_z) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;

	//K and its offset due to the multiples GPUs
	int k_real = k + gpu_offset;

	//index of current thread
	int idx = i + j * NX + k * NX * NY;

	//index with i -1, j -1 or k-1
	int idx_i, idx_j, idx_k;

	//index with i-1 and j-1 or i-1 and k-1
	int idx_ij, idx_ik;

	//index with j-1 and k-1
	int idx_jk;

	if (idx < (NX*NY*NZ_N)) {

		//uptade eps_r_x and sigma_e_x
		if ((i < (NX)) && (j < (NY)) && (k_real < (NZ)) && (j > 0) && (k_real > 0) && (k > 0)) {

			idx_j = i + (j - 1) * NX + k * NX * NY;

			idx_k = i + j * NX + (k - 1) * NX * NY;

			idx_jk = i + (j - 1) * NX + (k - 1) * NX * NY;

			d_eps_r_x[idx] = 0.25 * (d_material_3d_space_eps_x[idx] + d_material_3d_space_eps_x[idx_j] + d_material_3d_space_eps_x[idx_k] + d_material_3d_space_eps_x[idx_jk]);

			d_sigma_e_x[idx] = 0.25 * (d_material_3d_space_sigma_e_x[idx] + d_material_3d_space_sigma_e_x[idx_j] + d_material_3d_space_sigma_e_x[idx_k] + d_material_3d_space_sigma_e_x[idx_jk]);


		}

		__syncthreads();

		//uptade eps_r_y and sigma_e_y
		if ((i < (NX)) && (j < (NY)) && (k_real < (NZ)) && (i > 0) && (k_real > 0) && (k > 0)) {

			idx_i = (i - 1) + j * NX + k * NX * NY;

			idx_k = i + j * NX + (k - 1) * NX * NY;

			idx_ik = (i - 1) + j * NX + (k - 1) * NX * NY;

			d_eps_r_y[idx] = 0.25 * (d_material_3d_space_eps_y[idx] + d_material_3d_space_eps_y[idx_i] + d_material_3d_space_eps_y[idx_k] + d_material_3d_space_eps_y[idx_ik]);

			d_sigma_e_y[idx] = 0.25 * (d_material_3d_space_sigma_e_y[idx] + d_material_3d_space_sigma_e_y[idx_i] + d_material_3d_space_sigma_e_y[idx_k] + d_material_3d_space_sigma_e_y[idx_ik]);

		}

		__syncthreads();

		//uptade eps_r_z and sigma_e_z
		if ((i < (NX)) && (j < (NY)) && (k_real < (NZ)) && (i > 0) && (j > 0)) {

			idx_i = (i - 1) + j * NX + k * NX * NY;

			idx_j = i + (j - 1) * NX + k * NX * NY;

			idx_ij = (i - 1) + (j - 1) * NX + k * NX * NY;

			d_eps_r_z[idx] = 0.25 * (d_material_3d_space_eps_z[idx] + d_material_3d_space_eps_z[idx_i] + d_material_3d_space_eps_z[idx_j] + d_material_3d_space_eps_z[idx_ij]);

			d_sigma_e_z[idx] = 0.25 * (d_material_3d_space_sigma_e_z[idx] + d_material_3d_space_sigma_e_z[idx_i] + d_material_3d_space_sigma_e_z[idx_j] + d_material_3d_space_sigma_e_z[idx_ij]);

		}

		__syncthreads();

		//uptade mu_r_x and sigma_m_x
		if ((i < (NX)) && (j < (NY)) && (k_real < (NZ)) && (i > 0)) {

			idx_i = (i - 1) + j * NX + k * NX * NY;

			d_mu_r_x[idx] = (2 * d_material_3d_space_mu_x[idx] * d_material_3d_space_mu_x[idx_i]) / (d_material_3d_space_mu_x[idx] + d_material_3d_space_mu_x[idx_i]);

			d_sigma_m_x[idx] = (2 * d_material_3d_space_sigma_m_x[idx] * d_material_3d_space_sigma_m_x[idx_i]) / (d_material_3d_space_sigma_m_x[idx] + d_material_3d_space_sigma_m_x[idx_i]);

		}

		__syncthreads();

		//uptade mu_r_y and sigma_m_y
		if ((i < (NX)) && (j < (NY)) && (k_real < (NZ)) && (j > 0)) {

			idx_j = i + (j - 1) * NX + k * NX * NY;

			d_mu_r_y[idx] = (2 * d_material_3d_space_mu_y[idx] * d_material_3d_space_mu_y[idx_j]) / (d_material_3d_space_mu_y[idx] + d_material_3d_space_mu_y[idx_j]);

			d_sigma_m_y[idx] = (2 * d_material_3d_space_sigma_m_y[idx] * d_material_3d_space_sigma_m_y[idx_j]) / (d_material_3d_space_sigma_m_y[idx] + d_material_3d_space_sigma_m_y[idx_j]);

		}

		__syncthreads();

		//uptade mu_r_z and sigma_m_z
		if ((i < (NX)) && (j < (NY)) && (k_real < (NZ)) && (k_real > 0) && (k > 0)) {

			idx = i + j * NX + k * NX * NY;

			idx_k = i + j * NX + (k - 1) * NX * NY;

			d_mu_r_z[idx] = (2 * d_material_3d_space_mu_z[idx] * d_material_3d_space_mu_z[idx_k]) / (d_material_3d_space_mu_z[idx] + d_material_3d_space_mu_z[idx_k]);

			d_sigma_m_z[idx] = (2 * d_material_3d_space_sigma_m_z[idx] * d_material_3d_space_sigma_m_z[idx_k]) / (d_material_3d_space_sigma_m_z[idx] + d_material_3d_space_sigma_m_z[idx_k]);
		}

	}



}

//Set coefficients to calculate the fields
__global__ void setCoefficients(int NX, int NXX, int NY, int NYY, int NZ, int NZ_N, int gpu_offset,
	float * d_Cexe, float * d_Cexhz, float * d_Cexhy, float * d_Cexj, float * d_eps_r_x, float * d_sigma_e_x,
	float * d_Ceye, float * d_Ceyhx, float * d_Ceyhz, float * d_Ceyj, float * d_eps_r_y, float * d_sigma_e_y,
	float * d_Ceze, float * d_Cezhy, float * d_Cezhx, float * d_Cezj, float * d_eps_r_z, float * d_sigma_e_z,
	float * d_Chxh, float * d_Chxey, float * d_Chxez, float * d_Chxm, float * d_mu_r_x, float * d_sigma_m_x,
	float * d_Chyh, float * d_Chyez, float * d_Chyex, float * d_Chym, float * d_mu_r_y, float * d_sigma_m_y,
	float * d_Chzh, float * d_Chzex, float * d_Chzey, float * d_Chzm, float * d_mu_r_z, float * d_sigma_m_z,
	float dx, float dy, float dz, float dt,
	float eps_0, float pi, float mu_0) {


	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;

	//K and its offset due to the multiples GPUs
	int k_real = k + gpu_offset;

	//index of current thread
	int idx = i + j * NX + k * NX * NY;

	if (idx < (NX*NY*NZ_N)) {

		//For loop to set Hx Coefficients coeficients
		if ((i < (NXX)) && (j < (NYY - 1)) && (k_real < (NZ - 1))) {


			d_Chxh[idx] = (2 * d_mu_r_x[idx] * mu_0 - dt * d_sigma_m_x[idx]) / (2 * d_mu_r_x[idx] * mu_0 + dt * d_sigma_m_x[idx]);
			d_Chxey[idx] = (2 * dt) / ((2 * d_mu_r_x[idx] * mu_0 + dt * d_sigma_m_x[idx])*dz);
			d_Chxez[idx] = ((-1) * 2 * dt) / ((2 * d_mu_r_x[idx] * mu_0 + dt * d_sigma_m_x[idx])*dy);

		}
		//__syncthreads();

		//For loop to set Hy Coefficients coeficients
		if ((i < (NXX - 1)) && (j < (NYY)) && (k_real < (NZ - 1))) {

			d_Chyh[idx] = (2 * d_mu_r_y[idx] * mu_0 - dt * d_sigma_m_y[idx]) / (2 * d_mu_r_y[idx] * mu_0 + dt * d_sigma_m_y[idx]);
			d_Chyez[idx] = (2 * dt) / ((2 * d_mu_r_y[idx] * mu_0 + dt * d_sigma_m_y[idx])*dx);
			d_Chyex[idx] = ((-1) * 2 * dt) / ((2 * d_mu_r_y[idx] * mu_0 + dt * d_sigma_m_y[idx])*dz);

		}
		//__syncthreads();

		//For loop to set Hz Coefficients coeficients
		if ((i < (NXX - 1)) && (j < (NYY - 1)) && (k_real < (NZ))) {

			d_Chzh[idx] = (2 * d_mu_r_z[idx] * mu_0 - dt * d_sigma_m_z[idx]) / (2 * d_mu_r_z[idx] * mu_0 + dt * d_sigma_m_z[idx]);
			d_Chzex[idx] = (2 * dt) / ((2 * d_mu_r_z[idx] * mu_0 + dt * d_sigma_m_z[idx])*dy);
			d_Chzey[idx] = ((-1) * 2 * dt) / ((2 * d_mu_r_z[idx] * mu_0 + dt * d_sigma_m_z[idx])*dx);
		}
		//__syncthreads();

		//For loop to set Ex Coefficients coeficients
		if ((i < (NXX - 1)) && (j < (NYY - 1)) && (k_real < (NZ - 1)) && (j > 0) && (k_real > 0)) {


			d_Cexe[idx] = (2 * d_eps_r_x[idx] * eps_0 - dt * d_sigma_e_x[idx]) / (2 * d_eps_r_x[idx] * eps_0 + dt * d_sigma_e_x[idx]);
			d_Cexhz[idx] = (2 * dt) / ((2 * d_eps_r_x[idx] * eps_0 + dt * d_sigma_e_x[idx])*dy);
			d_Cexhy[idx] = ((-1) * 2 * dt) / ((2 * d_eps_r_x[idx] * eps_0 + dt * d_sigma_e_x[idx])*dz);

		}
		//__syncthreads();

		//For loop to set Ey Coefficients coeficients
		if ((i < (NXX - 1)) && (j < (NYY - 1)) && (k_real < (NZ - 1)) && (i > 0) && (k_real > 0)) {

			d_Ceye[idx] = (2 * d_eps_r_y[idx] * eps_0 - dt * d_sigma_e_y[idx]) / (2 * d_eps_r_y[idx] * eps_0 + dt * d_sigma_e_y[idx]);
			d_Ceyhx[idx] = (2 * dt) / ((2 * d_eps_r_y[idx] * eps_0 + dt * d_sigma_e_y[idx])*dz);
			d_Ceyhz[idx] = ((-1) * 2 * dt) / ((2 * d_eps_r_y[idx] * eps_0 + dt * d_sigma_e_y[idx])*dx);

		}
		//__syncthreads();


		//For loop to set Ez Coefficients coeficients
		if ((i < (NXX - 1)) && (j < (NYY - 1)) && (k_real < (NZ - 1)) && (i > 0) && (j > 0)) {

			d_Ceze[idx] = (2 * d_eps_r_z[idx] * eps_0 - dt * d_sigma_e_z[idx]) / (2 * d_eps_r_z[idx] * eps_0 + dt * d_sigma_e_z[idx]);
			d_Cezhy[idx] = (2 * dt) / ((2 * d_eps_r_z[idx] * eps_0 + dt * d_sigma_e_z[idx])*dx);
			d_Cezhx[idx] = ((-1) * 2 * dt) / ((2 * d_eps_r_z[idx] * eps_0 + dt * d_sigma_e_z[idx])*dy);

		}
	}

}

//define source in the X direction
__global__ void defineSourceX(int NX, int NXX, int NY, int NYY, int NZ, int NZ_N, int gpu_offset,
	int source_is, int source_js, int source_ks, int source_ie, int source_je, int source_ke,
    float * d_Cexe, float * d_Cexhz, float * d_Cexhy, float * d_Cexj, float * d_eps_r_x, float * d_sigma_e_x,
	float rs,
	float dx, float dy, float dz, float dt,
	float eps_0) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;

	int k_real = k + gpu_offset;

	//index of current thread
	int idx = i + j * NX + k * NX * NY;

	float r_comp;
	float rfactor, rfactor_aux;

	float _is, _js, _ks;
	float _ie, _je, _ke;

	_is = (float)source_is;
	_ie = (float)source_ie;

	_js = (float)source_js;
	_je = (float)source_je;

	_ks = (float)source_ks;
	_ke = (float)source_ke;

	if (idx < (NX*NY*NZ_N)) {

		//resistance per component
		r_comp = (1 + _je - _js) * (1 + _ke - _ks) / (_ie - _is);
		r_comp = r_comp * rs;
		//resistor factor Z oriented
		rfactor = (dt*dx) / (r_comp*dz*dy);
		rfactor_aux = (r_comp*dy*dz);

		//For loop to set values to the electric and magnetic coeficients related to the sources X direction
		if ((i >= (source_is)) && (i < (source_ie)) && (j >= (source_js)) && (j <= (source_ie)) && (k_real >= (source_ks)) && (k_real <= (source_ke))) {

			d_Cexe[idx] = (2 * d_eps_r_x[idx] * eps_0 - dt * d_sigma_e_x[idx] - rfactor) / (2 * d_eps_r_x[idx] * eps_0 + dt * d_sigma_e_x[idx] + rfactor);

			d_Cexhz[idx] = (2 * dt / dy) / (2 * d_eps_r_x[idx] * eps_0 + dt * d_sigma_e_x[idx] + rfactor);

			d_Cexhy[idx] = ((-1) * 2 * dt / dz) / (2 * d_eps_r_x[idx] * eps_0 + dt * d_sigma_e_x[idx] + rfactor);

			d_Cexj[idx] = ((-1) * 2 * dt / rfactor_aux) / (2 * d_eps_r_x[idx] * eps_0 + dt * d_sigma_e_x[idx] + rfactor);

		}
	}

}

//define source in the Z direction
__global__ void defineSourceZ(int NX, int NXX, int NY, int NYY, int NZ, int NZ_N, int gpu_offset,
	int source_is, int source_js, int source_ks, int source_ie, int source_je, int source_ke,
	float * d_Ceze, float * d_Cezhy, float * d_Cezhx, float * d_Cezj, float * d_eps_r_z, float * d_sigma_e_z,
	float rs,
	float dx, float dy, float dz, float dt,
	float eps_0) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;

	int k_real = k + gpu_offset;

	//index of current thread
	int idx = i + j * NX + k * NX * NY;

	float r_comp;
	float rfactor, rfactor_aux;

	float _is, _js, _ks;
	float _ie, _je, _ke;

	_is = (float)source_is;
	_ie = (float)source_ie;

	_js = (float)source_js;
	_je = (float)source_je;

	_ks = (float)source_ks;
	_ke = (float)source_ke;

	__syncthreads();

	//resistance per component
	r_comp = (1 + _ie - _is) * (1 + _je - _js) / (_ke - _ks);
	r_comp = r_comp * rs;
	//resistor factor Z oriented
	rfactor = (dt*dz) / (r_comp*dx*dy);
	rfactor_aux = (r_comp*dx*dy);

	__syncthreads();

	if (idx < (NX*NY*NZ_N)) {

		if ((i >= (source_is)) && (i <= (source_ie)) && (j >= (source_js)) && (j <= (source_je)) && (k_real >= (source_ks)) && (k_real < (source_ke)) ) {
			//set values to the electric and magnetic coeficients related to the sources Z direction

			d_Ceze[idx] = (2 * d_eps_r_z[idx] * eps_0 - dt * d_sigma_e_z[idx] - rfactor) / (2 * d_eps_r_z[idx] * eps_0 + dt * d_sigma_e_z[idx] + rfactor);

			d_Cezhy[idx] = (2 * dt / dx) / (2 * d_eps_r_z[idx] * eps_0 + dt * d_sigma_e_z[idx] + rfactor);

			d_Cezhx[idx] = ((-1) * 2 * dt / dy) / (2 * d_eps_r_z[idx] * eps_0 + dt * d_sigma_e_z[idx] + rfactor);

			d_Cezj[idx] = ((-1) * 2 * dt / rfactor_aux) / (2 * d_eps_r_z[idx] * eps_0 + dt * d_sigma_e_z[idx] + rfactor);

		}
	}

}

//Define Resistor in domain
	__global__ void defineResistor(int NX, int NXX, int NY, int NYY, int NZ, int NZ_N, int gpu_offset,
	int resistor_is, int resistor_js, int resistor_ks, int resistor_ie, int resistor_je, int resistor_ke, int direction,
	float * d_Ceze, float * d_Cezhy, float * d_Cezhx, float * d_eps_r_z, float * d_sigma_e_z,
	float rs,
	float dx, float dy, float dz, float dt,
	float eps_0) {

        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        int k = blockIdx.z * blockDim.z + threadIdx.z;
        //index of current thread
        int idx = i + j * NX + k * NX * NY;

        int k_real = k + gpu_offset;

		float r_comp;
		float rfactor;

		float _is, _js, _ks;
		float _ie, _je, _ke;

        _is = (float)resistor_is;
        _ie = (float)resistor_ie;

        _js = (float)resistor_js;
        _je = (float)resistor_je;

        _ks = (float)resistor_ks;
        _ke = (float)resistor_ke;

        //resistance per component
		r_comp = (1 + _ie - _is) * (1 + _je - _js) / (_ke - _ks);
		r_comp = r_comp * rs;
		//resistor factor Z oriented
		rfactor = (dt*dz) / (r_comp*dx*dy);

		//For loop to set values to the electric and magnetic coeficients related to the sources Z direction
        if (idx < (NX*NY*NZ_N)) {


		if ((i >= (resistor_is)) && (i <= (resistor_ie)) && (j >= (resistor_js)) && (j <= (resistor_je)) && (k_real >= (resistor_ks)) && (k_real < (resistor_ke)) ) {
			//for (int k = ks; k < (ke); k++) {
				//for (int j = js; j < (je + 1); j++) {
					//for (int i = is; i < (ie + 1); i++) {

						d_Ceze[idx] = (2 * d_eps_r_z[idx] * eps_0 - dt * d_sigma_e_z[idx] - rfactor) / (2 * d_eps_r_z[idx] * eps_0 + dt * d_sigma_e_z[idx] + rfactor);

						d_Cezhy[idx] = (2 * dt / dx) / (2 * d_eps_r_z[idx] * eps_0 + dt * d_sigma_e_z[idx] + rfactor);

						d_Cezhx[idx] = ((-1) * 2 * dt / dy) / (2 * d_eps_r_z[idx] * eps_0 + dt * d_sigma_e_z[idx] + rfactor);

        }

    }
}

//Define the type of the input signal as gaussian
__global__ void setGauss(float * d_signal, float * d_signal_per_node, float source_amp, int v_mag_factor, float dt, float t_0, float tau, int source_size) {

	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	//int t_real = threadId + gpu_offset;
	int t_real = threadId;

	if (threadId < source_size) {

		d_signal[threadId] = source_amp * exp((-(((t_real*dt) - t_0)*((t_real*dt) - t_0)) / (tau*tau)));

		//Input signal per node
		d_signal_per_node[threadId] = d_signal[threadId]  / (v_mag_factor);
		//d_signal_per_node[threadId] = 0;
	}
}


//Define the type of the input signal as sinusoidal
__global__ void setSine(float * d_signal, float * d_signal_per_node, int v_mag_factor, float amp, float freq, float dt, float pi, int source_size) {

	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	//int t_real = threadId + gpu_offset;
	int t_real = threadId;

	if (threadId < source_size) {

		d_signal[threadId] = amp*sin(2*pi*freq*dt*t_real);

		//Input signal per node
		d_signal_per_node[threadId] = d_signal[threadId]  / (v_mag_factor);
		//d_signal_per_node[threadId] = 0;
	}
}


//Set 3d bricks in the domain
__global__ void setBrick(int NX, int NY, int NZ, int NZ_N, int gpu_offset,
	float *d_material_3d_space_eps_x, float *d_material_3d_space_eps_y, float *d_material_3d_space_eps_z,
	float *d_material_3d_space_sigma_e_x, float *d_material_3d_space_sigma_e_y, float *d_material_3d_space_sigma_e_z,
	float *d_material_3d_space_mu_x, float *d_material_3d_space_mu_y, float *d_material_3d_space_mu_z,
	float *d_material_3d_space_sigma_m_x, float *d_material_3d_space_sigma_m_y, float *d_material_3d_space_sigma_m_z,
	int brick_is, int brick_js, int brick_ks, int brick_ie, int brick_je, int brick_ke,
	float brick_sigma_e_x, float brick_sigma_e_y, float brick_sigma_e_z, float brick_eps_r_x, float brick_eps_r_y, float brick_eps_r_z,
	float brick_sigma_m_x, float brick_sigma_m_y, float brick_sigma_m_z, float brick_mu_r_x, float brick_mu_r_y, float brick_mu_r_z) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;

	int k_real = k + gpu_offset;

	//index of current thread
	//int threadId = (brick_is + i) + (brick_js + j) * NX + (brick_ks + k) * NX * NY;

	int threadId = (i)+(j)* NX + (k)* NX * NY;

	if (threadId < (NX*NY*NZ_N)) {

		//if ((i == (brick_ie)) && (j < (brick_je)) && (k_real < (brick_ke))) {
		if ((i >= (brick_is)) && (i < (brick_ie)) && (j >= (brick_js)) && (j < (brick_je)) && (k_real >= (brick_ks)) && (k_real < (brick_ke))) {

			d_material_3d_space_sigma_e_x[threadId] = brick_sigma_e_x;
			d_material_3d_space_eps_x[threadId] = brick_eps_r_x;


			d_material_3d_space_sigma_e_y[threadId] = brick_sigma_e_y;
			d_material_3d_space_eps_y[threadId] = brick_eps_r_y;

			d_material_3d_space_sigma_e_z[threadId] = brick_sigma_e_z;
			d_material_3d_space_eps_z[threadId] = brick_eps_r_z;

			d_material_3d_space_sigma_m_x[threadId] = brick_sigma_m_x;
			d_material_3d_space_mu_x[threadId] = brick_mu_r_x;

			d_material_3d_space_sigma_m_y[threadId] = brick_sigma_m_y;
			d_material_3d_space_mu_y[threadId] = brick_mu_r_y;

			d_material_3d_space_sigma_m_z[threadId] = brick_sigma_m_z;
			d_material_3d_space_mu_z[threadId] = brick_mu_r_z;

		}

	}
}

//define zero thickness plate x axis
__global__ void definePlateX(int NX, int NY, int NZ, int NZ_N, int gpu_offset,
	float * d_sigma_e_y, float * d_sigma_e_z,
	float plate_sigma_e_y, float plate_sigma_e_z,
	int plate_is, int plate_js, int plate_ks, int plate_ie, int plate_je, int plate_ke) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;

	int k_real = k + gpu_offset;

	//index of current thread
	int threadId = i + j * NX + k * NX * NY;

	if (threadId < (NX*NY*NZ_N)) {
		//update sigma_e_y
		if ((i == (plate_is)) && (j >= (plate_js)) && (j < (plate_je)) && (k_real >= (plate_ks)) && (k_real <= (plate_ke))) {

			d_sigma_e_y[threadId] = plate_sigma_e_y;

		}
		__syncthreads();

		//update sigma_e_z
		if ((i == (plate_is)) && (j >= (plate_js)) && (j <= (plate_je)) && (k_real >= (plate_ks)) && (k_real < (plate_ke))) {

			d_sigma_e_z[threadId] = plate_sigma_e_z;

		}
	}

}

//define zero thickness plate y axis
__global__ void definePlateY(int NX, int NY, int NZ, int NZ_N, int gpu_offset,
	float * d_sigma_e_x, float * d_sigma_e_z,
	float plate_sigma_e_x, float plate_sigma_e_z,
	int plate_is, int plate_js, int plate_ks, int plate_ie, int plate_je, int plate_ke) {


	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;

	int k_real = k + gpu_offset;

	//index of current thread
	int threadId = i + j * NX + k * NX * NY;

	if (threadId < (NX*NY*NZ_N)) {
		//update sigma_e_z
		if ((i >= (plate_is)) && (i <= (plate_ie)) && (j == (plate_js)) && (k_real >= (plate_ks)) && (k_real < (plate_ke))) {

			d_sigma_e_z[threadId] = plate_sigma_e_z;

		}

		__syncthreads();
		//update sigma_e_x
		if ((i >= (plate_is)) && (i < (plate_ie)) && (j == (plate_js)) && (k_real >= (plate_ks)) && (k_real <= (plate_ke))) {

			d_sigma_e_x[threadId] = plate_sigma_e_x;

		}
	}

}

//define zero thickness plate z axis
__global__ void definePlateZ(int NX, int NY, int NZ, int NZ_N, int gpu_offset,
	float * d_sigma_e_x, float * d_sigma_e_y,
	float plate_sigma_e_x, float plate_sigma_e_y,
	int plate_is, int plate_js, int plate_ks, int plate_ie, int plate_je, int plate_ke) {


	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;

	int k_real = k + gpu_offset;

	//index of current thread
	int threadId = i + j * NX + k * NX * NY;

	if (threadId < (NX*NY*NZ_N)) {

		//update sigma_e_x
		if ((i >= (plate_is)) && (i < (plate_ie)) && (j >= (plate_js)) && (j <= (plate_je)) && (k_real == (plate_ks))) {

			d_sigma_e_x[threadId] = plate_sigma_e_x;

		}

		//__syncthreads();

		//update sigma_e_y
		if ((i >= (plate_is)) && (i <= (plate_ie)) && (j >= (plate_js)) && (j < (plate_je)) && (k_real == (plate_ks))) {

			d_sigma_e_y[threadId] = plate_sigma_e_y;

		}

	}

}

//define Sector
__global__ void defineSec(int NX, int NY, int NZ, int NZ_N, int gpu_offset, int MAX_X, int MAX_Y,
	int px0, int py0, int px1, int py1, int px2, int py2, int pz0, float raio,
	float sigma_sector_e_x, float sigma_sector_e_y,
	float * d_sigma_e_x, float * d_sigma_e_y,
	float dx, float dy, float dz) {

	int idx;

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;

	int k_real = k + gpu_offset;

	int min_x, min_y;
	int max_x, max_y;
	int zcoord;

	min_x = 0;
	max_x = MAX_X;

	min_y = 0;
	max_y = MAX_Y;

	zcoord = pz0;

	//index of current thread
	int threadId = i + j * NX + k * NX * NY;

	if (threadId < (NX*NY*NZ_N)) {

		//update sigma_e_x
		//if ((i >= (min_x)) && (i < (max_x)) && (j >= (min_y)) && (j <= (max_y))) {
		if ((i >= (min_x)) && (i < (max_x)) && (j >= (min_y)) && (j <= (max_y)) && (zcoord == k_real)) {

			if (tricheck(i*dx, j*dy, px0*dx, py0*dy, px1*dx, py1*dy, px2*dx, py2*dy, raio) == 0) {

				//3d to 1d index
				//idx = i + j * NX + zcoord * NX * NY;
				idx = i + j * NX + k * NX * NY;

				d_sigma_e_x[idx] = sigma_sector_e_x;
			}

		}

		//update sigma_e_y
		//if ((i >= (min_x)) && (i <= (max_x)) && (j >= (min_y)) && (j < (max_y))) {
		if ((i >= (min_x)) && (i <= (max_x)) && (j >= (min_y)) && (j < (max_y)) && (zcoord == k_real)) {
			if (tricheck(i*dx, j*dy, px0*dx, py0*dy, px1*dx, py1*dy, px2*dx, py2*dy, raio) == 0) {

				//3d to 1d index
				//idx = i + j * NX + zcoord * NX * NY;

				idx = i + j * NX + k * NX * NY;

				d_sigma_e_y[idx] = sigma_sector_e_y;
			}


		}

	}
}

//Initialize CPML factors
__global__ void setABCPML(int NX, int NXX, int NY, int NYY, int NZ, int NZ_N, int gpu_offset,
	float * d_cpml_b_ex, float * d_cpml_a_ex, float * d_cpml_b_mx, float * d_cpml_a_mx,
	float * d_cpml_b_ey, float * d_cpml_a_ey, float * d_cpml_b_my, float * d_cpml_a_my,
	float * d_cpml_b_ez, float * d_cpml_a_ez, float * d_cpml_b_mz, float * d_cpml_a_mz,
	float * d_cpsi_eyx, float * d_cpsi_ezx, float * d_cpsi_hyx, float * d_cpsi_hzx,
	float * d_cpsi_exy, float * d_cpsi_ezy, float * d_cpsi_hxy, float * d_cpsi_hzy,
	float * d_cpsi_exz, float * d_cpsi_eyz, float * d_cpsi_hxz, float * d_cpsi_hyz,
	float * d_Cexhz, float * d_Cexhy,
	float * d_Ceyhx, float * d_Ceyhz,
	float * d_Cezhy, float * d_Cezhx,
	float * d_Chxey, float * d_Chxez,
	float * d_Chyez, float * d_Chyex,
	float * d_Chzex, float * d_Chzey,
	float pml_x_n, float pml_y_n, float pml_z_n, float pml_x_p, float pml_y_p, float pml_z_p,
	float dx, float dy, float dz, float dt,
	float eps_0, float pi, float mu_0) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;

	//index of current thread
	int idx = i + j * NX + k * NX * NY;

	int k_real = k + gpu_offset;

	//order of the pml
	float npml = 3;

	//distance from a certain point to the pml - medium interface, x direction
	float roe_x, rom_x;
	//distance from a certain point to the pml - medium interface, y direction
	float roe_y, rom_y;
	//distance from a certain point to the pml - medium interface, z direction
	float roe_z, rom_z;

	//factor to scale the conductivities
	float sigma_factor = 1.5;

	//factor to scale the stretched coordinates (poles?)
	float kappa_max = 5;

	//factor to scale the stetched coordinates, maximum and minimum
	float alpha_max = 0.05;
	float alpha_min = 0.00;

	//optimized conductivity value for the x direction
	float sigma_opt_x = (npml + 1) / (150 * pi * 1 * dx);
	//optimized conductivity value for the y direction
	float sigma_opt_y = (npml + 1) / (150 * pi * 1 * dy);
	//optimized conductivity value for the z direction
	float sigma_opt_z = (npml + 1) / (150 * pi * 1 * dz);

	//maximum optimized conductivity value for lower x direction
	float sigma_max_x_n = sigma_factor * sigma_opt_x;
	//maximum optimized conductivity value for lower y direction
	float sigma_max_y_n = sigma_factor * sigma_opt_y;
	//maximum optimized conductivity value for lower z direction
	float sigma_max_z_n = sigma_factor * sigma_opt_z;

	//maximum optimized conductivity value for higher x direction
	float sigma_max_x_p = sigma_factor * sigma_opt_x;
	//maximum optimized conductivity value for higher y direction
	float sigma_max_y_p = sigma_factor * sigma_opt_y;
	//maximum optimized conductivity value for higher z direction
	float sigma_max_z_p = sigma_factor * sigma_opt_z;

	//auxiliar scaled electric conductivities, for lower and higher x
	float sigma_pex_x_n, sigma_pex_x_p;
	//auxiliar scaled magnetic conductivities, for lower and higher x
	float sigma_pmx_x_n, sigma_pmx_x_p;
	//auxiliar scaled electric conductivities, for lower and higher y
	float sigma_pey_y_n, sigma_pey_y_p;
	//auxiliar scaled magnetic conductivities, for lower and higher y
	float sigma_pmy_y_n, sigma_pmy_y_p;
	//auxiliar scaled electric conductivities, for lower and higher z
	float sigma_pez_z_n, sigma_pez_z_p;
	//auxiliar scaled magnetic conductivities, for lower and higher z
	float sigma_pmz_z_n, sigma_pmz_z_p;

	//auxiliar stretching electric kappa, for lower and higher x
	float kappa_ex_x_n, kappa_ex_x_p;
	//auxiliar stretching magnetic kappa, for lower and higher x
	float kappa_mx_x_n, kappa_mx_x_p;
	//auxiliar stretching electric kappa, for lower and higher y
	float kappa_ey_y_n, kappa_ey_y_p;
	//auxiliar stretching magnetic kappa, for lower and higher y
	float kappa_my_y_n, kappa_my_y_p;
	//auxiliar stretching electric kappa, for lower and higher z
	float kappa_ez_z_n, kappa_ez_z_p;
	//auxiliar stretching magnetic kappa, for lower and higher z
	float kappa_mz_z_n, kappa_mz_z_p;

	//auxiliar stretching electric alpha, for lower and higher x
	float alpha_ex_x_n, alpha_ex_x_p;
	//auxiliar stretching magnetic alpha, for lower and higher x
	float alpha_mx_x_n, alpha_mx_x_p;
	//auxiliar stretching electric alpha, for lower and higher y
	float alpha_ey_y_n, alpha_ey_y_p;
	//auxiliar stretching magnetic alpha, for lower and higher y
	float alpha_my_y_n, alpha_my_y_p;
	//auxiliar stretching electric alpha, for lower and higher z
	float alpha_ez_z_n, alpha_ez_z_p;
	//auxiliar stretching magnetic alpha, for lower and higher z
	float alpha_mz_z_n, alpha_mz_z_p;

	__syncthreads();

	if (idx < (NX*NY*NZ_N)) {

		//CPML XN: defining the parameters values
		//All the other pml regions folow the same comments
		if ((i >= (0)) && (i < (pml_x_n)) && (j >= (0)) && (j < (NYY)) && (k_real >= (0)) && (k_real < (NZ))) {


			//distance
			roe_x = (pml_x_n - i - 0.75) / pml_x_n;
			rom_x = (pml_x_n - i - 0.25) / pml_x_n;

			//scaled electric conductivity
			sigma_pex_x_n = sigma_max_x_n * pow(roe_x, npml);
			//scaled magnetic conductivity
			sigma_pmx_x_n = sigma_max_x_n * pow(rom_x, npml);
			sigma_pmx_x_n = sigma_pmx_x_n * (mu_0 / eps_0);

			//scale electric kappa stretcing parameter
			kappa_ex_x_n = 1 + (kappa_max - 1) * pow(roe_x, npml);
			//scale magnetic kappa stretcing parameter
			kappa_mx_x_n = 1 + (kappa_max - 1) * pow(rom_x, npml);

			//scale eletric alpha stretcing parameter
			alpha_ex_x_n = alpha_min + (alpha_max - alpha_min) * (1 - roe_x);
			//scale magnetic alpha stretcing parameter
			alpha_mx_x_n = alpha_min + (alpha_max - alpha_min) * (1 - rom_x);
			alpha_mx_x_n = alpha_mx_x_n * (mu_0 / eps_0);


			//CPML coefficients
			d_cpml_b_ex[i] = exp((-dt / eps_0)*((sigma_pex_x_n / kappa_ex_x_n) + alpha_ex_x_n));

			d_cpml_a_ex[i] = (1 / dx) * (d_cpml_b_ex[i] - 1) * sigma_pex_x_n / (kappa_ex_x_n * (sigma_pex_x_n + kappa_ex_x_n * alpha_ex_x_n));

			d_cpml_b_mx[i] = exp((-dt / mu_0)*((sigma_pmx_x_n / kappa_mx_x_n) + alpha_mx_x_n));

			d_cpml_a_mx[i] = (1 / dx) * (d_cpml_b_mx[i] - 1) * sigma_pmx_x_n / (kappa_mx_x_n * (sigma_pmx_x_n + kappa_mx_x_n * alpha_mx_x_n));

			d_cpsi_eyx[idx] = d_Ceyhz[idx] * dx;

			d_cpsi_ezx[idx] = d_Cezhy[idx] * dx;

			d_cpsi_hyx[idx] = d_Chyez[idx] * dx;

			d_cpsi_hzx[idx] = d_Chzey[idx] * dx;

			//Adusting the FDTD coefficients
			d_Ceyhz[idx] = d_Ceyhz[idx] / kappa_ex_x_n;
			d_Cezhy[idx] = d_Cezhy[idx] / kappa_ex_x_n;

			d_Chyez[idx] = d_Chyez[idx] / kappa_mx_x_n;
			d_Chzey[idx] = d_Chzey[idx] / kappa_mx_x_n;
		}

		__syncthreads();
		//CPML XP: defining the parameters values
		if ((i >= (NXX - pml_x_p)) && (i < (NXX)) && (j >= (0)) && (j < (NYY)) && (k_real >= (0)) && (k_real < (NZ))) {

			//roe_x = (i - (NX - pml_x_p - 1) - 0.75) / pml_x_p;
			//rom_x = (i - (NX - pml_x_p - 1) - 0.25) / pml_x_p;

			roe_x = (i - (NXX - pml_x_p - 1) - 0.75) / pml_x_p;
			rom_x = (i - (NXX - pml_x_p - 1) - 0.25) / pml_x_p;

			sigma_pex_x_p = sigma_max_x_p * pow(roe_x, npml);

			sigma_pmx_x_p = sigma_max_x_p * pow(rom_x, npml);
			sigma_pmx_x_p = sigma_pmx_x_p * (mu_0 / eps_0);

			kappa_ex_x_p = 1 + (kappa_max - 1) * pow(roe_x, npml);

			kappa_mx_x_p = 1 + (kappa_max - 1) * pow(rom_x, npml);

			alpha_ex_x_p = alpha_min + (alpha_max - alpha_min) * (1 - roe_x);

			alpha_mx_x_p = alpha_min + (alpha_max - alpha_min) * (1 - rom_x);
			alpha_mx_x_p = alpha_mx_x_p * (mu_0 / eps_0);

			d_cpml_b_ex[i] = exp((-dt / eps_0)*((sigma_pex_x_p / kappa_ex_x_p) + alpha_ex_x_p));

			d_cpml_a_ex[i] = (1 / dx) * (d_cpml_b_ex[i] - 1) * sigma_pex_x_p / (kappa_ex_x_p * (sigma_pex_x_p + kappa_ex_x_p * alpha_ex_x_p));

			d_cpml_b_mx[i] = exp((-dt / mu_0)*((sigma_pmx_x_p / kappa_mx_x_p) + alpha_mx_x_p));

			d_cpml_a_mx[i] = (1 / dx) * (d_cpml_b_mx[i] - 1) * sigma_pmx_x_p / (kappa_mx_x_p * (sigma_pmx_x_p + kappa_mx_x_p * alpha_mx_x_p));

			d_cpsi_eyx[idx] = d_Ceyhz[idx] * dx;

			d_cpsi_ezx[idx] = d_Cezhy[idx] * dx;

			d_cpsi_hyx[idx] = d_Chyez[idx] * dx;

			d_cpsi_hzx[idx] = d_Chzey[idx] * dx;

			d_Ceyhz[idx] = d_Ceyhz[idx] / kappa_ex_x_p;
			d_Cezhy[idx] = d_Cezhy[idx] / kappa_ex_x_p;

			d_Chyez[idx] = d_Chyez[idx] / kappa_mx_x_p;
			d_Chzey[idx] = d_Chzey[idx] / kappa_mx_x_p;


		}

		__syncthreads();
		//CPML YN: defining the parameters values
		if ((i >= (0)) && (i < (NXX)) && (j >= (0)) && (j < (pml_y_n)) && (k_real >= (0)) && (k_real < (NZ))) {



			roe_y = (pml_y_n - j - 0.75) / pml_y_n;
			rom_y = (pml_y_n - j - 0.25) / pml_y_n;

			sigma_pey_y_n = sigma_max_y_n * pow(roe_y, npml);

			sigma_pmy_y_n = sigma_max_y_n * pow(rom_y, npml);
			sigma_pmy_y_n = sigma_pmy_y_n * (mu_0 / eps_0);

			kappa_ey_y_n = 1 + (kappa_max - 1) * pow(roe_y, npml);

			kappa_my_y_n = 1 + (kappa_max - 1) * pow(rom_y, npml);

			alpha_ey_y_n = alpha_min + (alpha_max - alpha_min) * (1 - roe_y);

			alpha_my_y_n = alpha_min + (alpha_max - alpha_min) * (1 - rom_y);
			alpha_my_y_n = alpha_my_y_n * (mu_0 / eps_0);

			d_cpml_b_ey[j] = exp((-dt / eps_0)*((sigma_pey_y_n / kappa_ey_y_n) + alpha_ey_y_n));

			d_cpml_a_ey[j] = (1 / dy) * (d_cpml_b_ey[j] - 1) * sigma_pey_y_n / (kappa_ey_y_n * (sigma_pey_y_n + kappa_ey_y_n * alpha_ey_y_n));

			d_cpml_b_my[j] = exp((-dt / mu_0)*((sigma_pmy_y_n / kappa_my_y_n) + alpha_my_y_n));

			d_cpml_a_my[j] = (1 / dy) * (d_cpml_b_my[j] - 1) * sigma_pmy_y_n / (kappa_my_y_n * (sigma_pmy_y_n + kappa_my_y_n * alpha_my_y_n));

			d_cpsi_exy[idx] = d_Cexhz[idx] * dy;

			//Obs
			d_cpsi_ezy[idx] = d_Cezhx[idx] * dy;

			d_cpsi_hxy[idx] = d_Chxez[idx] * dy;

			d_cpsi_hzy[idx] = d_Chzex[idx] * dy;


			d_Cexhz[idx] = d_Cexhz[idx] / kappa_ey_y_n;

			d_Cezhx[idx] = d_Cezhx[idx] / kappa_ey_y_n;

			d_Chxez[idx] = d_Chxez[idx] / kappa_my_y_n;

			d_Chzex[idx] = d_Chzex[idx] / kappa_my_y_n;

		}

		__syncthreads();
		//CPML YP: defining the parameters values
		if ((i >= (0)) && (i < (NXX)) && (j >= (NYY - pml_y_p)) && (j < (NYY)) && (k_real >= (0)) && (k_real < (NZ))) {
			//for (int k = 0; k < (NZ); k++) {
			//for (int j = (NYY - pml_y_p); j < (NYY); j++) {
			//for (int i = 0; i < (NXX); i++) {


			roe_y = (j - (NYY - pml_y_p - 1) - 0.75) / pml_y_p;
			rom_y = (j - (NYY - pml_y_p - 1) - 0.25) / pml_y_p;

			sigma_pey_y_p = sigma_max_y_p * pow(roe_y, npml);

			sigma_pmy_y_p = sigma_max_y_p * pow(rom_y, npml);
			sigma_pmy_y_p = sigma_pmy_y_p * (mu_0 / eps_0);

			kappa_ey_y_p = 1 + (kappa_max - 1) * pow(roe_y, npml);

			kappa_my_y_p = 1 + (kappa_max - 1) * pow(rom_y, npml);

			alpha_ey_y_p = alpha_min + (alpha_max - alpha_min) * (1 - roe_y);

			alpha_my_y_p = alpha_min + (alpha_max - alpha_min) * (1 - rom_y);
			alpha_my_y_p = alpha_my_y_p * (mu_0 / eps_0);

			d_cpml_b_ey[j] = exp((-dt / eps_0)*((sigma_pey_y_p / kappa_ey_y_p) + alpha_ey_y_p));

			d_cpml_a_ey[j] = (1 / dy) * (d_cpml_b_ey[j] - 1) * sigma_pey_y_p / (kappa_ey_y_p * (sigma_pey_y_p + kappa_ey_y_p * alpha_ey_y_p));

			d_cpml_b_my[j] = exp((-dt / mu_0)*((sigma_pmy_y_p / kappa_my_y_p) + alpha_my_y_p));

			d_cpml_a_my[j] = (1 / dy) * (d_cpml_b_my[j] - 1) * sigma_pmy_y_p / (kappa_my_y_p * (sigma_pmy_y_p + kappa_my_y_p * alpha_my_y_p));

			d_cpsi_exy[idx] = d_Cexhz[idx] * dy;

			//Obs
			d_cpsi_ezy[idx] = d_Cezhx[idx] * dy;

			d_cpsi_hxy[idx] = d_Chxez[idx] * dy;

			d_cpsi_hzy[idx] = d_Chzex[idx] * dy;


			d_Cexhz[idx] = d_Cexhz[idx] / kappa_ey_y_p;

			d_Cezhx[idx] = d_Cezhx[idx] / kappa_ey_y_p;

			d_Chxez[idx] = d_Chxez[idx] / kappa_my_y_p;

			d_Chzex[idx] = d_Chzex[idx] / kappa_my_y_p;

		}

		__syncthreads();
		//CPML ZN: defining the parameters values
		if ((i >= (0)) && (i < (NXX)) && (j >= (0)) && (j < (NYY)) && (k_real >= (0)) && (k_real < (pml_z_n))) {


			roe_z = (pml_z_n - k_real - 0.75) / pml_z_n;
			rom_z = (pml_z_n - k_real - 0.25) / pml_z_n;

			sigma_pez_z_n = sigma_max_z_n * pow(roe_z, npml);

			sigma_pmz_z_n = sigma_max_z_n * pow(rom_z, npml);
			sigma_pmz_z_n = sigma_pmz_z_n * (mu_0 / eps_0);

			kappa_ez_z_n = 1 + (kappa_max - 1) * pow(roe_z, npml);

			kappa_mz_z_n = 1 + (kappa_max - 1) * pow(rom_z, npml);

			alpha_ez_z_n = alpha_min + (alpha_max - alpha_min) * (1 - roe_z);

			alpha_mz_z_n = alpha_min + (alpha_max - alpha_min) * (1 - rom_z);
			alpha_mz_z_n = alpha_mz_z_n * (mu_0 / eps_0);

			d_cpml_b_ez[k] = exp((-dt / eps_0)*((sigma_pez_z_n / kappa_ez_z_n) + alpha_ez_z_n));

			d_cpml_a_ez[k] = (1 / dz) * (d_cpml_b_ez[k] - 1) * sigma_pez_z_n / (kappa_ez_z_n * (sigma_pez_z_n + kappa_ez_z_n * alpha_ez_z_n));

			d_cpml_b_mz[k] = exp((-dt / mu_0)*((sigma_pmz_z_n / kappa_mz_z_n) + alpha_mz_z_n));

			d_cpml_a_mz[k] = (1 / dz) * (d_cpml_b_mz[k] - 1) * sigma_pmz_z_n / (kappa_mz_z_n * (sigma_pmz_z_n + kappa_mz_z_n * alpha_mz_z_n));

			d_cpsi_exz[idx] = d_Cexhy[idx] * dz;

			d_cpsi_eyz[idx] = d_Ceyhx[idx] * dz;

			d_cpsi_hxz[idx] = d_Chxey[idx] * dz;

			d_cpsi_hyz[idx] = d_Chyex[idx] * dz;

			d_Cexhy[idx] = d_Cexhy[idx] / kappa_ez_z_n;

			d_Ceyhx[idx] = d_Ceyhx[idx] / kappa_ez_z_n;

			d_Chxey[idx] = d_Chxey[idx] / kappa_mz_z_n;

			d_Chyex[idx] = d_Chyex[idx] / kappa_mz_z_n;

		}

		__syncthreads();

		//CPML ZP: defining the parameters values
		if ((i >= (0)) && (i < (NXX)) && (j >= (0)) && (j < (NYY)) && (k_real >= (NZ - pml_z_p)) && (k_real < (NZ))) {

			roe_z = (k_real - (NZ - pml_z_p - 1) - 0.75) / pml_z_p;
			rom_z = (k_real - (NZ - pml_z_p - 1) - 0.25) / pml_z_p;

			sigma_pez_z_p = sigma_max_z_p * pow(roe_z, npml);

			sigma_pmz_z_p = sigma_max_z_p * pow(rom_z, npml);
			sigma_pmz_z_p = sigma_pmz_z_p * (mu_0 / eps_0);

			kappa_ez_z_p = 1 + (kappa_max - 1) * pow(roe_z, npml);

			kappa_mz_z_p = 1 + (kappa_max - 1) * pow(rom_z, npml);

			alpha_ez_z_p = alpha_min + (alpha_max - alpha_min) * (1 - roe_z);

			alpha_mz_z_p = alpha_min + (alpha_max - alpha_min) * (1 - rom_z);
			alpha_mz_z_p = alpha_mz_z_p * (mu_0 / eps_0);

			d_cpml_b_ez[k] = exp((-dt / eps_0)*((sigma_pez_z_p / kappa_ez_z_p) + alpha_ez_z_p));

			d_cpml_a_ez[k] = (1 / dz) * (d_cpml_b_ez[k] - 1) * sigma_pez_z_p / (kappa_ez_z_p * (sigma_pez_z_p + kappa_ez_z_p * alpha_ez_z_p));

			d_cpml_b_mz[k] = exp((-dt / mu_0)*((sigma_pmz_z_p / kappa_mz_z_p) + alpha_mz_z_p));

			d_cpml_a_mz[k] = (1 / dz) * (d_cpml_b_mz[k] - 1) * sigma_pmz_z_p / (kappa_mz_z_p * (sigma_pmz_z_p + kappa_mz_z_p * alpha_mz_z_p));

			d_cpsi_exz[idx] = d_Cexhy[idx] * dz;

			d_cpsi_eyz[idx] = d_Ceyhx[idx] * dz;

			d_cpsi_hxz[idx] = d_Chxey[idx] * dz;

			d_cpsi_hyz[idx] = d_Chyex[idx] * dz;

			d_Cexhy[idx] = d_Cexhy[idx] / kappa_ez_z_p;

			d_Ceyhx[idx] = d_Ceyhx[idx] / kappa_ez_z_p;

			d_Chxey[idx] = d_Chxey[idx] / kappa_mz_z_p;

			d_Chyex[idx] = d_Chyex[idx] / kappa_mz_z_p;
		}

	}


}


//adjust the voltage array
__global__ void sampledAdj(int NUMDEV, float * d_sampled, float * d_sampled0, int n_t_steps){

	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    float sampled_sum = 0;


    if(threadId < n_t_steps){

        for(int i = 0; i < NUMDEV; i++){

           sampled_sum  = sampled_sum + d_sampled0[threadId + n_t_steps*i];

        }

        d_sampled[threadId] = sampled_sum;
    }

}

//calculate voltage on multiple gp
__global__ void voltMgpu(int volt_NX, int volt_NY, int volt_NZ, int volt_NZ_N, int gpu_offset,
	int sampled_voltage_is, int sampled_voltage_js, int sampled_voltage_ks, int sampled_voltage_ie, int sampled_voltage_je, int sampled_voltage_ke,
    float * d_volt_tran, float * E,
	float dx, float dy, float dz, int n_t_steps, int direction){

    float csuf = 0;

	float _is, _ie;
    float _js, _je;
    float _ks, _ke;

	float voltage = 0;

	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	int idx_voltage = 0;

    _is = (float)sampled_voltage_is;
	_js = (float)sampled_voltage_js;
	_ks = (float)sampled_voltage_ks;

	_ie = (float)sampled_voltage_ie;
	_je = (float)sampled_voltage_je;
	_ke = (float)sampled_voltage_ke;




    if(threadId < n_t_steps) {

        switch(direction){

            case 1:

                csuf = (-dx) / (((_je - _js + 1) * (_ke - _ks + 1)));

                for (int k = 0; k < volt_NZ_N; k++) {

                    for (int j = sampled_voltage_js; j <= (sampled_voltage_je); j++) {

                        for (int i = sampled_voltage_is; i < (sampled_voltage_ie); i++) {

                            if( ((k+gpu_offset) >= sampled_voltage_ks) && ((k+gpu_offset) <= sampled_voltage_ke) ) {

                                    //3d to 1d index
                                    idx_voltage = (i - sampled_voltage_is) + (j - sampled_voltage_js) * volt_NX + (k) * volt_NX * volt_NY + threadId * volt_NX * volt_NY * volt_NZ_N;

                                    voltage = voltage + E[idx_voltage];

                            }
                        }
                    }
                }

                d_volt_tran[threadId] = voltage*csuf;

                break;

            case 2:

                csuf = (-dy) / (((_ke - _ks + 1) * (_ie - _is + 1)));

                for( int k = 0; k < volt_NZ_N; k++){

                    for (int j = sampled_voltage_js; j < (sampled_voltage_je); j++) {

                        for (int i = sampled_voltage_is; i <= (sampled_voltage_ie); i++) {

                            if( ((k+gpu_offset) >= sampled_voltage_ks) && ((k+gpu_offset) <= sampled_voltage_ke) ) {

                                    //3d to 1d index
                                    idx_voltage = (i - sampled_voltage_is) + (j - sampled_voltage_js) * volt_NX + (k) * volt_NX * volt_NY + threadId * volt_NX * volt_NY * volt_NZ_N;

                                    voltage = voltage + E[idx_voltage];
                            }
                        }
                    }
                }

                d_volt_tran[threadId] = voltage*csuf;

                break;

            case 3:

                csuf = (-dz) / (((_ie - _is + 1) * (_je - _js + 1)));

                for( int k = 0; k < volt_NZ_N; k++){

                    for (int j = sampled_voltage_js; j <= (sampled_voltage_je); j++) {

                        for (int i = sampled_voltage_is; i <= (sampled_voltage_ie); i++) {

                            if( ((k+gpu_offset) >= sampled_voltage_ks) && ((k+gpu_offset) < sampled_voltage_ke)   ) {

                                    //3d to 1d index
                                    idx_voltage = (i - sampled_voltage_is) + (j - sampled_voltage_js) * volt_NX + (k) * volt_NX * volt_NY + threadId * volt_NX * volt_NY * volt_NZ_N;

                                    voltage = voltage + E[idx_voltage];
                            }
                        }
                    }
                }

                d_volt_tran[threadId] = voltage*csuf;

                break;
        }


    }


}

//calculate current on multiple gpu
__global__ void currMgpu(int current_NX, int current_NY, int current_NZ, int current_NZ_N, int gpu_offset,
	int sampled_current_is, int sampled_current_js, int sampled_current_ks, int sampled_current_ie, int sampled_current_je, int sampled_current_ke,
	float * d_current_tran, float * Hx, float * Hy, float * Hz,
	float dx, float dy, float dz, int n_t_steps, int direction){

	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    int k_real;
	int idx;
    float sum_xp = 0;
	float sum_xn = 0;
	float sum_yp = 0;
	float sum_yn = 0;
	float sum_zp = 0;
	float sum_zn = 0;



    if((threadId < n_t_steps)) {

        switch(direction){

            case 1:


                for(int k=0; k < current_NZ_N; k++){

                    k_real = k + gpu_offset;

                    for (int j = 1; j < (current_NY); j++) {

                        //Calculate sampled current sum_yp
                        if(k_real == (sampled_current_ks - 1)){

                            //3d to 1d index
                            idx = ((current_NX-1) -1) + (j)* current_NX + (k) * current_NX *current_NY + threadId * current_NX *current_NY * current_NZ_N;

                            sum_yp = sum_yp + Hy[idx];

                        }

                        //Calculate sampled current sum_yn
                        if(k_real == (sampled_current_ke)){

                             //3d to 1d index
                            idx = ((current_NX-1) - 1) + (j)* current_NX + (k)* current_NX * current_NY + threadId * current_NX *current_NY * current_NZ_N;

                            sum_yn = sum_yn + Hy[idx];

                        }

                    }

                    //Calculate sampled current sum_zp
                    //Calculate sampled current sum_zn
                    if((k_real >= sampled_current_ks) && (k_real <= sampled_current_ke)){


                        //3d to 1d index
                        idx = ((current_NX-1) - 1) + (current_NY-1) * current_NX + k * current_NX * current_NY+ threadId * current_NX *current_NY * current_NZ_N;

                        sum_zp = sum_zp + Hz[idx];

                        //3d to 1d index
                        idx = ((current_NX - 1) -1) + (0) * current_NX + k * current_NX * current_NY + threadId * current_NX *current_NY * current_NZ_N;

                        sum_zn = sum_zn + Hz[idx];

                    }

                }

                d_current_tran[threadId] = dy*sum_yp + dz*sum_zp - dy*sum_yn - dz*sum_zn;

                break;

            case 2:

                for (int k = 0; k < current_NZ_N; k++) {

                    k_real = k + gpu_offset;

                    //Calculate sampled current sum_zp
                    //Calculate sampled current sum_zn
                    if((k_real >= sampled_current_ks) && (k_real <= sampled_current_ke)){

                        //3d to 1d index
                        idx = (0) + ((current_NY - 1) - 1) * current_NX + k * current_NX * current_NY + threadId * current_NX * current_NY * current_NZ_N;

                        sum_zp = sum_zp + Hz[idx];

                        //3d to 1d index
                        idx = (current_NX-1) + ((current_NY-1) - 1) * current_NX + k * current_NX * current_NY + threadId * current_NX * current_NY * current_NZ_N;

                        sum_zn = sum_zn + Hz[idx];

                    }

                    for (int i = 1; i < current_NX; i++) {

                        //Calculate sampled current sum_xp
                        if((k_real == sampled_current_ke)){

                            //3d to 1d index
                            idx = i + ((current_NY - 1) - 1) * current_NX + k * current_NX * current_NY + threadId * current_NX * current_NY * current_NZ_N;

                            sum_xp = sum_xp + Hx[idx];

                        }

                        //Calculate sampled current sum_xn
                        if((k_real == (sampled_current_ks - 1))){

                            //3d to 1d index
                            idx = i + ((current_NY - 1)- 1)* current_NX + (k) * current_NX * current_NY + threadId * current_NX * current_NY * current_NZ_N;

                            sum_xn = sum_xn + Hx[idx];

                        }
                    }

                }

                d_current_tran[threadId] = dz * sum_zp + dx*sum_xp - dz*sum_zn - dx*sum_xn;

                break;

            case 3:

               for (int k = 0; k < current_NZ_N; k++) {

                    k_real = k + gpu_offset;

                    for (int i = 1; i < current_NX; i++) {

                        //Calculate sampled current sum_xp
                        if((k_real == (sampled_current_ke - 1))){
                                //3d to 1d index
                                idx = i + (0) * current_NX + (k) * current_NX * current_NY + threadId * current_NX * current_NY * current_NZ_N;

                                sum_xp = sum_xp + Hx[idx];
                        }

                        //Calculate sampled current sum_xn
                        if((k_real == (sampled_current_ke - 1))){

                            //3d to 1d index
                            idx = i + (current_NY - 1)* current_NX +  (k) * current_NX * current_NY + threadId * current_NX * current_NY * current_NZ_N;

                            sum_xn = sum_xn + Hx[idx];

                        }
                    }

                    for (int j = 1; j < (current_NY); j++) {

                        //Calculate sampled current sum_yp
                        if((k_real == (sampled_current_ke - 1))){

                            //3d to 1d index
                            idx = (current_NX - 1)+(j)* current_NX + (k) * current_NX * current_NY + threadId * current_NX * current_NY * current_NZ_N;

                            sum_yp = sum_yp + Hy[idx];

                        }

                        //Calculate sampled current sum_yn
                        if((k_real == (sampled_current_ke - 1))){

                            //3d to 1d index
                            idx = (0) + (j)* current_NX + (k) * current_NX * current_NY + threadId * current_NX * current_NY * current_NZ_N;

                            sum_yn = sum_yn + Hy[idx];

                        }
                    }

            }

                d_current_tran[threadId] = dy * sum_yp + dx*sum_xp - dy*sum_yn - dx*sum_xn;

                break;

        }

    }

}

__global__ void correcI(float * I, int n_t_steps) {

	//helper to calculate threads global index
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	//float aux;

    if(threadId < n_t_steps){

        if (threadId < (n_t_steps - 1)) {

            I[threadId] = (I[threadId] + I[threadId + 1]) / 2;

        } else {

            I[threadId] = (I[threadId] + I[threadId - 1]) / 2;
        }
    }
}

__global__ void c_zinS11(cuFloatComplex *V, cuFloatComplex *I, int n_t_steps, cuFloatComplex *a, cuFloatComplex *b, cuFloatComplex *s11, cuFloatComplex *zin, cuFloatComplex z0, float dt) {

	//helper to calculate threads global index
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	//auxiliar for the divisor
	cuFloatComplex aux;
	aux.y = 0;
	aux.x = 0;

	cuFloatComplex aux2 = make_cuFloatComplex(1, 0);

	//cuFloatComplex aux3 = make_cuFloatComplex(-1, 0);

	__syncthreads();

	if (threadId < (n_t_steps / 2 + 1)) {

		aux.x = 2 * sqrt(cuCrealf(z0));

		__syncthreads();

		a[threadId] = cuCdivf(cuCaddf(V[threadId], cuCmulf(z0, I[threadId])), aux);
		b[threadId] = cuCdivf(cuCsubf(V[threadId], cuCmulf(z0, I[threadId])), aux);

		__syncthreads();

		s11[threadId] = cuCdivf(b[threadId], a[threadId]);

		__syncthreads();

		//zin[threadId] = cuCmulf(aux3,cuCmulf(z0, cuCdivf(cuCaddf(aux2, s11[threadId]), cuCsubf(s11[threadId], aux2))));
		zin[threadId] = cuCmulf(z0, cuCdivf(cuCaddf(aux2, s11[threadId]), cuCsubf(aux2, s11[threadId])));
		//zin[threadId] = cuCdivf(V[threadId],I[threadId]);

	}


}

void sectorPoints(float raio, float sec_angle, float rot_angle, float2 ponto1, float2 ponto2, float2 ponto0,
	int *px0, int *py0, int *pz0, int *px1, int *py1, int *px2, int *py2,
	float gap, float trans_x, float trans_y, int i,
	int pml_x_n, int pml_y_n, int pml_z_n,
	int air_buff_x_n, int air_buff_y_n, int air_buff_z_n,
	float dx, float dy, float dz) {

	//adjust the rotated points
	float desloc;
	//p1 of each rotated point
	float2 p1_init;


	ponto2.x = -((raio * cosf(sec_angle / 2)) + gap);
	ponto2.y = (raio * sinf(sec_angle / 2));

	ponto0.x = -((raio * cosf(sec_angle / 2)) + gap);
	ponto0.y = -(raio * sinf(sec_angle / 2));


	ponto0 = rotate_point(ponto0, (rot_angle));
	p1_init = rotate_point(ponto1, (rot_angle));
	ponto2 = rotate_point(ponto2, (rot_angle));

	desloc = gap - fabs(p1_init.x);

	if (i < 3) {

		ponto0.x = ponto0.x - desloc;
		p1_init.x = p1_init.x - desloc;
		ponto2.x = ponto2.x - desloc;

		ponto0.y = ponto0.y + desloc*tanf(rot_angle);
		p1_init.y = p1_init.y + desloc*tanf(rot_angle);
		ponto2.y = ponto2.y + desloc*tanf(rot_angle);

	}
	else {

		ponto0.x = ponto0.x + desloc;
		p1_init.x = p1_init.x + desloc;
		ponto2.x = ponto2.x + desloc;

		ponto0.y = ponto0.y - desloc*tanf(rot_angle);
		p1_init.y = p1_init.y - desloc*tanf(rot_angle);
		ponto2.y = ponto2.y - desloc*tanf(rot_angle);

	}


	p1_init.x = trans_point(p1_init.x, -(trans_x));
	p1_init.y = trans_point(p1_init.y, -trans_y);

	ponto2.x = trans_point(ponto2.x, -(trans_x));
	ponto2.y = trans_point(ponto2.y, -trans_y);

	ponto0.x = trans_point(ponto0.x, -(trans_x));
	ponto0.y = trans_point(ponto0.y, -trans_y);

	*px0 = air_buff_x_n + pml_x_n + calc_index(ponto0.x, dx);
	*py0 = air_buff_y_n + pml_y_n + calc_index(ponto0.y, dy);

	*px1 = air_buff_x_n + pml_x_n + calc_index(p1_init.x, dx);
	*py1 = air_buff_y_n + pml_y_n + calc_index(p1_init.y, dy);

	*px2 = air_buff_x_n + pml_x_n + calc_index(ponto2.x, dx);
	*py2 = air_buff_y_n + pml_y_n + calc_index(ponto2.y, dy);

}

void saveComplex(cuFloatComplex * Q, int n_t_steps, float dt, int fac) {

	FILE *file;
	char filename[20];
	sprintf(filename, "results%d.dat", fac);
	file = fopen(filename, "w");
	float freq;

	for (int i = 0; i < (n_t_steps / 2 + 1); i++) {

		freq = (i * (1 / dt)) / n_t_steps;

		fprintf(file, "%e %e\n", freq, 20*log10(cuCabsf(Q[i])) );

	}

	fclose(file);
}

void saveImpReal(cuFloatComplex * Q, int n_t_steps, float dt, int fac) {

	FILE *file;
	char filename[20];
	sprintf(filename, "results%d.dat", fac);
	file = fopen(filename, "w");
	float freq;

	for (int i = 0; i < (n_t_steps / 2 + 1); i++) {

		freq = (i * (1 / dt)) / n_t_steps;
		fprintf(file, "%e %e \n", freq, cuCrealf(Q[i]));

	}

	fclose(file);
}

void saveImpImag(cuFloatComplex * Q, int n_t_steps, float dt, int fac) {

	FILE *file;
	char filename[20];
	sprintf(filename, "results%d.dat", fac);
	file = fopen(filename, "w");
	float freq;

	for (int i = 0; i < (n_t_steps / 2 + 1); i++) {

		freq = (i * (1 / dt)) / n_t_steps;
		fprintf(file, "%e %e \n", freq, cuCimagf(Q[i]));

	}

	fclose(file);
}

void saveFilePm3d(int NX, int NXX, int NY, int NYY, int NZ_N, float *h_Q, int fac, float k_index) {

	FILE *file;
	char filename[20];
	sprintf(filename, "results%d.dat", fac);
	file = fopen(filename, "w");

	int idx;

	for (int j = 0; j < NYY; j++) {
		for (int i = 0; i < NXX; i++) {

			idx = i + j * NX + k_index * NX * NY;

			fprintf(file, "%e ", h_Q[idx]);

		}

		//to the NXt line
		fprintf(file, "\n");

	}

	fclose(file);



}

void saveTDomain(int n_t_steps, float *h_Q, int fac) {

	FILE *file;
	char filename[20];
	sprintf(filename, "results%d.dat", fac);
	file = fopen(filename, "w");

	for (int i = 0; i < n_t_steps; i++) {

		fprintf(file, "%e \n", (h_Q[i]));

	}

	fclose(file);

}

void saveTDomain2(int n_t_steps, float * h_Q, int fac, int gpu) {

	FILE *file;
	char filename[20];
	sprintf(filename, "results%d.dat", fac);
	file = fopen(filename, "w");

	for (int i = 0; i < n_t_steps; i++) {

		fprintf(file, "%e \n", (h_Q[i + gpu * n_t_steps]));

	}

	fclose(file);

}


void saveTDomainu(int n_t_steps, float *h_Q, int fac) {

	FILE *file;
	char filename[20];
	sprintf(filename, "results%d.dat", fac);
	file = fopen(filename, "w");

	for (int i = 0; i < n_t_steps; i++) {

		fprintf(file, "%e \n", (h_Q[i+4*n_t_steps]));

	}

	fclose(file);

}



/*
float genRadius(float xi_min, float xi_max) {

	std::random_device rdev1{};
	std::default_random_engine	e1{ rdev1() };
	std::uniform_real_distribution<double> d{ xi_min, xi_max };
	return d(e1);

}

int genAngle(int xi_min, int xi_max) {

	std::random_device rdev2{};
	std::default_random_engine	e2{ rdev2() };
	std::uniform_int_distribution<int> d{ xi_min, xi_max };
	return d(e2);

}

float genU() {
	std::random_device rdev1{};
	std::default_random_engine	e1{ rdev1() };
	std::uniform_real_distribution<double> d{ 0, 1 };
	return d(e1);
}

int genJDe(int min, int max) {

	std::random_device rdev3{};
	std::default_random_engine	e3{ rdev3() };
	std::uniform_int_distribution<int> d{ min, max };
	return d(e3);
}

void printPop(float * pop, int npop, int nvar) {

	//index do vetor população, mapeamento 2d->1d
	int idx;

	for (int j = 0; j < npop; j++)
	{
		cout << "Individuo " << j << ": " << endl;
		for (int i = 0; i < nvar / 2; i++)
		{
			idx = i + j*nvar;
			cout << "Raio " << i << ": " << pop[idx] << endl;

		}
		for (int i = nvar / 2; i < nvar; i++)
		{
			idx = i + j*nvar;
			cout << "Angulo " << i - nvar / 2 << ": " << pop[idx] << endl;

		}
		cout << endl;
	}

}

void genPop(float * pop, int npop, int nvar, float r_min, float r_max, float angle_min, float angle_max) {

	//index do vetor população, mapeamento 2d->1d
	int idx;

	//gerando pupulação inicial aleatória
	for (int j = 0; j < npop; j++)
	{
		for (int i = 0; i < nvar / 2; i++)
		{
			idx = i + j*nvar;
			//para o raio
			pop[idx] = genRadius(r_min, r_max);
		}

		for (int i = nvar / 2; i < nvar; i++)
		{
			idx = i + j*nvar;
			//para os angulos
			pop[idx] = genAngle(angle_min, angle_max);
		}
	}

}

void de_cross(float * pop, float * upop, int npop, int nvar, float F, float C, int pcut, float r_min, float r_max,
	int angle_min, int angle_max) {

	float U;
	int idx, idx1, idx2, idx3;

	int j1, j2, j3;

	U = genU();


	//gerando pupulação inicial aleatória
	for (int j = 0; j < npop; j++)
	{

		U = genU();
		//cout << U << "\t";

		do
		{
			j1 = genJDe(0, npop - 1);

		} while (j1 == j);

		do
		{
			j2 = genJDe(0, npop - 1);

		} while ((j2 == j) || (j2 == j1));

		do
		{
			j3 = genJDe(0, npop - 1);

		} while ((j3 == j) || (j3 == j1) || (j3 == j2));


		for (int i = 0; i < nvar / 2; i++)
		{
			idx = i + j*nvar;
			idx1 = i + j1*nvar;
			idx2 = i + j2*nvar;
			idx3 = i + j3*nvar;

			if ((U <= C) || (i == pcut)) {

				upop[idx] = pop[idx1] + (F * (pop[idx2] - pop[idx3]));

			}

			else
			{
				upop[idx] = pop[idx];
			}

			if (upop[idx] < r_min) {
				upop[idx] = r_min;
			}
			if (upop[idx] > r_max) {
				upop[idx] = r_max;
			}

		}

		for (int i = nvar / 2; i < nvar; i++)
		{
			idx = i + j*nvar;
			idx1 = i + j1*nvar;
			idx2 = i + j2*nvar;
			idx3 = i + j3*nvar;

			if ((U <= C) || (i == pcut)) {

				upop[idx] = pop[idx1] + (F * (pop[idx2] - pop[idx3]));

			}

			else
			{
				upop[idx] = pop[idx];
			}

			if (upop[idx] < angle_min) {
				upop[idx] = angle_min;
			}
			if (upop[idx] > angle_max) {
				upop[idx] = angle_max;
			}

		}

	}


}

void de_selec(float * pop, float * p_fobj, float * upop, float * u_fobj, int npop, int nvar) {

	//index do vetor população, mapeamento 2d->1d
	int idx;

	for (int j = 0; j < npop; j++)
	{
		if (p_fobj[j] < u_fobj[j]) {
			for (int i = 0; i < nvar; i++)
			{
				idx = i + j*nvar;
				pop[idx] = upop[idx];

			}
		}
	}

}

void avFobj(float *pop, int npop, int nvar, float * result) {

	//index do vetor população, mapeamento 2d->1d
	int idx;
	float sum1, sum2;

	for (int j = 0; j < npop; j++)
	{
		sum1 = 0;
		sum2 = 0;


		for (int i = 0; i < nvar / 2; i++)
		{
			idx = i + j*nvar;

			sum1 = sum1 + (pop[idx]) * 1000;
		}

		for (int i = nvar / 2; i < nvar; i++)
		{
			idx = i + j*nvar;

			sum2 = sum2 + (pop[idx]);
		}
		result[j] = sum1 + sum2;

	}
}
*/

void calc_otm(float * d_voltage, float * d_current, int n_t_steps, float z0, float dt, int grid, int block) {

    cuFloatComplex z;
    z.x = z0;
    z.y = 0;
    cudaSetDevice(0);
    cufftResult result;

    //Frquency variables
    cuFloatComplex *d_V, *d_I;
    cuFloatComplex *d_a, *d_b;
    cuFloatComplex *d_s11;
    cuFloatComplex *d_zin;
    //Cuda Memory allocation
    HANDLE_ERROR(cudaMalloc((void**)&d_V, sizeof(cuFloatComplex)*(n_t_steps / 2 + 1)));
    HANDLE_ERROR(cudaMalloc((void**)&d_I, sizeof(cuFloatComplex)*(n_t_steps / 2 + 1)));
    HANDLE_ERROR(cudaMalloc((void**)&d_a, sizeof(cuFloatComplex)*(n_t_steps / 2 + 1)));
    HANDLE_ERROR(cudaMalloc((void**)&d_b, sizeof(cuFloatComplex)*(n_t_steps / 2 + 1)));
    HANDLE_ERROR(cudaMalloc((void**)&d_s11, sizeof(cuFloatComplex)*(n_t_steps / 2 + 1)));
    HANDLE_ERROR(cudaMalloc((void**)&d_zin, sizeof(cuFloatComplex)*(n_t_steps / 2 + 1)));

    cuFloatComplex *h_s11;
    cuFloatComplex *h_zin;
    cuFloatComplex *h_V;
    cuFloatComplex *h_I;
    h_s11 = (cuFloatComplex*)malloc(sizeof(cuFloatComplex)*(n_t_steps / 2 + 1));
    h_zin = (cuFloatComplex*)malloc(sizeof(cuFloatComplex)*(n_t_steps / 2 + 1));
    h_V = (cuFloatComplex*)malloc(sizeof(cuFloatComplex)*(n_t_steps / 2 + 1));
    h_I = (cuFloatComplex*)malloc(sizeof(cuFloatComplex)*(n_t_steps / 2 + 1));


    //CuDA fft plan
    cufftHandle plan;
    result = cufftPlan1d(&plan, n_t_steps, CUFFT_R2C, 2);
    if (result != CUFFT_SUCCESS){ cout << "CUFFT error: Plan creation failed" << endl; return; }

    cudaDeviceSynchronize();
    //Execution of V fft
    result = cufftExecR2C(plan, d_voltage, d_V);
    if (result != CUFFT_SUCCESS){ cout << "CUFFT error: ExecC2C Forward failed 1" << endl; return; }
    cudaDeviceSynchronize();

    //Execution of I fft
    result = cufftExecR2C(plan, d_current, d_I);
    if (result != CUFFT_SUCCESS){ cout << "CUFFT error: ExecC2C Forward failed 2" << endl; return; }
    cudaDeviceSynchronize();

    cufftDestroy(plan);

    HANDLE_ERROR(cudaMemcpy(h_V, d_V, sizeof(cuFloatComplex)*(n_t_steps / 2 + 1), cudaMemcpyDeviceToHost));
    saveComplex(h_V, n_t_steps, dt, 1005);
    HANDLE_ERROR(cudaMemcpy(h_I, d_I, sizeof(cuFloatComplex)*(n_t_steps / 2 + 1), cudaMemcpyDeviceToHost));
    saveComplex(h_I, n_t_steps, dt, 1006);

    c_zinS11 << <grid, block >> > (d_V, d_I, (n_t_steps), d_a, d_b, d_s11, d_zin, z, dt);

    HANDLE_ERROR(cudaMemcpy(h_s11, d_s11, sizeof(cuFloatComplex)*(n_t_steps / 2 + 1), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(h_zin, d_zin, sizeof(cuFloatComplex)*(n_t_steps / 2 + 1), cudaMemcpyDeviceToHost));

    saveComplex(h_s11, n_t_steps, dt, 1007);
    saveImpImag(h_zin, n_t_steps, dt, 1008);
    saveImpReal(h_zin, n_t_steps, dt, 1009);

}

/*
void calc_otm2(float * d_voltage, float * d_current, int n_t_steps, cuFloatComplex z0, float dt) {

//Frquency variables
cuFloatComplex *d_V, *d_I;
cuFloatComplex *d_a, *d_b;
cuFloatComplex *d_s11;
cuFloatComplex *d_zin;

//Cuda Memory allocation
HANDLE_ERROR(cudaMalloc((void**)&d_V, sizeof(cuFloatComplex)*(n_t_steps / 2 + 1)));
HANDLE_ERROR(cudaMalloc((void**)&d_I, sizeof(cuFloatComplex)*(n_t_steps / 2 + 1)));
HANDLE_ERROR(cudaMalloc((void**)&d_a, sizeof(cuFloatComplex)*(n_t_steps / 2 + 1)));
HANDLE_ERROR(cudaMalloc((void**)&d_b, sizeof(cuFloatComplex)*(n_t_steps / 2 + 1)));
HANDLE_ERROR(cudaMalloc((void**)&d_s11, sizeof(cuFloatComplex)*(n_t_steps / 2 + 1)));
HANDLE_ERROR(cudaMalloc((void**)&d_zin, sizeof(cuFloatComplex)*(n_t_steps / 2 + 1)));


cuFloatComplex *h_s11;
cuFloatComplex *h_zin;


h_s11 = (cuFloatComplex*)malloc(sizeof(cuFloatComplex)*(n_t_steps / 2 + 1));
h_zin = (cuFloatComplex*)malloc(sizeof(cuFloatComplex)*(n_t_steps / 2 + 1));

//CuDA fft plan
cufftHandle plan;
if (cufftPlan1d(&plan, n_t_steps, CUFFT_R2C, 1) != CUFFT_SUCCESS)
{
fprintf(stderr, "CUFFT error: Plan creation failed 1");
return;
}
cudaDeviceSynchronize();
//Execution of V fft
if (cufftExecR2C(plan, d_voltage, d_V) != CUFFT_SUCCESS)
{
fprintf(stderr, "CUFFT error: ExecC2C Forward failed 1");
return;
}
cudaDeviceSynchronize();

//Execution of I fft
if (cufftExecR2C(plan, d_current, d_I) != CUFFT_SUCCESS)
{
fprintf(stderr, "CUFFT error: ExecC2C Forward failed 2");
return;
}

cufftDestroy(plan);

int grid;
int threads = 1024;

grid = ((n_t_steps / 2 + 1) / threads) + ((((n_t_steps / 2 + 1) % 1024) == 0) ? 0 : 1);

c_zinS11 << <grid, threads >> > (d_V, d_I, (n_t_steps), d_a, d_b, d_s11, d_zin, z0, dt);


HANDLE_ERROR(cudaMemcpy(h_s11, d_s11, sizeof(cuFloatComplex)*(n_t_steps / 2 + 1), cudaMemcpyDeviceToHost));
HANDLE_ERROR(cudaMemcpy(h_zin, d_zin, sizeof(cuFloatComplex)*(n_t_steps / 2 + 1), cudaMemcpyDeviceToHost));

saveComplex(h_s11, n_t_steps, dt, 5);
saveImpImag(h_zin, n_t_steps, dt, 6);
saveImpReal(h_zin, n_t_steps, dt, 7);

cudaFree(d_V);
cudaFree(d_I);
cudaFree(d_a);
cudaFree(d_b);
cudaFree(d_s11);
cudaFree(d_zin);

}
*/


void calcVoltCurrent(int NUMDEV, float dx, float dy, float dz, int n_t_steps, int gpu_offset, float source_bt, int grid, int block,
        int volt_NX, int volt_NY, int volt_NZ, int volt_NZ_N, int sampled_voltage_is, int sampled_voltage_js, int sampled_voltage_ks, int sampled_voltage_ie, int sampled_voltage_je, int sampled_voltage_ke,
        float **d_volt_tran, float **E, int voltage_direction,
        int current_NX, int current_NY, int current_NZ, int current_NZ_N,  int sampled_current_is, int sampled_current_js, int sampled_current_ks, int sampled_current_ie, int sampled_current_je, int sampled_current_ke, float **d_current_tran, float **Hx, float **Hy, float **Hz, int current_direction,
        float *d_volt0, float *d_voltage, float * d_curr0, float *d_current){

    for (int i = 0; i < NUMDEV; i++){

        // set current device
		cudaSetDevice(i);

		//Calculate the gpu offset
		gpu_offset = i * volt_NZ_N;

        voltMgpu<<<grid, block>>>(volt_NX, volt_NY, volt_NZ, volt_NZ_N, gpu_offset, sampled_voltage_is, sampled_voltage_js, sampled_voltage_ks, sampled_voltage_ie, sampled_voltage_je, sampled_voltage_ke, d_volt_tran[i], E[i], dx, dy, dz, n_t_steps, voltage_direction);

        gpu_offset = i * current_NZ_N;
        currMgpu<<<grid, block>>>(current_NX, current_NY, current_NZ, current_NZ_N,  gpu_offset, sampled_current_is, sampled_current_js, sampled_current_ks, sampled_current_ie, sampled_current_je, sampled_current_ke, d_current_tran[i], Hx[i], Hy[i], Hz[i], dx, dy, dz, n_t_steps, current_direction);
    }

    //synchronize devices
    for (int i = 0; i<NUMDEV; i++) {
			cudaSetDevice(i);
			cudaDeviceSynchronize();
    }

    //copy voltage to GPU0
    //copy current to GPU0
    for(int i=0; i < NUMDEV; i++){

        HANDLE_ERROR(cudaMemcpy(d_volt0 + i*n_t_steps, d_volt_tran[i], source_bt, cudaMemcpyDeviceToDevice));

        HANDLE_ERROR(cudaMemcpy(d_curr0 + i*n_t_steps, d_current_tran[i], source_bt, cudaMemcpyDeviceToDevice));

    }

    //synchronize devices
    for (int i = 0; i<NUMDEV; i++) {

			cudaSetDevice(i);
			cudaDeviceSynchronize();
    }

    //Adjust voltage, current
    cudaSetDevice(0);
    sampledAdj<<<grid,block>>>(NUMDEV, d_voltage, d_volt0, n_t_steps);
    sampledAdj<<<grid,block>>>(NUMDEV, d_current, d_curr0, n_t_steps);
    correcI<<<grid,block>>>(d_current, n_t_steps);

    //synchronize devices
    for (int i = 0; i<NUMDEV; i++) {

			cudaSetDevice(i);
			cudaDeviceSynchronize();
    }


}

void solver(int NUMDEV, int m, int k_real, int NX, int NXX, int NY, int NYY, int NZ, int NZ_N, int gpu_offset,
    dim3 grid3d, dim3 block3d, dim3 grid2d, dim3 block2d,
	int pml_x_n, int pml_x_p, int pml_y_n, int pml_y_p, int pml_z_n, int pml_z_p,
	float ** d_Ex, float ** d_Jx, float ** d_Cexe, float ** d_Cexhz, float ** d_Cexhy, float ** d_Cexj,
	float ** d_Ey, float ** d_Jy, float ** d_Ceye, float ** d_Ceyhx, float ** d_Ceyhz, float ** d_Ceyj,
	float ** d_Ez, float ** d_Jz, float ** d_Ceze, float ** d_Cezhy, float ** d_Cezhx, float ** d_Cezj,
	float ** d_Hx, float ** d_Mx, float ** d_Chxh, float ** d_Chxey, float ** d_Chxez, float ** d_Chxm,
	float ** d_Hy, float ** d_My, float ** d_Chyh, float ** d_Chyez, float ** d_Chyex, float ** d_Chym,
	float ** d_Hz, float ** d_Mz, float ** d_Chzh, float ** d_Chzex, float ** d_Chzey, float ** d_Chzm,
    float ** d_gEx, float ** d_gEy, float ** d_gHx, float ** d_gHy,
    float ** d_cpml_b_mx, float ** d_cpml_a_mx,
	float ** d_cpml_b_my, float ** d_cpml_a_my,
	float ** d_cpml_b_mz, float ** d_cpml_a_mz,
    float ** d_cpml_b_ex, float ** d_cpml_a_ex,
	float ** d_cpml_b_ey, float ** d_cpml_a_ey,
	float ** d_cpml_b_ez, float ** d_cpml_a_ez,
    float ** d_Psi_eyx, float ** d_Psi_ezx, float ** d_Psi_hyx, float ** d_Psi_hzx,
	float ** d_cpsi_eyx, float ** d_cpsi_ezx, float ** d_cpsi_hyx, float ** d_cpsi_hzx,
	float ** d_Psi_exy, float ** d_Psi_ezy, float ** d_Psi_hxy, float ** d_Psi_hzy,
	float ** d_cpsi_exy, float ** d_cpsi_ezy, float ** d_cpsi_hxy, float ** d_cpsi_hzy,
	float ** d_Psi_exz, float ** d_Psi_eyz, float ** d_Psi_hxz, float ** d_Psi_hyz,
	float ** d_cpsi_exz, float ** d_cpsi_eyz, float ** d_cpsi_hxz, float ** d_cpsi_hyz,
	float ** d_signal_per_node, int source_is, int source_js,
	int source_ks, int source_ie, int source_je, int source_ke,
	int sampled_voltage_is, int sampled_voltage_js, int sampled_voltage_ks, int sampled_voltage_ie, int sampled_voltage_je, int sampled_voltage_ke,
	int volt_NX, int volt_NY, int volt_NZ_N,
	float ** E, int volt_offset,
    int sampled_current_is, int sampled_current_js, int sampled_current_ks, int sampled_current_ie, int sampled_current_je, int sampled_current_ke,
	int current_NX, int current_NY, int current_NZ_N,
	float ** Hx, float ** Hy, float ** Hz,
    cudaStream_t * stream_copy, cudaStream_t * stream_compute, cudaEvent_t *event_i, cudaEvent_t *event_j,
    float size_bt, float ** d_sigma_e_x, float ** d_sigma_e_y, float ** d_sigma_e_z, float ** d_current_tran){

		//calculations Electric fields
		for (int i = 0; i < NUMDEV; i++)
		{

			// set current device
			cudaSetDevice(i);

			//Calculate the gpu offset
			gpu_offset = i * NZ_N;
            volt_offset = i * volt_NZ_N;

			//cudaEventRecord(event_j[i], stream_compute[i]);

            calc_e << <grid3d, block3d >> > (NX, NXX, NY, NYY, NZ, NZ_N, gpu_offset, volt_offset,
				pml_x_n, pml_x_p, pml_y_n, pml_y_p, pml_z_n, pml_z_p,
				d_Ex[i], d_Jx[i], d_Cexe[i], d_Cexhz[i], d_Cexhy[i], d_Cexj[i],
				d_Ey[i], d_Jy[i], d_Ceye[i], d_Ceyhx[i], d_Ceyhz[i], d_Ceyj[i],
				d_Ez[i], d_Jz[i], d_Ceze[i], d_Cezhy[i], d_Cezhx[i], d_Cezj[i],
				d_Hx[i], d_Mx[i],
				d_Hy[i], d_My[i],
				d_Hz[i], d_Mz[i],
                d_gHx[i], d_gHy[i],
				d_cpml_b_ex[i], d_cpml_a_ex[i],
				d_cpml_b_ey[i], d_cpml_a_ey[i],
				d_cpml_b_ez[i], d_cpml_a_ez[i],
				d_Psi_eyx[i], d_Psi_ezx[i], d_Psi_hyx[i], d_Psi_hzx[i],
				d_cpsi_eyx[i], d_cpsi_ezx[i], d_cpsi_hyx[i], d_cpsi_hzx[i],
				d_Psi_exy[i], d_Psi_ezy[i], d_Psi_hxy[i], d_Psi_hzy[i],
				d_cpsi_exy[i], d_cpsi_ezy[i], d_cpsi_hxy[i], d_cpsi_hzy[i],
				d_Psi_exz[i], d_Psi_eyz[i], d_Psi_hxz[i], d_Psi_hyz[i],
				d_cpsi_exz[i], d_cpsi_eyz[i], d_cpsi_hxz[i], d_cpsi_hyz[i],
				d_signal_per_node[i], source_is, source_js,
				source_ks, source_ie, source_je, source_ke,
				sampled_voltage_is, sampled_voltage_js, sampled_voltage_ks, sampled_voltage_ie, sampled_voltage_je, sampled_voltage_ke,
				volt_NX, volt_NY, volt_NZ_N,
				m, E[i]);

		}

        //synchronize streams

        //synchronize devices
		for (int i = 0; i<NUMDEV; i++) {
			cudaSetDevice(i);
			cudaDeviceSynchronize();
		}

		//Copying Electric Send Left
		for (int i = 1; i < (NUMDEV); i++) {

			//EX
			//PB[i-1] <-- HA[i]
			//HANDLE_ERROR(cudaMemcpyAsync(d_Ex[i - 1] + NX*NY*(NZ_N - 1), d_Ex[i] + NX*NY, NX*NY * sizeof(float), cudaMemcpyDeviceToDevice, stream_Hx[i]));
			//HANDLE_ERROR(cudaMemcpyPeerAsync(d_Ex[i - 1] + NX*NY*(NZ_N - 1), i - 1, d_Ex[i] + NX*NY, i, NX*NY * sizeof(float), stream_compute[i]));
			HANDLE_ERROR(cudaMemcpy(d_gEx[i-1], d_Ex[i], NX*NY * sizeof(float), cudaMemcpyDeviceToDevice));

			//EY
			//PB[i-1] <-- HA[i]
			//HANDLE_ERROR(cudaMemcpyAsync(d_Ey[i - 1] + NX*NY*(NZ_N - 1), d_Ey[i] + NX*NY, NX*NY * sizeof(float), cudaMemcpyDeviceToDevice, stream_Hx[i]));
			//HANDLE_ERROR(cudaMemcpyPeerAsync(d_Ey[i - 1] + NX*NY*(NZ_N - 1), i - 1, d_Ey[i] + NX*NY, i, NX*NY * sizeof(float), stream_compute[i]));
			HANDLE_ERROR(cudaMemcpy(d_gEy[i-1], d_Ey[i], NX*NY * sizeof(float), cudaMemcpyDeviceToDevice));

			//EZ
			//PB[i-1] <-- HA[i]
			//HANDLE_ERROR(cudaMemcpyAsync(d_Ez[i - 1] + NX*NY*(NZ_N - 1), d_Ez[i] + NX*NY, NX*NY * sizeof(float), cudaMemcpyDeviceToDevice, stream_Hx[i]));
			//HANDLE_ERROR(cudaMemcpyPeerAsync(d_Ez[i - 1] + NX*NY*(NZ_N - 1), i - 1, d_Ez[i] + NX*NY, i, NX*NY * sizeof(float), stream_compute[i]));
			//HANDLE_ERROR(cudaMemcpy(d_gEz[i-1], d_Ez[i], NX*NY * sizeof(float), cudaMemcpyDeviceToDevice));

		}

        //synchronize streams

        //synchronize devices
		for (int i = 0; i<NUMDEV; i++) {
			cudaSetDevice(i);
			cudaDeviceSynchronize();
		}

		//Calculations Magnetic Field
		for (int i = 0; i < NUMDEV; i++)	{
			// set current device
			cudaSetDevice(i);

			//Calculate the gpu offset
			gpu_offset = i * NZ_N;
            volt_offset = i * volt_NZ_N;

			//cudaEventRecord(event_i[i], stream_compute[i]);

			calc_h << < grid3d, block3d >> > (NX, NXX, NY, NYY, NZ, NZ_N, gpu_offset,
				pml_x_n, pml_x_p, pml_y_n, pml_y_p, pml_z_n, pml_z_p,
				d_Ex[i], d_Jx[i],
				d_Ey[i], d_Jy[i],
				d_Ez[i], d_Jz[i],
                d_gEx[i], d_gEy[i],
				d_Hx[i], d_Mx[i], d_Chxh[i], d_Chxey[i], d_Chxez[i], d_Chxm[i],
				d_Hy[i], d_My[i], d_Chyh[i], d_Chyez[i], d_Chyex[i], d_Chym[i],
				d_Hz[i], d_Mz[i], d_Chzh[i], d_Chzex[i], d_Chzey[i], d_Chzm[i],
				d_cpml_b_mx[i], d_cpml_a_mx[i],
				d_cpml_b_my[i], d_cpml_a_my[i],
				d_cpml_b_mz[i], d_cpml_a_mz[i],
				d_Psi_eyx[i], d_Psi_ezx[i], d_Psi_hyx[i], d_Psi_hzx[i],
				d_cpsi_eyx[i], d_cpsi_ezx[i], d_cpsi_hyx[i], d_cpsi_hzx[i],
				d_Psi_exy[i], d_Psi_ezy[i], d_Psi_hxy[i], d_Psi_hzy[i],
				d_cpsi_exy[i], d_cpsi_ezy[i], d_cpsi_hxy[i], d_cpsi_hzy[i],
				d_Psi_exz[i], d_Psi_eyz[i], d_Psi_hxz[i], d_Psi_hyz[i],
				d_cpsi_exz[i], d_cpsi_eyz[i], d_cpsi_hxz[i], d_cpsi_hyz[i],
				sampled_current_is, sampled_current_js, sampled_current_ks, sampled_current_ie, sampled_current_je, sampled_current_ke,
				current_NX, current_NY, current_NZ_N,
				m, Hx[i], Hy[i], Hz[i]);


		}

        //synchronize streams

        //synchronize devices
		for (int i = 0; i<NUMDEV; i++) {
			cudaSetDevice(i);
			cudaDeviceSynchronize();
		}

		//Copying Magnetic Send Right
        for (int i = 0; i < (NUMDEV - 1); i++) {

			//cudaStreamWaitEvent(stream_copy[i], event_i[i], 0);
			//Magnetic
			//HX
			//PA[i+1] <-- HB[i]
			//HANDLE_ERROR(cudaMemcpyAsync(d_Hx[i + 1], d_Hx[i] + NX*NY*(NZ_N - 2), NX*NY * sizeof(float), cudaMemcpyDeviceToDevice, stream_halo[i]));
			//HANDLE_ERROR(cudaMemcpyPeerAsync(d_Hx[i + 1], i+1, d_Hx[i] + NX*NY*(NZ_N - 2), i, NX*NY * sizeof(float), stream_compute[i]));
			HANDLE_ERROR(cudaMemcpy(d_gHx[i+1], d_Hx[i] + NX*NY*(NZ_N - 1),NX*NY * sizeof(float), cudaMemcpyDeviceToDevice));

			//HY
			//PA[i+1] <-- HB[i]
			//HANDLE_ERROR(cudaMemcpyAsync(d_Hy[i + 1], d_Hy[i] + NX*NY*(NZ_N - 2), NX*NY * sizeof(float), cudaMemcpyDeviceToDevice, stream_halo[i]));
			//HANDLE_ERROR(cudaMemcpyPeerAsync(d_Hy[i + 1], i + 1, d_Hy[i] + NX*NY*(NZ_N - 2), i, NX*NY * sizeof(float), stream_compute[i]));
			HANDLE_ERROR(cudaMemcpy(d_gHy[i+1], d_Hy[i] + NX*NY*(NZ_N - 1), NX*NY * sizeof(float), cudaMemcpyDeviceToDevice));

			//HZ
			//PA[i+1] <-- HB[i]
			//HANDLE_ERROR(cudaMemcpyAsync(d_Hz[i + 1], d_Hz[i] + NX*NY*(NZ_N - 2), NX*NY * sizeof(float), cudaMemcpyDeviceToDevice, stream_halo[i]));
			//HANDLE_ERROR(cudaMemcpyPeerAsync(d_Hz[i + 1], i + 1, d_Hz[i] + NX*NY*(NZ_N - 2), i, NX*NY * sizeof(float), stream_compute[i]));
			//HANDLE_ERROR(cudaMemcpy(d_gHy[i+1], d_Hy[i] + NX*NY*(NZ_N - 1), NX*NY * sizeof(float), cudaMemcpyDeviceToDevice));

		}

		//synchronize streams

        //synchronize devices
		for (int i = 0; i<NUMDEV; i++) {
			cudaSetDevice(i);
			cudaDeviceSynchronize();
		}

		//testing saving E fields
		for (int i = 0; i < NUMDEV; i++) {

			cudaSetDevice(i);

			//HANDLE_ERROR(cudaMemcpy(h_test[i], d_Ez[i], size_bt, cudaMemcpyDeviceToHost));


			//gpu offset
			gpu_offset = i * NZ_N;

			//check k to see the right one to save
			for (int k = 0; k < NZ_N; k++) {

				//adjust k to the current gpu
				k_real = k + gpu_offset;

				//compare the kreal with the k position to be saved
				if ((k_real == (source_ke)) && ((m % 10) == 0)  ) {
					//cout << "saving" << endl;
					//saving in k position
					//saveFilePm3d(NX, NXX, NY, NYY, NZ_N, d_Hz[i], m/10, k);

                    if(m == 1999){
                        //HANDLE_ERROR(cudaMemcpy(h_test[i], d_sigma_e_x[i], size_bt, cudaMemcpyDeviceToHost));
                        //saveFilePm3d(NX, NXX, NY, NYY, NZ_N, h_test[i], 3000, k);

                        //HANDLE_ERROR(cudaMemcpy(h_test[i], d_sigma_e_y[i], size_bt, cudaMemcpyDeviceToHost));
                        //saveFilePm3d(NX, NXX, NY, NYY, NZ_N, h_test[i], 3001, k);

                        //HANDLE_ERROR(cudaMemcpy(h_test[i], d_sigma_e_z[i], size_bt, cudaMemcpyDeviceToHost));
                        //saveFilePm3d(NX, NXX, NY, NYY, NZ_N, h_test[i], 3002, k);
                    }
				}

			}
		}

        //synchronize streams

        //synchronize devices
        for (int i = 0; i<NUMDEV; i++) {
			cudaSetDevice(i);
			cudaDeviceSynchronize();
        }

}


void marchingLoop(const int NUMDEV, float eps_0,  float pi ,  float mu_0,  float c,  float dx, float dy, float dz, float dt,
    int X, int Y, int Z,
    int n_t_steps,
    int pml_x_n, int pml_y_n, int pml_z_n, int pml_x_p, int pml_y_p, int pml_z_p,
    int air_buff_x_n, int air_buff_y_n,int air_buff_z_n,int air_buff_x_p,int air_buff_y_p,int air_buff_z_p,
    int source_tp, float source_freq, float source_amp,
    float source_min_x ,float source_min_y , float source_min_z,
    float source_max_x, float source_max_y, float source_max_z,
    int source_direction, float rs, float nc, float tau, float t_0,
    float *brick_min_x,float *brick_min_y,float *brick_min_z,
    float *brick_max_x,float *brick_max_y,float *brick_max_z,
    float *brick_sigma_e_x, float *brick_sigma_e_y, float *brick_sigma_e_z,
    float *brick_eps_r_x, float *brick_eps_r_y, float *brick_eps_r_z,
    float *brick_sigma_m_x, float *brick_sigma_m_y, float *brick_sigma_m_z,
    float *brick_mu_r_x, float *brick_mu_r_y, float *brick_mu_r_z,
    int brick_opt, int brick_num,
    float *pec_min_x, float *pec_min_y, float *pec_min_z,
    float *pec_max_x, float *pec_max_y, float *pec_max_z,
    float *pec_sigma_e_x, float *pec_sigma_e_y, float *pec_sigma_e_z,
    int pec_opt, int pec_num,
    float resistor_min_x, float resistor_min_y, float resistor_min_z,
    float resistor_max_x, float resistor_max_y, float resistor_max_z,
    float resistor_resist, int resistor_direction, int resistor_opt,
    float sampled_voltage_min_x, float sampled_voltage_min_y, float sampled_voltage_min_z,
    float sampled_voltage_max_x, float sampled_voltage_max_y, float sampled_voltage_max_z,
    int voltage_direction,
    float sampled_current_min_x, float sampled_current_min_y, float sampled_current_min_z,
    float sampled_current_max_x, float sampled_current_max_y, float sampled_current_max_z,
    int current_direction)
{

    //Set fastest Device Access
	for (int i = 0; i<NUMDEV; i++) {
		//Set the device
		cudaSetDevice(i);
		//Run across to check P2P
		for (int j = 0; j<NUMDEV; j++) {

			if (i != j) {

				int access;
				//check P2P
				 HANDLE_ERROR(cudaDeviceCanAccessPeer(&access, i, j));

				if (access) {
					//Enable if possible
					HANDLE_ERROR(cudaDeviceEnablePeerAccess(j, 0));

					//cout << "Peer Access " << i << " -> " << j << endl;

				}
				else {
					//cout << "Peer Access " << i << " -> " << j << " not avaible"<< endl;
				}
			}
		}
	}


   //Domain sizee//Domain Size coalesced
    int NX, NY, NZ, NXX, NYY;
    // X direction length
	NXX = X + air_buff_x_n + air_buff_x_p + pml_x_n + pml_x_p;
	// Y direction length
	NYY = Y + air_buff_y_n + air_buff_y_p + pml_y_n + pml_y_p;
	// Z direction length
	NZ = Z + air_buff_z_n + air_buff_z_p + pml_z_n + pml_z_p;
	//Coalescing data access
	NX = ((NXX % 16) == 0) ? NXX : (floor(NXX / 16) + 1) * 16;
	NY = ((NYY % 16) == 0) ? NYY : (floor(NYY / 16) + 1) * 16;
	NZ = ((NZ % NUMDEV) == 0) ? NZ : (floor(NZ / NUMDEV) + 1) * NUMDEV;

	//Z per GPU
    int NZ_N = (NZ / NUMDEV);

    //Global k
    int k_real = 0;
    //Offset due to multiples GPUS
    int gpu_offset = 0;
    //Offset due to multiples GPUS for the voltage calculation
    int volt_offset = 0;

    //Print Domain Characteristics
	//print the domain size on the screen
	cout << "Domain Size: ";
	cout << "NX: " << NX << " | ";
	cout << "NY: " << NY << " | ";
	cout << "NZ: " << NZ << " | ";
	cout << "Total Number of points: " << NX*NY*NZ << endl;
	cout << "Number of GPUs: " << NUMDEV << endl;
	cout << "Arrays divided in Z. Z Size for each GPU: " << NZ_N << endl;
	cout << "Time Step: " << dt << endl;
	cout << "Sampling Rate: " << 1 / dt << " Hz" << endl;
	cout << "Time Steps: " << n_t_steps << endl;

    //3d bricks
    int brick_is[brick_num], brick_js[brick_num], brick_ks[brick_num];
    int brick_ie[brick_num], brick_je[brick_num], brick_ke[brick_num];
    //brick coordinates in cells

    if(brick_opt != 0){

        for(int i=0; i<brick_num ; i++){

            brick_is[i] = air_buff_x_n + pml_x_n + calc_index(brick_min_x[i] , dx);
            brick_js[i]  = air_buff_y_n + pml_y_n + calc_index(brick_min_y[i] , dy);
            brick_ks[i]  = air_buff_z_n + pml_z_n + calc_index(brick_min_z[i] , dz);
            brick_ie[i]  = air_buff_x_n + pml_x_n + calc_index(brick_max_x[i] , dx);
            brick_je[i]  = air_buff_y_n + pml_y_n + calc_index(brick_max_y[i] , dy);
            brick_ke[i]  = air_buff_z_n + pml_z_n + calc_index(brick_max_z[i] , dz);

            cout << "brick_is[" << i << "]: " << brick_is[i];
            cout << "\t brick_ie["<< i <<"]: " << brick_ie[i] << endl;
            cout << "brick_js[" << i << "]: " << brick_js[i];
            cout << "\t brick_je["<< i <<"]: " << brick_je[i] << endl;
            cout << "brick_ks[" << i << "]: " << brick_ks[i];
            cout << "\t brick_ke["<< i <<"]: " << brick_ke[i] << endl;
        }

    }

    //PEC plates
    int pec_is[pec_num], pec_js[pec_num], pec_ks[pec_num];
    int pec_ie[pec_num], pec_je[pec_num], pec_ke[pec_num];
    //PEC coordinates in cells

    if(pec_opt != 0){

        for(int i=0; i<pec_num ; i++){

            pec_is[i] = air_buff_x_n + pml_x_n + calc_index(pec_min_x[i], dx);
            pec_js[i] = air_buff_y_n + pml_y_n + calc_index(pec_min_y[i], dy);
            pec_ks[i] = air_buff_z_n + pml_z_n + calc_index(pec_min_z[i], dz);
            pec_ie[i] = air_buff_x_n + pml_x_n + calc_index(pec_max_x[i], dx);
            pec_je[i] = air_buff_y_n + pml_y_n + calc_index(pec_max_y[i], dy);
            pec_ke[i] = air_buff_z_n + pml_z_n + calc_index(pec_max_z[i], dz);

            cout << "pec_is[" << i << "]: " << pec_is[i];
            cout << "\t pec_ie["<< i <<"]: " << pec_ie[i] << endl;
            cout << "pec_js[" << i << "]: " << pec_js[i];
            cout << "\t pec_je["<< i <<"]: " << pec_je[i] << endl;
            cout << "pec_ks[" << i << "]: " << pec_ks[i];
            cout << "\t pec_ke["<< i <<"]: " << pec_ke[i] << endl;

        }
    }

    //resistorindex in simulation domain
    int resistor_is, resistor_js, resistor_ks;
    int resistor_ie, resistor_je, resistor_ke;
    //resistor coordinates in cells

    if(resistor_opt != 0){

        resistor_is = air_buff_x_n + pml_x_n + calc_index(resistor_min_x, dx);
        resistor_js = air_buff_y_n + pml_y_n + calc_index(resistor_min_y, dy);
        resistor_ks = air_buff_z_n + pml_z_n + calc_index(resistor_min_z, dz);
        resistor_ie = air_buff_x_n + pml_x_n + calc_index(resistor_max_x, dx);
        resistor_je = air_buff_y_n + pml_y_n + calc_index(resistor_max_y, dy);
        resistor_ke = air_buff_z_n + pml_z_n + calc_index(resistor_max_z, dz);

        cout << "resistor_ks: " << resistor_ks;
        cout << "\t resistor_ke: " << resistor_ke << endl;

    }

     //source index in simulation domain
    int source_is, source_js, source_ks;
    int source_ie, source_je, source_ke;

    //Source coordinates in cells
	source_is = air_buff_x_n + pml_x_n + calc_index(source_min_x, dx);
	source_js = air_buff_y_n + pml_y_n + calc_index(source_min_y, dy);
	source_ks = air_buff_z_n + pml_z_n + calc_index(source_min_z, dz);
	source_ie = air_buff_x_n + pml_x_n + calc_index(source_max_x, dx);
	source_je = air_buff_y_n + pml_y_n + calc_index(source_max_y, dy);
	source_ke = air_buff_z_n + pml_z_n + calc_index(source_max_z, dz);

    //source is going to be as the same size as the n_steps
    int source_size = (n_t_steps);

    //relation between the voltage input array and is value per node
	float v_mag_factor;
	//For x directed source  ;
	//v_mag_factor = source_ie - source_is;
	//For y directed source
	//v_mag_factor = source_je - source_js;
	//For z directed source
	v_mag_factor = source_ke - source_ks;

    cout << "Source Position:" << "(is,js,ks) = (" << source_is << ", " << source_js << ", " << source_ks << ") | (ie,je,ke) = (" << source_ie << ", " << source_je << ", " << source_ke << ")" << endl;

    //sampled voltage index in simulation domain
	int sampled_voltage_is, sampled_voltage_ie, sampled_voltage_js, sampled_voltage_je, sampled_voltage_ks, sampled_voltage_ke;
    //sampled voltage index
	sampled_voltage_is = air_buff_x_n + pml_x_n + calc_index(sampled_voltage_min_x, dx);
	sampled_voltage_js = air_buff_y_n + pml_y_n + calc_index(sampled_voltage_min_y, dy);
	sampled_voltage_ks = air_buff_z_n + pml_z_n + calc_index(sampled_voltage_min_z, dz);
	sampled_voltage_ie = air_buff_x_n + pml_x_n + calc_index(sampled_voltage_max_x, dx);
	sampled_voltage_je = air_buff_y_n + pml_y_n + calc_index(sampled_voltage_max_y, dy);
	sampled_voltage_ke = air_buff_z_n + pml_z_n + calc_index(sampled_voltage_max_z, dz);

    //sampled voltage size in the axis
	int volt_NX = sampled_voltage_ie - sampled_voltage_is +1;
	int volt_NY = sampled_voltage_je - sampled_voltage_js +1;
    //size in Z for each gpu
    int volt_NZ = NZ;
    int volt_NZ_N = NZ_N;
    cout << "Sampled Voltage Position:" << "(is,js,ks) = (" << sampled_voltage_is << ", " << sampled_voltage_js << ", " << sampled_voltage_ks << ") | (ie,je,ke) = (" << sampled_voltage_ie << ", " << sampled_voltage_je << ", " << sampled_voltage_ke << ")" << endl;
    //cout << "Sampled Voltage vector size: X: " << volt_NX << " Y: " << volt_NY << " e Z: " << volt_NZ << "(Z_N: " << volt_NZ_N << " )" << endl;

     //sampled current index in simulation domain
	int sampled_current_is, sampled_current_ie, sampled_current_js, sampled_current_je, sampled_current_ks, sampled_current_ke;

    //sampled current index
	sampled_current_is = air_buff_x_n + pml_x_n + calc_index(sampled_current_min_x, dx);
	sampled_current_js = air_buff_y_n + pml_y_n + calc_index(sampled_current_min_y, dy);
	sampled_current_ks = air_buff_z_n + pml_z_n + calc_index(sampled_current_min_z, dz);
	sampled_current_ie = air_buff_x_n + pml_x_n + calc_index(sampled_current_max_x, dx);
	sampled_current_je = air_buff_y_n + pml_y_n + calc_index(sampled_current_max_y, dy);
	sampled_current_ke = air_buff_z_n + pml_z_n + calc_index(sampled_current_max_z, dz);

	//sampled current size in axis
	int current_NX = sampled_current_ie - sampled_current_is + 2;
	int current_NY = sampled_current_je - sampled_current_js + 2;
	//size in Z for each gpu
    int current_NZ = NZ;
	int current_NZ_N = NZ_N;
    cout << "Sampled Current Position:" << "(is,js,ks) = (" << sampled_current_is << ", " << sampled_current_js << ", " << sampled_current_ks << ") | (ie,je,ke) = (" << sampled_current_ie << ", " << sampled_current_je << ", " << sampled_current_ke << ")" << endl;
    //cout << "Sampled Current vector size: X: " << current_NX << " Y: " << current_NY << " e Z: " <<current_NZ << "(Z_N: " << current_NZ_N << " )" << endl;

    //Input signal array and its equivalent per node
	float * d_signal[NUMDEV];
	float * d_signal_per_node[NUMDEV];

 //Define Output variables
	//Sampled Voltage device array

	float * d_voltage;
    float * h_voltage;
    float * d_volt_tran[NUMDEV];
    float * d_volt0;

    //Electric field auxiliars to calculate voltage device array
	float * E[NUMDEV];

	//Sampled Current device array
	float * d_current;
	float * h_current;
	float * d_current_tran[NUMDEV];
	float * d_curr0;

//Magnetic field auxiliars to calculate current device array
	float * Hx[NUMDEV];
	float * Hy[NUMDEV];
	float * Hz[NUMDEV];

	cout << "Create test and exhibiton variables" << endl;
	//Create Test host variables to check the program
	float * h_test[NUMDEV];
	//Create Test host variables to check the program
	float * h_Hx[NUMDEV];
	float * h_Hy[NUMDEV];
	float * h_Hz[NUMDEV];
	float * h_E[NUMDEV];

//Create cudastreams
	cudaStream_t stream_copy[NUMDEV];
	cudaStream_t stream_compute[NUMDEV];
    //Events to synchronize streams
	cudaEvent_t event_i[NUMDEV];
    cudaEvent_t event_j[NUMDEV];


    //Vectors size in bytes
	float size_bt = (NX * NY) * (NZ_N) * sizeof(float);
    float ghost_size_bt = (NX * NY) * sizeof(float);

    //source size in bytes
	float source_bt = source_size * sizeof(float);

    //size in bytes for E sampled
    float size_volt_bt = volt_NX*volt_NY*volt_NZ_N*n_t_steps*sizeof(float);

    //size in bytes for H sampled per device
	float size_current_bt = current_NX*current_NY*current_NZ_N*n_t_steps*sizeof(float);

    //grid and block definition
	// block of threads 1024
	dim3 block3d(WIDTH*WIDTH, WIDTH*WIDTH, WIDTH);
	// each dimension of the grid in blocks
	int grid_X = (NX / block3d.x) + (((NX % block3d.x) == 0) ? 0 : 1);
	int grid_Y = (NY / block3d.y) + (((NY % block3d.y) == 0) ? 0 : 1);
	int grid_Z = (NZ_N / block3d.z) + (((NZ_N % block3d.z) == 0) ? 0 : 1);
	// difinition of the 3D grid
	dim3 grid3d(grid_X, grid_Y, grid_Z);
	// definition of the 2d block
	dim3 block2d(WIDTH*WIDTH, WIDTH*WIDTH, 1);
    // difinition of the 2D grid
	dim3 grid2d(grid_X, grid_Y, 1);

    //1D block size for the source, voltage and current arrays
	int block = 512;
	//1D grid Size for the source, voltage and current arrays
	int grid = (source_size / block) + (((source_size % block) == 0) ? 0 : 1);

    cout << "Simulation Domain Variables" << endl;
    //Ghost Nodes
    float * d_gEx[NUMDEV];
    float * d_gEy[NUMDEV];
    float * d_gHx[NUMDEV];
    float * d_gHy[NUMDEV];

	// Device variables
	//Ex and its coefficients
	float * d_Ex[NUMDEV];
	float * d_Jx[NUMDEV];
	float * d_Cexe[NUMDEV];
	float * d_Cexhz[NUMDEV];
	float * d_Cexhy[NUMDEV];
	float * d_Cexj[NUMDEV];
	float * d_eps_r_x[NUMDEV];
	float * d_sigma_e_x[NUMDEV];
	//Ey and its coefficients
	float * d_Ey[NUMDEV];
	float * d_Jy[NUMDEV];
	float * d_Ceye[NUMDEV];
	float * d_Ceyhx[NUMDEV];
	float * d_Ceyhz[NUMDEV];
	float * d_Ceyj[NUMDEV];
	float * d_eps_r_y[NUMDEV];
	float * d_sigma_e_y[NUMDEV];
	//Ez and its coefficients
	float * d_Ez[NUMDEV];
	float * d_Jz[NUMDEV];
	float * d_Ceze[NUMDEV];
	float * d_Cezhy[NUMDEV];
	float * d_Cezhx[NUMDEV];
	float * d_Cezj[NUMDEV];
	float * d_eps_r_z[NUMDEV];
	float * d_sigma_e_z[NUMDEV];
	//Hx and its coefficients
	float * d_Hx[NUMDEV];
	float * d_Mx[NUMDEV];
	float * d_Chxh[NUMDEV];
	float * d_Chxey[NUMDEV];
	float * d_Chxez[NUMDEV];
	float * d_Chxm[NUMDEV];
	float * d_mu_r_x[NUMDEV];
	float * d_sigma_m_x[NUMDEV];
	//Hy and its coefficients
	float * d_Hy[NUMDEV];
	float * d_My[NUMDEV];
	float * d_Chyh[NUMDEV];
	float * d_Chyez[NUMDEV];
	float * d_Chyex[NUMDEV];
	float * d_Chym[NUMDEV];
	float * d_mu_r_y[NUMDEV];
	float * d_sigma_m_y[NUMDEV];
	//Hz and its coefficients
	float * d_Hz[NUMDEV];
	float * d_Mz[NUMDEV];
	float * d_Chzh[NUMDEV];
	float * d_Chzex[NUMDEV];
	float * d_Chzey[NUMDEV];
	float * d_Chzm[NUMDEV];
	float * d_mu_r_z[NUMDEV];
	float * d_sigma_m_z[NUMDEV];

	// Material Grid coefficients for the nodes
	//electric permittivity
	float *d_material_3d_space_eps_x[NUMDEV];
	float *d_material_3d_space_eps_y[NUMDEV];
	float *d_material_3d_space_eps_z[NUMDEV];
	//electric conductivity
	float *d_material_3d_space_sigma_e_x[NUMDEV];
	float *d_material_3d_space_sigma_e_y[NUMDEV];
	float *d_material_3d_space_sigma_e_z[NUMDEV];
	//magnetic permeability
	float *d_material_3d_space_mu_x[NUMDEV];
	float *d_material_3d_space_mu_y[NUMDEV];
	float *d_material_3d_space_mu_z[NUMDEV];
	//magnetic conductivity
	float *d_material_3d_space_sigma_m_x[NUMDEV];
	float *d_material_3d_space_sigma_m_y[NUMDEV];
	float *d_material_3d_space_sigma_m_z[NUMDEV];

	//device variables for the cpml
	//bex,aex
	float * d_cpml_b_ex[NUMDEV];
	float * d_cpml_a_ex[NUMDEV];
	//bmx,amx
	float * d_cpml_b_mx[NUMDEV];
	float * d_cpml_a_mx[NUMDEV];
	//bey,aey
	float * d_cpml_b_ey[NUMDEV];
	float * d_cpml_a_ey[NUMDEV];
	//bmy,amy
	float * d_cpml_b_my[NUMDEV];
	float * d_cpml_a_my[NUMDEV];
	//bez,aez
	float * d_cpml_b_ez[NUMDEV];
	float * d_cpml_a_ez[NUMDEV];
	//bmz,amz
	float * d_cpml_b_mz[NUMDEV];
	float * d_cpml_a_mz[NUMDEV];

	//device variables for the cpml
	//for x_n and x_p
	float * d_Psi_eyx[NUMDEV];
	float * d_Psi_ezx[NUMDEV];

	float * d_Psi_hyx[NUMDEV];
	float * d_Psi_hzx[NUMDEV];

	float * d_cpsi_eyx[NUMDEV];
	float * d_cpsi_ezx[NUMDEV];
	float * d_cpsi_hyx[NUMDEV];
	float * d_cpsi_hzx[NUMDEV];

	//device variables for the cpml
	//for y_n and y_p
	float * d_Psi_exy[NUMDEV];
	float * d_Psi_ezy[NUMDEV];

	float * d_Psi_hxy[NUMDEV];
	float * d_Psi_hzy[NUMDEV];

	float * d_cpsi_exy[NUMDEV];
	float * d_cpsi_ezy[NUMDEV];
	float * d_cpsi_hxy[NUMDEV];
	float * d_cpsi_hzy[NUMDEV];

	//device variables for the cpml
	//for z_n and z_p
	float * d_Psi_exz[NUMDEV];
	float * d_Psi_eyz[NUMDEV];

	float * d_Psi_hxz[NUMDEV];
	float * d_Psi_hyz[NUMDEV];

	float * d_cpsi_exz[NUMDEV];
	float * d_cpsi_eyz[NUMDEV];
	float * d_cpsi_hxz[NUMDEV];
	float * d_cpsi_hyz[NUMDEV];

	cout << "Allocate Memory on Device and set air to the whole space" << endl;

//allocate memory on device
	for (int i = 0; i < NUMDEV; i++) {

		// set current device
		cudaSetDevice(i);

		// allocate device memory
		HANDLE_ERROR(cudaMalloc((void**)&d_gEx[i], ghost_size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_gEy[i], ghost_size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_gHx[i], ghost_size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_gHy[i], ghost_size_bt));

		// allocate device memory
		HANDLE_ERROR(cudaMalloc((void**)&d_Ex[i], size_bt));
		//HANDLE_ERROR(cudaMallocManaged(&d_Ex, size_bt));

		HANDLE_ERROR(cudaMalloc((void**)&d_Jx[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_Cexe[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_Cexhz[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_Cexhy[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_Cexj[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_eps_r_x[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_sigma_e_x[i], size_bt));

		HANDLE_ERROR(cudaMalloc((void**)&d_Ey[i], size_bt));
		//HANDLE_ERROR(cudaMallocManaged(&d_Ey[i], size_bt));

		HANDLE_ERROR(cudaMalloc((void**)&d_Jy[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_Ceye[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_Ceyhx[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_Ceyhz[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_Ceyj[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_eps_r_y[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_sigma_e_y[i], size_bt));

		HANDLE_ERROR(cudaMalloc((void**)&d_Ez[i], size_bt));
		//HANDLE_ERROR(cudaMallocManaged(&d_Ez[i], size_bt));

		HANDLE_ERROR(cudaMalloc((void**)&d_Jz[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_Ceze[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_Cezhy[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_Cezhx[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_Cezj[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_eps_r_z[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_sigma_e_z[i], size_bt));

		HANDLE_ERROR(cudaMalloc((void**)&d_Hx[i], size_bt));
		//HANDLE_ERROR(cudaMallocManaged(&d_Hx[i], size_bt));

		HANDLE_ERROR(cudaMalloc((void**)&d_Mx[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_Chxh[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_Chxey[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_Chxez[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_Chxm[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_mu_r_x[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_sigma_m_x[i], size_bt));

		HANDLE_ERROR(cudaMalloc((void**)&d_Hy[i], size_bt));
		//HANDLE_ERROR(cudaMallocManaged(&d_Hy[i], size_bt));

		HANDLE_ERROR(cudaMalloc((void**)&d_My[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_Chyh[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_Chyez[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_Chyex[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_Chym[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_mu_r_y[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_sigma_m_y[i], size_bt));

		HANDLE_ERROR(cudaMalloc((void**)&d_Hz[i], size_bt));
		//HANDLE_ERROR(cudaMallocManaged(&d_Hz[i], size_bt));

		HANDLE_ERROR(cudaMalloc((void**)&d_Mz[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_Chzh[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_Chzex[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_Chzey[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_Chzm[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_mu_r_z[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_sigma_m_z[i], size_bt));

		// Material Grid coefficients for the nodes
		//electric permittivity
		HANDLE_ERROR(cudaMalloc((void**)&d_material_3d_space_eps_x[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_material_3d_space_eps_y[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_material_3d_space_eps_z[i], size_bt));
		//electric conductivity
		HANDLE_ERROR(cudaMalloc((void**)&d_material_3d_space_sigma_e_x[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_material_3d_space_sigma_e_y[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_material_3d_space_sigma_e_z[i], size_bt));
		//magnetic permeability
		HANDLE_ERROR(cudaMalloc((void**)&d_material_3d_space_mu_x[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_material_3d_space_mu_y[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_material_3d_space_mu_z[i], size_bt));
		//magnetic conductivity
		HANDLE_ERROR(cudaMalloc((void**)&d_material_3d_space_sigma_m_x[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_material_3d_space_sigma_m_y[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_material_3d_space_sigma_m_z[i], size_bt));

		//device variables for the cpml
		HANDLE_ERROR(cudaMalloc((void**)&d_cpml_b_ex[i], NX * sizeof(float)));
		HANDLE_ERROR(cudaMalloc((void**)&d_cpml_a_ex[i], NX * sizeof(float)));
		HANDLE_ERROR(cudaMalloc((void**)&d_cpml_b_mx[i], NX * sizeof(float)));
		HANDLE_ERROR(cudaMalloc((void**)&d_cpml_a_mx[i], NX * sizeof(float)));

		HANDLE_ERROR(cudaMalloc((void**)&d_cpml_b_ey[i], NY * sizeof(float)));
		HANDLE_ERROR(cudaMalloc((void**)&d_cpml_a_ey[i], NY * sizeof(float)));
		HANDLE_ERROR(cudaMalloc((void**)&d_cpml_b_my[i], NY * sizeof(float)));
		HANDLE_ERROR(cudaMalloc((void**)&d_cpml_a_my[i], NY * sizeof(float)));

		HANDLE_ERROR(cudaMalloc((void**)&d_cpml_b_ez[i], NZ_N * sizeof(float)));
		HANDLE_ERROR(cudaMalloc((void**)&d_cpml_a_ez[i], NZ_N * sizeof(float)));
		HANDLE_ERROR(cudaMalloc((void**)&d_cpml_b_mz[i], NZ_N * sizeof(float)));
		HANDLE_ERROR(cudaMalloc((void**)&d_cpml_a_mz[i], NZ_N * sizeof(float)));

		//for x_n and x_p
		HANDLE_ERROR(cudaMalloc((void**)&d_Psi_eyx[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_Psi_ezx[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_Psi_hyx[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_Psi_hzx[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_cpsi_eyx[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_cpsi_ezx[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_cpsi_hyx[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_cpsi_hzx[i], size_bt));

		//for y_n and y_p
		HANDLE_ERROR(cudaMalloc((void**)&d_Psi_exy[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_Psi_ezy[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_Psi_hxy[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_Psi_hzy[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_cpsi_exy[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_cpsi_ezy[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_cpsi_hxy[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_cpsi_hzy[i], size_bt));

		//for z_n and z_p
		HANDLE_ERROR(cudaMalloc((void**)&d_Psi_exz[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_Psi_eyz[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_Psi_hxz[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_Psi_hyz[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_cpsi_exz[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_cpsi_eyz[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_cpsi_hxz[i], size_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_cpsi_hyz[i], size_bt));

		//signal and signal per node
		HANDLE_ERROR(cudaMalloc((void**)&d_signal_per_node[i], source_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_signal[i], source_bt));

		//Voltage and current
		HANDLE_ERROR(cudaMalloc((void**)&d_volt_tran[i], source_bt));
		HANDLE_ERROR(cudaMalloc((void**)&d_current_tran[i], source_bt));

		HANDLE_ERROR(cudaMalloc((void**)&Hx[i], size_current_bt));
		HANDLE_ERROR(cudaMalloc((void**)&Hy[i], size_current_bt));
		HANDLE_ERROR(cudaMalloc((void**)&Hz[i], size_current_bt));
		HANDLE_ERROR(cudaMalloc((void**)&E[i], size_volt_bt));

		HANDLE_ERROR(cudaMallocHost((void**)&h_Hx[i], size_current_bt));
		HANDLE_ERROR(cudaMallocHost((void**)&h_Hy[i], size_current_bt));
		HANDLE_ERROR(cudaMallocHost((void**)&h_Hz[i], size_current_bt));
		HANDLE_ERROR(cudaMallocHost((void**)&h_E[i], size_volt_bt));

		//test variable **always check memory size allocated
		HANDLE_ERROR(cudaMallocHost((void **)&h_test[i], size_bt));

		// create streams for timing and synchronizing
		HANDLE_ERROR(cudaStreamCreate(&stream_copy[i]));

		// create streams for timing and synchronizing
		HANDLE_ERROR(cudaStreamCreate(&stream_compute[i]));
		HANDLE_ERROR(cudaEventCreate(&event_i[i]));
        HANDLE_ERROR(cudaEventCreate(&event_j[i]));



		setZero << <grid3d, block3d >> > (
			d_Ex[i], d_Jx[i], d_Cexe[i], d_Cexhz[i], d_Cexhy[i], d_Cexj[i], d_eps_r_x[i], d_sigma_e_x[i],
			d_Ey[i], d_Jy[i], d_Ceye[i], d_Ceyhx[i], d_Ceyhz[i], d_Ceyj[i], d_eps_r_y[i], d_sigma_e_y[i],
			d_Ez[i], d_Jz[i], d_Ceze[i], d_Cezhy[i], d_Cezhx[i], d_Cezj[i], d_eps_r_z[i], d_sigma_e_z[i],
			d_Hx[i], d_Mx[i], d_Chxh[i], d_Chxey[i], d_Chxez[i], d_Chxm[i], d_mu_r_x[i], d_sigma_m_x[i],
			d_Hy[i], d_My[i], d_Chyh[i], d_Chyez[i], d_Chyex[i], d_Chym[i], d_mu_r_y[i], d_sigma_m_y[i],
			d_Hz[i], d_Mz[i], d_Chzh[i], d_Chzex[i], d_Chzey[i], d_Chzm[i], d_mu_r_z[i], d_sigma_m_z[i],
			d_material_3d_space_eps_x[i], d_material_3d_space_eps_y[i], d_material_3d_space_eps_z[i],
			d_material_3d_space_sigma_e_x[i], d_material_3d_space_sigma_e_y[i], d_material_3d_space_sigma_e_z[i],
			d_material_3d_space_mu_x[i], d_material_3d_space_mu_y[i], d_material_3d_space_mu_z[i],
			d_material_3d_space_sigma_m_x[i], d_material_3d_space_sigma_m_y[i], d_material_3d_space_sigma_m_z[i],
			d_cpml_b_ex[i], d_cpml_a_ex[i], d_cpml_b_mx[i], d_cpml_a_mx[i],
			d_cpml_b_ey[i], d_cpml_a_ey[i], d_cpml_b_my[i], d_cpml_a_my[i],
			d_cpml_b_ez[i], d_cpml_a_ez[i], d_cpml_b_mz[i], d_cpml_a_mz[i],
			d_Psi_eyx[i], d_Psi_ezx[i], d_Psi_hyx[i], d_Psi_hzx[i],
			d_cpsi_eyx[i], d_cpsi_ezx[i], d_cpsi_hyx[i], d_cpsi_hzx[i],
			d_Psi_exy[i], d_Psi_ezy[i], d_Psi_hxy[i], d_Psi_hzy[i],
			d_cpsi_exy[i], d_cpsi_ezy[i], d_cpsi_hxy[i], d_cpsi_hzy[i],
			d_Psi_exz[i], d_Psi_eyz[i], d_Psi_hxz[i], d_Psi_hyz[i],
			d_cpsi_exz[i], d_cpsi_eyz[i], d_cpsi_hxz[i], d_cpsi_hyz[i],
			NX, NY, NZ_N);


        switch(source_tp){
            case 1:

                setGauss << < grid, block>> > (d_signal[i], d_signal_per_node[i], source_amp, v_mag_factor, dt, t_0, tau, source_size);

                break;

            case 2:

                break;

            case 3:

                setSine<< < grid, block>> > ( d_signal[i], d_signal_per_node[i], v_mag_factor, source_amp, source_freq, dt, pi, source_size) ;
                break;
        }
	}


    cudaSetDevice(0);

    HANDLE_ERROR(cudaMalloc((void**)&d_volt0, NUMDEV*source_bt));
    HANDLE_ERROR(cudaMalloc((void**)&d_curr0, NUMDEV*source_bt));

	//d_voltage e d_current
    cudaSetDevice(0);
	HANDLE_ERROR(cudaMalloc((void**)&d_voltage, source_bt));
    HANDLE_ERROR(cudaMallocHost((void**)&h_voltage, source_bt));
	HANDLE_ERROR(cudaMalloc((void**)&d_current, source_bt));
    HANDLE_ERROR(cudaMallocHost((void**)&h_current, source_bt));

	//Define 3D objects in the Domain
	cout << "CUDA Define 3D Objetcs in Simulation domain" << endl;

    if(brick_opt != 0){

        for(int j = 0; j < brick_num; j++){

            for (int i = 0; i < NUMDEV; i++){

                // set current device
                cudaSetDevice(i);

                //Calculate the gpu offset
                gpu_offset = i * NZ_N;

                //set the brick kernel
                setBrick << <grid3d, block3d >> > (NX, NY, NZ, NZ_N, gpu_offset,
                    d_material_3d_space_eps_x[i], d_material_3d_space_eps_y[i], d_material_3d_space_eps_z[i],
                    d_material_3d_space_sigma_e_x[i], d_material_3d_space_sigma_e_y[i], d_material_3d_space_sigma_e_z[i],
                    d_material_3d_space_mu_x[i], d_material_3d_space_mu_y[i], d_material_3d_space_mu_z[i],
                    d_material_3d_space_sigma_m_x[i], d_material_3d_space_sigma_m_y[i], d_material_3d_space_sigma_m_z[i],
                    brick_is[j], brick_js[j], brick_ks[j], brick_ie[j], brick_je[j], brick_ke[j],
                    brick_sigma_e_x[j], brick_sigma_e_y[j], brick_sigma_e_z[j], brick_eps_r_x[j], brick_eps_r_y[j], brick_eps_r_z[j],
                    brick_sigma_m_x[j], brick_sigma_m_y[j], brick_sigma_m_z[j], brick_mu_r_x[j], brick_mu_r_y[j], brick_mu_r_z[j]);

            }
        }
	}


	//Snap the sigma_e and sigma_mu according to the cells around
	//related to the 3D objects
	cout << "CUDA Snapping Material Grid for 3D Objects" << endl;
	for (int i = 0; i < NUMDEV; i++)
	{

		// set current device
		cudaSetDevice(i);

		//Calculate the gpu offset
		gpu_offset = i * NZ_N;
		//snap
		setSigmaEpsSigmaMu << <grid3d, block3d>> > (NX, NY, NZ, NZ_N, gpu_offset,
			d_material_3d_space_eps_x[i], d_material_3d_space_eps_y[i], d_material_3d_space_eps_z[i],
			d_material_3d_space_sigma_e_x[i], d_material_3d_space_sigma_e_y[i], d_material_3d_space_sigma_e_z[i],
			d_material_3d_space_mu_x[i], d_material_3d_space_mu_y[i], d_material_3d_space_mu_z[i],
			d_material_3d_space_sigma_m_x[i], d_material_3d_space_sigma_m_y[i], d_material_3d_space_sigma_m_z[i],
			d_eps_r_x[i], d_sigma_e_x[i], d_eps_r_y[i], d_sigma_e_y[i], d_eps_r_z[i], d_sigma_e_z[i],
			d_mu_r_x[i], d_sigma_m_x[i], d_mu_r_y[i], d_sigma_m_y[i], d_mu_r_z[i], d_sigma_m_z[i]);

	}


	cout << "CUDA Define 2D Objects in Simulation Domain" << endl;

	//Defining the 6 sector for the dipole
	for (int i = 0; i < 0; i++) {
        /*
		//Defining the points for the sector
		sectorPoints(raio[i], sec_angle[i], rot_angle[i], ponto1, ponto2[i], ponto0[i],
			&px0, &py0, &pz0, &px1, &py1, &px2, &py2,
			gap, 0.05, 0.03, i,
			pml_x_n, pml_y_n, pml_z_n,
			air_buff_x_n, air_buff_y_n, air_buff_z_n,
			dx, dy, dz);

		for (int j = 0; j < 0; j++){
				// set current device
				cudaSetDevice(j);

				//Calculate the gpu offset
				gpu_offset = j * NZ_N;

				//define the sector on the simulation domain
				//defineSec << <grid3d, block3d >> > (NX, NY, NZ, NZ_N, gpu_offset, MAX_X, MAX_Y, px0, py0, px1, py1, px2, py2, pz0, raio[i], sigma_sector_e_x, sigma_sector_e_y, d_sigma_e_x[i], d_sigma_e_y[i], dx, dy, dz);
			}
			*/
		}



	//create 2D Objects in Material Grid, such as conduction plates
	if(pec_opt != 0){

        for(int j = 0; j< pec_num; j++){

            for (int i = 0; i < NUMDEV; i++) {
                // set current device
                cudaSetDevice(i);

                //Calculate the gpu offset
                gpu_offset = i * NZ_N;

                //define the sector on the simulation domain
                definePlateZ << <grid3d, block3d >> > (NX, NY, NZ, NZ_N, gpu_offset, d_sigma_e_x[i], d_sigma_e_y[i], pec_sigma_e_x[j], pec_sigma_e_y[j], pec_is[j], pec_js[j], pec_ks[j], pec_ie[j], pec_je[j], pec_ke[j]);


                //define the sector on the simulation domain
                //definePlateZ << <grid3d, block3d >> > (NX, NY, NZ, NZ_N, gpu_offset, d_sigma_e_x[i], d_sigma_e_y[i], plate1_sigma_e_x, plate1_sigma_e_y, plate1_is, plate1_js, plate1_ks, plate1_ie, plate1_je, plate1_ke);

                //define the sector on the simulation domain
                //definePlateZ << <grid3d, block3d>> > (NX, NY, NZ, NZ_N, gpu_offset, d_sigma_e_x[i], d_sigma_e_y[i], plate2_sigma_e_x, plate2_sigma_e_y, plate2_is, plate2_js, plate2_ks, plate2_ie, plate2_je, plate2_ke);

                //define the sector on the simulation domain
                //definePlateZ << <grid3d, block3d>> > (NX, NY, NZ, NZ_N, gpu_offset, d_sigma_e_x[i], d_sigma_e_y[i], plate3_sigma_e_x, plate3_sigma_e_y, plate3_is, plate3_js, plate3_ks, plate3_ie, plate3_je, plate3_ke);

            }
        }
    }


	//Set the Domain Coefficiens Chxy etc...
	cout << "Set Domain Coefficients" << endl;

	//Coefficients for all the gpus
	for (int i = 0; i < NUMDEV; i++){
		// set current device
		cudaSetDevice(i);

		//Calculate the gpu offset
		gpu_offset = i * NZ_N;
		//set the coefficients
		setCoefficients << <grid3d, block3d>> > (NX, NXX, NY, NYY, NZ, NZ_N, gpu_offset,
			d_Cexe[i], d_Cexhz[i], d_Cexhy[i], d_Cexj[i], d_eps_r_x[i], d_sigma_e_x[i],
			d_Ceye[i], d_Ceyhx[i], d_Ceyhz[i], d_Ceyj[i], d_eps_r_y[i], d_sigma_e_y[i],
			d_Ceze[i], d_Cezhy[i], d_Cezhx[i], d_Cezj[i], d_eps_r_z[i], d_sigma_e_z[i],
			d_Chxh[i], d_Chxey[i], d_Chxez[i], d_Chxm[i], d_mu_r_x[i], d_sigma_m_x[i],
			d_Chyh[i], d_Chyez[i], d_Chyex[i], d_Chym[i], d_mu_r_y[i], d_sigma_m_y[i],
			d_Chzh[i], d_Chzex[i], d_Chzey[i], d_Chzm[i], d_mu_r_z[i], d_sigma_m_z[i],
			dx, dy, dz, dt,
			eps_0, pi, mu_0);
	}



	//Define the CPML boudary Condition
	cout << "Define Boundary Conditions" << endl;
	for (int i = 0; i < NUMDEV; i++){
		// set current device
		cudaSetDevice(i);

		//Calculate the gpu offset
		gpu_offset = i*(NZ_N);

		//set ABC kernel
		setABCPML << <grid3d, block3d >> > (NX, NXX, NY, NYY, NZ, NZ_N, gpu_offset,
			d_cpml_b_ex[i], d_cpml_a_ex[i], d_cpml_b_mx[i], d_cpml_a_mx[i],
			d_cpml_b_ey[i], d_cpml_a_ey[i], d_cpml_b_my[i], d_cpml_a_my[i],
			d_cpml_b_ez[i], d_cpml_a_ez[i], d_cpml_b_mz[i], d_cpml_a_mz[i],
			d_cpsi_eyx[i], d_cpsi_ezx[i], d_cpsi_hyx[i], d_cpsi_hzx[i],
			d_cpsi_exy[i], d_cpsi_ezy[i], d_cpsi_hxy[i], d_cpsi_hzy[i],
			d_cpsi_exz[i], d_cpsi_eyz[i], d_cpsi_hxz[i], d_cpsi_hyz[i],
			d_Cexhz[i], d_Cexhy[i], d_Ceyhx[i], d_Ceyhz[i], d_Cezhy[i], d_Cezhx[i], d_Chxey[i], d_Chxez[i], d_Chyez[i], d_Chyex[i], d_Chzex[i], d_Chzey[i],
			pml_x_n, pml_y_n, pml_z_n + gpu_offset, pml_x_p, pml_y_p, pml_z_p + gpu_offset,
			dx, dy, dz, dt,
			eps_0, pi, mu_0);

	}


	//Define the Domain Coefficients alatered by the source
	cout << "Define Source in Simulation domain" << endl;
	//source in the domain
	for (int i = 0; i < NUMDEV; i++)
	{

		// set current device
		cudaSetDevice(i);

		//Calculate the gpu offset
		gpu_offset = i * NZ_N;

		//define the source kernel
        switch(source_direction){

            case 1:

                defineSourceX << <grid3d, block3d >> >(NX, NXX, NY, NYY, NZ, NZ_N, gpu_offset, source_is, source_js, source_ks, source_ie, source_je, source_ke, d_Cexe[i], d_Cexhz[i], d_Cexhy[i], d_Cexj[i], d_eps_r_x[i], d_sigma_e_x[i], rs, dx, dy, dz, dt, eps_0);

                break;

            case 3:

                //defineSourceZ<<<grid3d, block3d>>>(NX, NXX, NY, NYY, NZ, NZ_N, gpu_offset, PA[i], PB[i], source_is, source_js, source_ks, source_ie, source_je, source_ke,   source_ks_ori, source_ke_ori, d_Ceze[i],  d_Cezhy[i], d_Cezhx[i], d_Cezj[i], d_eps_r_z[i], d_sigma_e_z[i], rs, dx, dy, dz, dt, eps_0);
                defineSourceZ<<<grid3d, block3d>>>(NX, NXX, NY, NYY, NZ, NZ_N, gpu_offset, source_is, source_js, source_ks, source_ie, source_je, source_ke, d_Ceze[i],  d_Cezhy[i], d_Cezhx[i], d_Cezj[i], d_eps_r_z[i], d_sigma_e_z[i], rs, dx, dy, dz, dt, eps_0);

                break;
        }

	}

	//Define Resistor in the simulation domain
    cout << "Define Resistor in Simulation domain" << endl;
	//source in the domain
    if(resistor_direction != 0) {

        for (int i = 0; i < NUMDEV; i++){
            // set current device
            cudaSetDevice(i);

            //Calculate the gpu offset
            gpu_offset = i * NZ_N;

        //    defineResistor<<<grid3d,block3d>>>(NX, NXX, NY, NYY, NZ, NZ_N, gpu_offset, PA[i], PB[i], resistor_is, resistor_js, resistor_ks, resistor_ie, resistor_je, resistor_ke, resistor_ks_ori, resistor_ke_ori, resistor_direction, d_Ceze[i], d_Cezhy[i], d_Cezhx[i], d_eps_r_z[i], d_sigma_e_z[i], resistor_resist, dx, dy, dz, dt, eps_0);

            defineResistor<<<grid3d,block3d>>>(NX, NXX, NY, NYY, NZ, NZ_N, gpu_offset, resistor_is, resistor_js, resistor_ks, resistor_ie, resistor_je, resistor_ke, resistor_direction, d_Ceze[i], d_Cezhy[i], d_Cezhx[i], d_eps_r_z[i], d_sigma_e_z[i], resistor_resist, dx, dy, dz, dt, eps_0);


        }
    }
	//FDTD Calculation
	cout << "FDTD CALC" << endl;

    //synchronize devices
	for (int i = 0; i<NUMDEV; i++) {
		cudaSetDevice(i);
		cudaDeviceSynchronize();
	}

	for(int m = 0; m < n_t_steps; m++){

        solver( NUMDEV, m,  k_real,  NX,  NXX,  NY,  NYY,  NZ,  NZ_N,  gpu_offset,
        grid3d, block3d, grid2d, block2d,
        pml_x_n,  pml_x_p,  pml_y_n,  pml_y_p,  pml_z_n,  pml_z_p,
        d_Ex,   d_Jx,   d_Cexe,   d_Cexhz,   d_Cexhy,   d_Cexj,
        d_Ey,   d_Jy,   d_Ceye,   d_Ceyhx,   d_Ceyhz,   d_Ceyj,
        d_Ez,   d_Jz,   d_Ceze,   d_Cezhy,   d_Cezhx,   d_Cezj,
        d_Hx,   d_Mx,   d_Chxh,   d_Chxey,   d_Chxez,   d_Chxm,
        d_Hy,   d_My,   d_Chyh,   d_Chyez,   d_Chyex,   d_Chym,
        d_Hz,   d_Mz,   d_Chzh,   d_Chzex,   d_Chzey,   d_Chzm,
        d_gEx, d_gEy, d_gHx, d_gHy,
        d_cpml_b_mx,   d_cpml_a_mx,
        d_cpml_b_my,   d_cpml_a_my,
        d_cpml_b_mz,   d_cpml_a_mz,
        d_cpml_b_ex,   d_cpml_a_ex,
        d_cpml_b_ey,   d_cpml_a_ey,
        d_cpml_b_ez,   d_cpml_a_ez,
        d_Psi_eyx,   d_Psi_ezx,   d_Psi_hyx,   d_Psi_hzx,
        d_cpsi_eyx,   d_cpsi_ezx,   d_cpsi_hyx,   d_cpsi_hzx,
        d_Psi_exy,   d_Psi_ezy,   d_Psi_hxy,   d_Psi_hzy,
        d_cpsi_exy,   d_cpsi_ezy,   d_cpsi_hxy,   d_cpsi_hzy,
        d_Psi_exz,   d_Psi_eyz,   d_Psi_hxz,   d_Psi_hyz,
        d_cpsi_exz,   d_cpsi_eyz,   d_cpsi_hxz,   d_cpsi_hyz,
        d_signal_per_node,  source_is,  source_js,
        source_ks,  source_ie,  source_je,  source_ke,
        sampled_voltage_is,  sampled_voltage_js,  sampled_voltage_ks,  sampled_voltage_ie,  sampled_voltage_je,  sampled_voltage_ke,
        volt_NX,  volt_NY,  volt_NZ_N,
        E,  volt_offset,
        sampled_current_is,  sampled_current_js,  sampled_current_ks,  sampled_current_ie,  sampled_current_je,  sampled_current_ke,
        current_NX,  current_NY,  current_NZ_N,
        Hx,   Hy,   Hz,
        stream_copy,   stream_compute, event_i, event_j,
        size_bt,   d_sigma_e_x,   d_sigma_e_y,   d_sigma_e_z,   d_current_tran);

    }

    //Calculate volt and current using multigpu
    calcVoltCurrent(NUMDEV, dx,  dy,  dz,  n_t_steps, gpu_offset, source_bt, grid, block,
        volt_NX, volt_NY, volt_NZ, volt_NZ_N, sampled_voltage_is, sampled_voltage_js, sampled_voltage_ks, sampled_voltage_ie, sampled_voltage_je, sampled_voltage_ke,
        d_volt_tran, E, voltage_direction,
        current_NX, current_NY, current_NZ,  current_NZ_N,  sampled_current_is, sampled_current_js, sampled_current_ks, sampled_current_ie, sampled_current_je, sampled_current_ke, d_current_tran, Hx, Hy, Hz, current_direction,
        d_volt0, d_voltage, d_curr0, d_current);

    HANDLE_ERROR(cudaMemcpy(h_voltage, d_voltage, source_bt, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(h_current, d_current, source_bt, cudaMemcpyDeviceToHost));
    saveTDomain(n_t_steps, h_voltage, 1003);
    saveTDomain(n_t_steps, h_current, 1004);

    //synchronize devices
	for (int i = 0; i<NUMDEV; i++) {
		cudaSetDevice(i);
		cudaDeviceSynchronize();
	}

	//obtain fft
   // calc_otm( d_voltage, d_current, n_t_steps, rs, dt, grid, block);

    //synchronize devices
	for (int i = 0; i<NUMDEV; i++) {
		cudaSetDevice(i);
		cudaDeviceSynchronize();
	}

    cout << "Free memory" << endl;


}



#endif
