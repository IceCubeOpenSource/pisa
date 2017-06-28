#include "mosc3.h"
#include "mosc.h"

#include <stdio.h>

// This improves readability of the equations
#define re (0)
#define im (1)

__device__ void clear_complex_matrix(fType A[][3][2])
{
  // CUDA NOTE: Since I'm on the device, I don't need cudaMemset, only
  // memset, because this is called from an execution thread and will
  // refer to the thread's local memory.
  memset(A,0,sizeof(fType)*18);

  // Does this actually work? YES!
  //printf("A: %f %f %f\n",A[0][0][0],A[0][1][0],A[0][2][0]);
  //printf("A: %f %f %f\n",A[1][0][0],A[1][1][0],A[1][2][0]);
  //printf("A: %f %f %f\n",A[2][0][0],A[2][1][0],A[2][2][0]);

}

__device__ void copy_complex_matrix(fType A[][3][2], fType B[][3][2] )
{
  memcpy(B,A,sizeof(fType)*18);
}

__device__ void multiply_complex_matrix(fType A[][3][2], fType B[][3][2], fType C[][3][2] )
{
  for (unsigned i=0; i<3; i++) {
    for (unsigned j=0; j<3; j++) {
      for (unsigned k=0; k<3; k++) {
        C[i][j][0] += A[i][k][re]*B[k][j][re]-A[i][k][im]*B[k][j][im];
        C[i][j][1] += A[i][k][im]*B[k][j][re]+A[i][k][re]*B[k][j][im];
      }
    }
  }
}

__device__ void clear_real_matrix(fType R[3][3])
{
  memset(R,0,sizeof(fType)*9);
}


// Multiply complex 3x3 matrix and 3 vector: W = A X V
__device__ void multiply_complex_matvec(fType A[][3][2], fType V[][2], fType W[][2])
{
  for(unsigned i=0;i<3;i++) {
    W[i][re] = A[i][0][re]*V[0][re]-A[i][0][im]*V[0][im]+
      A[i][1][re]*V[1][re]-A[i][1][im]*V[1][im]+
      A[i][2][re]*V[2][re]-A[i][2][im]*V[2][im] ;
    W[i][im] = A[i][0][re]*V[0][im]+A[i][0][im]*V[0][re]+
      A[i][1][re]*V[1][im]+A[i][1][im]*V[1][re]+
      A[i][2][re]*V[2][im]+A[i][2][im]*V[2][re] ;
  }
}


// Complex conjugate all elements of 3x3 matrix A: B=A*
__device__ void conjugate_complex_matrix(fType A[][3][2], fType B[][3][2])
{
  for (unsigned i=0; i<3; i++){
    for (unsigned j=0; j<3; j++){
      B[i][j][re] = A[i][j][re];
      B[i][j][im] = -A[i][j][im];
    }
  }
}
// Complex conjugate all elements of 3x3 matrix A and transpose: B = (A^T)*
__device__ void conjugate_transpose_complex_matrix(fType A[][3][2], fType B[][3][2])
{
  for (unsigned i=0; i<3; i++){
    for (unsigned j=0; j<3; j++){
      B[j][i][re] = A[i][j][re];
      B[j][i][im] = -A[i][j][im];
    }
  }
}

// Add two complex 3x3 matrices: C_{ij} = A_{ij} + B_{ij} (for real & imaginary parts)
__device__ void add_complex_matrix(fType A[][3][2], fType B[][3][2], fType C[][3][2])
{
  for (unsigned i=0; i<3; i++) {
    for (unsigned j=0; j<3; j++) {
        C[i][j][re] = A[i][j][re] + B[i][j][re];
        C[i][j][im] = A[i][j][im] + B[i][j][im];
    }
  }
}

__device__ void convert_from_mass_eigenstate( int state, fType pure[][2],
                                              fType mixNuType[][3][2])
{
  fType mass[3][2];
  int    lstate  = state - 1;

  for (int i=0; i<3; i++) {
    mass[i][0] = ( lstate == i ? 1.0 : 0. );
    mass[i][1] = (                     0. );
  }
  // note: mixNuType is already taking into account whether we're considering
  // nu or anti-nu
  multiply_complex_matvec(mixNuType, mass, pure);

}

/* Calculate neutrino flavour transition amplitude matrix for neutrino (nutype > 0)
   or antineutrino (nutype < 0) with energy Enu traversing layer of matter of
   uniform density rho with thickness Len.
*/
__device__ void get_transition_matrix( int nutype, fType Enu, fType rho, fType Len,
                                       fType Aout[][3][2], fType phase_offset,
                                       fType mixNuType[3][3][2], fType nsi_eps[3][3],
                                       fType HVac2Enu[3][3][2], fType dm[3][3])
{
  fType dmMatVac[3][3], dmMatMat[3][3];
  fType HFull[3][3][2], HMat[3][3][2], HMatMassEigenstateBasis[3][3][2];
  clear_complex_matrix(HFull); clear_complex_matrix(HMatMassEigenstateBasis);

  /* Compute the matter potential including possible non-standard interactions
     in the flavor basis */
  getHMat(rho, nsi_eps, nutype, HMat);

  /* Get the full Hamiltonian by adding together matter and vacuum parts */
  add_HVac_HMat(Enu, HVac2Enu, HMat, HFull);

  /* Calculate modified mass eigenvalues in matter from the full Hamiltonian and
     the vacuum mass splittings */
  getM(Enu, rho, dm, dmMatMat, dmMatVac, HFull);
  getHMatMassEigenstateBasis(mixNuType, HMat, HMatMassEigenstateBasis);
  getAGen(Len, Enu, rho, mixNuType, dmMatVac, dmMatMat, HMatMassEigenstateBasis,
          Aout, phase_offset);

}
