#ifndef CODIM1_H
#define CODIM1_H
typedef enum
{
allocate_mem=0,
check_start_point,
find_transverse,
path_start_a,
path_start_b,
path_advance,
free_mem
} SimpconTask;

void codim1
(
int const n,         // dimension of domain
int const nu,
double* y,           // [n+1] point on the zero curve
double* u,           // [n+1] point on the zero curve
double const grain,  // size for simplicies
double* cum_al,      // arc-length so far ...
FILE* fp_simp,       // file for output data
SimpconTask const task,
void (*user_func)(int const,int const,const double*,
		  double*,double*,const void*),
const void* param    // user-defined; can be null ...
);
#endif
