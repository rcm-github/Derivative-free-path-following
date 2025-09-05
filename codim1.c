#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "codim1.h"
// LICENSE CPL Version 1.0; see LICENSE file ...
//#define CPLUSPLUS

/*
For plain C, comment out CPLUSPLUS  above and use

gcc codim1.c -lblas -llapack -lm -o codim1

to link in the standard numerical libraries BLAS and LaPack.

--------------------------------------------------------------------------------
Let  f()  be a continuous mapping from R^{n+1} into R^n.
The set ("level-set") of points  y \in R^{n+1}  such that  f(y) = 0 ,
and passing through a user-specified starting point  y_0
generically forms a one-dimensional manifold 
(a curve) in  R^{n+1} and the curve is described as having 
"co-dimension 1" in the  ambient space  of dimension (n+1).
Given such a starting point  y_0  such that  f(y_0) = 0,
this code attempts to compute a piecewise linear approximation
to said curve starting at  y_0  and continuing for a user-specified
arc-length. 

For engineering applications, it is often convenient to
imagine  y \in R^{n+1}  as partitioned into  (x,\mu)  where
x \in R^n and  \mu  is a single real number sometimes called the
"continuation parameter" or "bifurcation parameter". 
In our development below and in the code
\mu  is given somewhat special treatment however it is also
entirely valid to simply treat  y \in R^{n+1}  with no
coordinate distinguished.

This curve in represented as a piecewise linear approximation involving
a "scafold" of simplices hugging the zero curve and sharing
exactly one adjacent "facet", where a facet is the convex hull of
the vertex set of the simplex with one vertex removed. E.g., in
R^3, a simplex is a tetrahedron and a facet is a triangular face of
the tetrahedron. Baring some (rare) degenericies, the zero
curve intersects each scafold simplex in exactly two facets.

The code incorporates some features that make it
particularly suitable for {\it experimental} continuation in which
the evaluation of the residual ( f  above) is performed by
a piece of test equipment rather than a numerical
computation (i.e., measured rather than computed).
Of course, numerical evaluation of the
residual can be done as well and some examples of
residual computation by code are provided.

In particular, this code does NOT require partial derivatives
of  f() ! There are numerous published codes for path-following
(including even bifurcation detection along the curve)
[Hompack,Pitcon,AUTO] but these codes typically require very accurate derivative
information.  In an experimental setting, the only practical
way to get such derivatives is by finite-differencing which
is slow and tends to accentuate noise. Even in a purely
computational setting (i.e.  f() above is computed, not
measured) there are situations -- such as retrofitting a
path-following code to a legacy simulator -- in which
residual evaluation only without derivatives is advantageous.

Be advised that when performing such experimental or hardware-driven
sweeps, one should not expect ANYWHERE near the precision or
dynamic range of even single-precision computations, let alone
double precision. Double-precision computation usually supports
around 16 significant digits which -- in electronic language --
is something like 320 (!) dB. A typical experimental, measurement-based
continuation might provide -- on a good day -- 80 dB, which means something like
4 (!) significant digits. In addition, measured data will
always contain some noise and not be entirely re-producible.
The theme of this code is robustness over anything else.

It is assumed that a single residual evaluation (measurement)
is much, much slower than any reasonable matrix manipulations
(e.g., solving  ~n  linear systems of dimension  ~n ).
For moderate  n  as described above (n < 20),
these sort of computations are extremely fast on a modern computer.
The cost of one residual measurement will typically far outweigh
any such numerical manipulations, once residual values are available.

Particularly for experimental continuation, it may be useful and more
efficient to capture certain user-defined functions at the same time as evaluating
residuals at the vertices of a simplex. This is facilitated by setting the
parameter  nu  to more than 0 and providing a vector  u[]  large enough
to hold the values. After computing an approximate exit point at the
facet of a simplex (which is a convex combination of the vertex coordinates)
the same combination is applied to  u[]  values at vertices to provide
approximate interpolated values at the exit point.

To start a continuation, the user calls the subroutine with  task == allocate_mem
which sets up various data structures in static storage, so they remain
across subsequent calls to the subroutine. 

The user then calls the code with  task == check_start_point  to verify
that  ||f(y_0)||  is sufficiently small; i.e., less than  grain/100 .
Here  grain  is a global constant that sets the size of the simplices.
If the initial starting point fails this test, it is unlikely
that subsequent path following will succeed.

If residuals are computed to the typical accuracy of
double precision arithmetic, then  grain  can take essentially
any value > 0; the only issue will be the fidelity of the resulting
zero curve and the number of steps required to traverse it.
However, for a measured residual, some experimentation with  grain
is typically required, depending on the resolution of the measurement. 
I.e., if  grain  is so small that the resulting simplices fall below
the resolution of the measurement, the path following will most likely
crash.

Next, for  task ==  find_transverse , the code constructs an initial
simplex  sigma0  centered on the user-supplied initial
point  y_0 , with user-supplied size controlled by the  grain  
parameter. The code looks for exactly
two facets of  sigma0  that are _transversal_ to the zero curve
(i.e., intersect the zero curve in a point),
computes approximate crossing points (i.e., on the
zero curve and on one facet)  y1a  and  y1b , and finally
reports these to the user.

Typically, these two points (y1a,y1b) correspond to
(small) steps in different directions along the zero
curve and the user must select the desired direction,
for example to achieve an increase of an experimental
quantity, such as  \mu . Typically the point  y_0  is chosen
so that the  \mu  can be used as a local parameter for the
zero curve in the vicinity of  y_0 . I.e., not
near a turning point in the curve w.r.t. \mu.
It it also typical to chose  y_0 = (x_0,\mu_0)  
such that the system with  \mu  fixed at
\mu_0  has a unique solution (*) in  R^n ;
see the diagram below. Here the user would
select  y1b  to advance in arc-length into the
interesting portion of the zero curve.


     |                 ....> increasing arc-length for  y1b
     |               ..
     |                 ..
     |                    ....
     |                        ..
R^n  x                          .. << turning point
     |                        ..
     |                    ....
     |  y1a y1b         ...
 x_0 |  .^.*.^........
     |
     |--------------- \mu --------------
         \mu_0                  ^ (bad choice for \mu_0  near turning point)
           ^(good choice for \mu_0)
                          ^ (dubious choice for \mu_0 because multiple solutions)

The user then calls the code again with  task == path_start_[ab]
to select desired direction along the curve. 
The code will return with the point where approximately the curve
crosses the selected facet. The routine uses static allocation
of data structures to preserve the data structures
created and computed during the setup call (with  task
== allocate_mem ).  The code will perform some initialization
and return with the cumulative arc length.

Then, the user calls the code repeatedly with  task==path_advance
and monitors the value of the cumulative arc length, or some
other indication of progress along the zero curve.
For each re-entry, the code enumerates
a new adjacent simplex (and their points of intersection
with the zero curve)  and returns the new crossing point.

For each new point enumerated along the curve, the code
returns to the calling program with a point near
the zero curve in the third argument (y).
The calling code can then, for example, print one or
more coordinates of  y  to a file or evaluate the
residual at  y . These operations are outside the scope
of the path-following code.

The code also allows the user to record  nu >= 0  user-defined
functions at each vertex of a simplex when calling the
residual subroutine. The code computes an exit point on a transversal facet
as a convex combination of the coordinates of the vertices of the facet.
If  nu > 0  the code will compute the same convex combination of
the values of the user-defined functions and return this value
along the the coordinates of the exit point.
This feature is included primarily for experimental applications.
In some cases, obtaining the values of the user-defined functions
takes almost no additional effort over obtaining the values of the
residuals, but the user really wants the value of the user-defined
functions at the exit point. It may be a savings of work to estimate
the values of the user-defined functions at the exit by interpolation rather
than re-measuring the values at the computed exit point.

The programming paradigm of doing static memory allocation, then
returning to the calling program repeatedly for each
new point on the zero curve is a bit like the idea
of "reverse call", but not completely because the code
still expects the user to provide a forward-call subroutine
for evaluation of  f() . A better term might "single step" mode.

In a true reverse-call implementation, the path-following
code would return to the calling program every time
it needs to evaluate  f()  at a point, so the calling
program could evaluate  f()  in its own environment.
The implementation of this technique is a bit tricky.
We find the combination of single-step mode with the
(void*)param  pointer for a user-defined structure
to be a reasonable compromise.

Here is a possible application of the code.
A waveform synthesizer is programmed with Fourier
coefficients to generate a stimulus for
a device under test (DUT). Then, an oscilloscope
is used to measure the current through the waveform generator
as a residual to be nulled to zero.
Suppose the stimulus and resulting current are both
represented by 6 Fourier coefficients.
Then 13 real numbers (DC + sin/cos for each term) would
be required to represent the waveforms implying
n = 14 , if a continuation parameter is also used.
The residual is measured as 13 real numbers giving
a mapping R^{14} -> R^{13}.

It is better geometrically if the code can manipulate
simplices in which each dimension is more-or-less
on the same scale!!! This is the responsibility of the calling
program. For example, if coordinates are a mixture of
voltages and currents, it might be advisable for the
path-following code to work with the logarithm of
a current.

Matrix manipulation and linear algebra are done by calling
the standard Fortran BLAS (basic linear algebra subroutine)
and the LaPack codes. A reader of this code must be very
familiar with the BLAS and in particular, understand the
so-called "stride" arguments. RTFM, as they say.
As a matter of programming style, the code attempts
to avoid manipulation of individual matrix or vector
subscripts using "for" loops, preferring instead
to hides such details in the level-1 BLAS, like  dcopy  or  dscal .

The user must provide a routine to compute or measure the residual  f  
The  void*  argument can be used for example to point to
global data needed by a piece of test equipment.
Note also the discussion of  u[]  and  nu  above.

// compute or measure a mapping from R^{n+1} -> R^n
void resid
(
int const n,      // dimension of range
int const nu,     // # user-defined functions; nu >= 0
double* const x,  // [n+1]; point of evaluation
double* rsd,      // [n]; residual
double* u,        // [nu]; return values of user-defined functions
const void* param // user defined, passed in from solver
);

Acknowledgements:
This code is based on ideas from the classic Algower and Georg reference,
as well as notes on Mike Henderson's website (IBM Watson Research).
Steve Mackey provided the equations to represent a helix
in R^3 in implicit form.
*/

// take address of these for Fortran routines
static int const _i0 = 0;
static int const _i1 = 1;
static double const _d0 = 0;
static double const _d1 = 1;
static double const _dn1 = -1;

static double min(double const a,double const b)
{
  return (a<b? a : b);
}

static double max(double const a,double const b)
{
  return (a>b? a : b);
}

// external routines, mostly Fortran (BLAS and Lapack)

#ifdef CPLUSPLUS
extern "C" void drandn_(...);
extern "C" double dnrm2_(...);
extern "C" double ddot_(...);
extern "C" void dscal_(...);
extern "C" void dcopy_(...);
extern "C" void daxpy_(...);
extern "C" void dgetrs_(...);
extern "C" void dgetrf_(...);
#else
extern void drandn_();
extern double dnrm2_();
extern double ddot_();
extern void dscal_();
extern void dcopy_();
extern void daxpy_();
extern void dgetrs_();
extern void dgetrf_();
#endif

static void freudenthal
(
int const nr,  // range dimension
double* s,    // [(nr+1) x (nr+2)]; vertices
int const ld  // >= nr+1
);

static void equilateral
(
int const nr,  // range dimension
double* s,    // [(nr+1) x (nr+2)]; vertices
int const lds // >= nr+1
);

static void show
(
  const char* title,
  double* dat,
  int const nrow,
  int const ncol,
  int const lda,
  const char* suffix,
  FILE* fp_o
);

/* 
********************** some test cases ******************
Residual function and starting point; a mapping from
R^{nr+1} -> R^nr .
*/

// diagonal line going through the origin in R^3
static void diag
(
int const nr,    // dimension of range
int const nu,
const double* y, // [nr+1]
double* rsd,     // [nr]
double* u,       // [nu]
const void* param
)
{
  assert( nr == 2 );
  assert( nu == 0 );
  rsd[0] = y[0] - y[1];
  rsd[1] = y[1] - y[2];
}

// unit circle in the plane
static void u_circ
(
int const nr,    // dimension of range
int const nu,
const double* v, // [nr+1]; dimension of domain
double* rsd,     // [nr]
double* u,       // [nu]
const void* param
)
{
  assert( nr == 1 );
  assert( nu == 1 );
  rsd[0] = pow(v[0],2) + pow(v[1],2) - 1;
  // distance from starting point ...
  u[0] = sqrt(pow(v[0]-1,2) + pow(v[1],2));
}

// Unit sphere intersects plane orthogonal to the y[0] axis
static void sph
(
int const nr,    // dimension of range
int const nu,
const double* y, // [nr+1]
double* rsd,     // [nr]
double* u,       // [nu]
const void* param
)
{
  double const rad = 1;
  assert( nr == 2 );
  assert( nu == 0 );
  rsd[0] = pow(y[0],2) + pow(y[1],2) + pow(y[2],2) - rad;
  rsd[1] = y[0];
}

static void sph_init
(
int const nr, // dimension of range
double* y0   // [n+1]
)
{
  assert( nr == 2 );
  double const rad = 1;
  y0[0] = 0.0;
  y0[1] = sqrt(rad);
  y0[2] = 0.0;
};

/* Steve Mackey's 3D helix example (!) ...
It is easy to describe a helix in 3D in
parametric form; it is less obvious how to
accomplish this in implicit form as the
zero set of a mapping from R^3 into R^2.
See below and be amazed ...
*/

static void helix
(
int const nr,      // dimension of range
int const nu,      // # user functions; >= 0
const double* y,   // [nr+1]
double* rsd,       // [nr]
double* u,         // [nu]
const void* param  // unused ...
)
{
  assert( nr == 2 );
  assert( nu == 0 );
  double const a = 0.65;
  double const b = 1.35;
  rsd[0] = y[0] * sin(y[2]) - y[1]*cos(y[2]);
  rsd[1] = pow(y[0],2)/(a*a) + pow(y[1],2)/(b*b) - 1;
};

/*
Find starting point for implicitly-defined helix by using
external reverse-call zero finder ...
*/
static void helix_init
(
int const nr,
double* y0  //[n+1]
)
{
  assert( nr == 2 );
  y0[0] = 0.459619;
  y0[1] = 0.954594;
  y0[2] = 1.122073;
  return;
#if 0
  double const a = 0.65;
  double const b = 1.35;
  int code;
  double ax = 0;
  double bx = M_PI;
  double farg;
  double const tol = 1e-13;
  int count;

  // select y[0],y[1] to satisfy the second residual
  y0[0] = a/sqrt(2);
  y0[1] = b/sqrt(2);
  // find  y0[2]  to satisfy first residual
  code = 0;
  rczero_(&ax,&bx,&tol,&y0[2],&farg,&code);
  assert( code == 1 );
  count = 0;
  while((code == 1)&&(count<20))
  {
    farg = y0[0]*sin(y0[2]) - y0[1]*cos(y0[2]);
    count++;
    rczero_(&ax,&bx,&tol,&y0[2],&farg,&code);
  }
#endif
};

//************************ end of test cases ********************

/*
A simplex in the domain space is (n+2) points,
each of dimension  n+1  stored in so-called
"column major" order, which is the Fortran standard
for matrices; serial memory addresses run "down" the page.

      n+2
[.  .      .   ]
[.  .      .   ]
[.  .      .   ]
[s0 s1 ... sn1 ]  n+1
[.  .      .   ]
[.  .      .   ]
[.  .      .   ]

memory:
 0    1    2   3 ...   n+1	

[0    n+1    2n+2      (n+1)*(n+1)  ]
[1    n+2                           ]
[2                                  ]
[.                                  ] n+1
[.                                  ]
[.                                  ]
[n    2n+1   3n+2      (n+1)*(n+1)+n]  

Note that this format allows the residual code to be called
on a _contiguous_ run of  n+1  numbers for each vertex.

The code below generates a canonical  Freudenthal  simplex,
in which all vertices are also vertices of the unit hypercube.
*/

void bary_center
(
int const n,
const double* sigma, //[(n+1)*(n+2}
double* bc           //[n+1] return bary-center of sigma
)
{
  int const np1 = n+1;
  double const rd = 1.0/(double)(n+2);
  dcopy_(&np1,&_d0,&_i0,bc,&_i1);
  for( int k = 0; k < n+2; k++ )
    daxpy_(&np1,&_d1,&sigma[k*(n+1)],&_i1,bc,&_i1);
  dscal_(&np1,&rd,bc,&_i1);
}

/*
This code generates a simplex in  R^{n+1}  with all
sides equal length and forming equal angles.
Hacked from some code found on Wikipedia (?)
*/
static void equilateral
(
int const n, // range dimension
double* s,   // [lds x (n+2)]; vertices
int const lds
)
{
  int const np1 = (n+1);
  double const ndn1 = -1.0/(double)(n+1);
  assert( n >= 0 );

  memset(s,0,lds*(n+2)*sizeof(double));
  // first column is unit vector (1,0,...,0)
  dcopy_(&n,&_d0,&_i0,&s[1],&_i1);
  s[0] = 1;

  /*
  since all sides have length 1,
  dot product of s_0 with all other columns must equal -1/(n+1);
  hence first entry of all other vertices is this number.
  */
  dcopy_(&np1,&ndn1,&_i0,&s[lds],&lds);

  /*
  Do bulk of entries in columns 1...n by
  alternating length constraint and angle constraint
  */

  for( int col = 1; col <= n-1; col++ )
  {
    int colrem = n+1 - col - 1;
    assert( colrem >= 0 );
    // length constraint
    double const len = dnrm2_(&col,&s[col*lds],&_i1);
    s[col*lds+col] = sqrt(1 - len*len); // diagonal entry
    dcopy_(&colrem,&_d0,&_i0,&s[col*lds+col+1],&_i1);

    // angle constraint
    for( int col1 = col+1; col1 <= n+1; col1++ )
    {
      double const dp = ddot_(&col,&s[col*lds],&_i1,&s[col1*lds],&_i1);
      s[col1*lds+col] = ((-1.0/(double)(n+1))-dp) / s[col*lds+col];      
    }
  }

  // use length constraint to fill in last two entries ...
  double const lenp = dnrm2_(&n,&s[n    *lds],&_i1);
  double const lenm = dnrm2_(&n,&s[(n+1)*lds],&_i1);
  s[n*lds    +n] =  sqrt(1 - lenp*lenp );
  s[(n+1)*lds+n] = -sqrt(1 - lenm*lenm );
}

//Freudenthal simplices tesselate space of any dimension.
static void freudenthal
(
int const n,  // range dimension
double* s,    // [lds x (n+2)]; vertices
int const lds // lds >= n+1
)
{
  assert( n >= 0 );
  int const np1 = n+1;
  double* bc = (double*)malloc((n+1)*sizeof(double));

  memset(s,0,lds*(n+2)*sizeof(double));

  for( int c = 0; c < n+2; c++ )
  {
    int const n0 = (n+1) - c;
    dcopy_(&n0,&_d0,&_i0,&s[c*lds],&_i1);
    dcopy_(&c,&_d1,&_i0,&s[c*lds+(n+1-c)],&_i1);
  }

  // move barycenter to origin ...
  bary_center(n,s,bc);

  for( int k = 0; k < n+2; k++ )
    daxpy_(&np1,&_dn1,bc,&_i1,&s[k*lds],&_i1);
  free(bc);
}

/*
Important data structures:

n+1 == dimension of ambient space; domain of residual mapping
n   == dimension of range space of residual mapping
"n  equations in  n+1  unknowns", "co-dimension 1"

A simplex; n+2  points in  (n+1)-dimensional space that
do not lie in any lower-dimensional sub-space; i.e.,
in "general position"

              n+2
[                             ]
[                             ]
[s_0  s_1  ...         s_{n+1}]  n+1
[                             ]
[                             ]

A  facet  of a simplex is any collection of  n+1
vertices; it can be identified by the index
of the vertex of the simplex which is _not_
included; hence there are n+2 of them:

          n+1
[                   ]
[                   ]
[v_0  v_1  ...  v_n ]  n+1
[                   ]
[                   ]

Observe that selecting a facet  \tau  of a simplex
\sigma  amounts to a copying operation in which exactly
one column of the  \sigma  data structure is deleted.

A residual is the result of the mapping f:R^{n+1} -> R^n
and is supposed to be zero or near zero:
A facet above is "labeled" by a residual for each
of its vertices: r_k = f(v_k) ...

The type  f_rsd  associates a residual with each
vertex of a facet.

           n+1
[                   ]
[                   ]
[r_0  r_1  ...  r_n ] n  
[                   ]

The type  s_rsd  associates a residual with each
vertex of a simplex; i.e., the simplex is "labeled".

           n+2
[                       ]
[                       ]
[r_0  r_1  ...  r_{n+1} ]  n
[                       ]

The type  su  associates  nu  user-function values with
each vertex of a simplex

          n+2
[u_0             u_{n+1}]
[                       ]
[                       ] nu >= 0
[w_0             w_{n+1}]

And the corresponding  fu  associates  nu  user-function values with
each vertex of a facet

          n+1
[u_0             u_{n}]
[                     ]
[                     ] nu >= 0
[w_0             w_{n}]

The idea is to keep the  s_rsd  and  su  structures up-to-date as new
simplices are enumerated, then obtain facets and their
associated  f_rsd  and  fu  by simply copying (which is cheap
and does not require new calls to the residual computation function).

When a new simplex is generated by "pivoting" across one facet
of an existing simplex, it is only necessary to do one new call to
the residual function to update the  s_rsd (and possibly the
su  structure)  associated with the
new simplex, because it shares  n  vertices with the previous one.
*/

//Utility function for printing ...
static void show
(
const char* title,
double* dat,
int const nrow,
int const ncol,
int const lda,
const char* suffix,
FILE* fp_o
)
{
  fprintf(fp_o,"%s",title);
  for( int r = 0; r < nrow-1; r++ )
  {
    for( int c = 0; c < ncol; c++ )
      fprintf(fp_o,"%15.7e ",dat[c*lda+r]);
    fprintf(fp_o,";\n");
  }
  for( int c = 0; c < ncol; c++ )
    fprintf(fp_o,"%15.7e ",dat[c*lda+(nrow-1)]);
  fprintf(fp_o,";");
  fprintf(fp_o,"%s",suffix);
  fflush(fp_o);
}

/*
Simplex  sigma0  has a vertex with index  p_index .
Perform a "pivot" across the opposite face to
construct a new simplex  sigma1  sharing this
opposite face but a new vertex, which will have
the same index in the new simplex.
The routine assumes that memory has
already been allocated for  sigma1 .

The code also updates the residuals for
the output simplex by copying, except for vertex
[p_index], which must be updated by the
calling program.

This code works for the Freudenthal triangularization.
The same basic idea can be extended to different triangularization,
but might need to be modified(!).
*/
static void pivot
(
int const n,
int const nu,
int const p_index,    // [0,n+1]; identify a vertex of  sigma0
double* const sigma0, // [(n+1) x (n+2)]
double* sigma1,       // [(n+1) x (n+2)]
double* const s_rsd0, // [n x (n+2)]
double* s_rsd1,       // [n x (n+2)]
double* const su0,    // [nu x (n+2)]
double* su1           // [nu x (n+2)]
)
{
  assert( 0 <= p_index );
  assert( p_index <= n+1 );
  int const np1 = n+1;

  for( int k = 0; k < n+2; k++ )
    if( k != p_index )
    {
      dcopy_(&np1,&sigma0[k*(n+1)],&_i1,&sigma1[k*(n+1)],&_i1);
      dcopy_(&n ,&s_rsd0[k*n], &_i1,&s_rsd1[k*n], &_i1);
      dcopy_(&nu,&su0[k*nu],   &_i1,&su1[k*nu],   &_i1);
    }

  // now, do new "pivoted" vertex
  if( p_index == 0 ){
    dcopy_(&np1       ,&sigma0[(n+1)*(n+1)]     ,&_i1,&sigma1[p_index*(n+1)],&_i1);
    daxpy_(&np1,&_dn1,&sigma0[p_index*(n+1)]    ,&_i1,&sigma1[p_index*(n+1)],&_i1);
    daxpy_(&np1,&_d1 ,&sigma0[(p_index+1)*(n+1)],&_i1,&sigma1[p_index*(n+1)],&_i1);
  }else if( p_index == n+1 ){
    dcopy_(&np1      ,&sigma0[(p_index-1)*(n+1)],&_i1,&sigma1[p_index*(n+1)],&_i1);
    daxpy_(&np1,&_dn1,&sigma0[p_index*(n+1)]    ,&_i1,&sigma1[p_index*(n+1)],&_i1);
    daxpy_(&np1,&_d1 ,&sigma0[0*(n+1)]          ,&_i1,&sigma1[p_index*(n+1)],&_i1);
  }else{
    dcopy_(&np1      ,&sigma0[(p_index-1)*(n+1)],&_i1,&sigma1[p_index*(n+1)],&_i1);
    daxpy_(&np1,&_dn1,&sigma0[p_index*(n+1)]    ,&_i1,&sigma1[p_index*(n+1)],&_i1);
    daxpy_(&np1,&_d1 ,&sigma0[(p_index+1)*(n+1)],&_i1,&sigma1[p_index*(n+1)],&_i1);
  }
}

/*
This routine determines if a facet is "transversal" to the zero curve.
There are  n+1  residuals (because a facet has one fewer
vertices than a simplex) each of dimension  n .
We seek a point  y  _interior_ to the facet such that
the residual evaluated at this point is zero. If such
a point cannot be found, then the zero curve is
not transversal to the facet. In other words, the
zero curve can intersect the hyper-plane containing the
facet, but the point of intersection is outside the facet.

Another possibility (rather rare) is that the zero
curve (more precisely, its linear approximation) is
parallel to the facet so does not intersect it.
In this case, the matrix formed
below is singular, which is flagged by the program
Again, the zero curve is not transversal to the facet.

Express  y  in barycentric coords using the vertices (v_k) of the facet:

  beta_0 v_0 + ... + beta_n v_n = y (which is essentially exact)

We want  f(y)=0, so write

  f(beta_0 v_0 + ... + beta_n v_n) = f(y) = 0

But now assume  f  can be approximated locally by a linear
function; by abuse of notation we use  f  for this linear approx;
we can write (approximately):

  beta_0 f(v_0) + ... + beta_n f(v_n) ~= 0

Now, in barycentric coords,

  beta_0 + ... + beta_n = 1
s
so we can write

  beta_0 f(v_0) + ... beta_{n-1} f(v_{n-1}) +
    (1-beta_0-beta_1-...-beta_{n-1}) f(v_n}) = 0

or
  
  beta_0 (f(v_0)-f(v_n)) + beta_1 (f(v_1)-f(v_n) + ...
  beta_{n-1} (f(v_{n-1})-f(v_n))
  =
  -f(v_n)

which can be expressed as a linear algebra solve with a square matrix
of dimension  n .

Solve this matrix problem for  (beta_0,...,beta_{n-1}) 
then recover beta_n as 1-SUM(beta_{0}...beta_{n-1}) .
Test if all the components of the solution are between 0 and 1.
within a constant  TransEps , i.e., in the interval [TransEps,1-TransEps].
If so, then declare the face transversal and compute an approximate
point where zero curve crosses the facet.

Numerical issue: The system above _can_ be formulated as a square
matrix problem of dimension  n  using the matrix  F_{\tau}  for
facet  \tau :

[1      1    ...    1     ]         [1   ]
[                         ]         [0   ]
[f(v_0) f(v_1) ...  f(v_n)] \beta = [... ]
[                         ]         [0   ]

which imposes the constraint that \sigma \beta = 1.

However, our computational experience leads us to prefer the
formulation of dimension  n-1  , performing algebraic
substitution for  \beta_n ; the second formulation is
smaller and (more importantly) seems significantly better
conditioned.
*/

static int transversal
(
int const n,
const double* facet,// [(n+1) x (n+1)]; return 1 iff this facet transversal to the zero curve
const double* lab,  // [nx(n+1)]; facet "labels", i.e. residuals; f(\tau)
double* y1,         // [n+1] output; point on facet (and near zero curve), if transversal; else 0
double* beta,       // [n+1] output; b.c. coords of  y1  on that facet
double* mat,        // [n x n] pre-allocated memory for matrix
int* ipiv,          // [n] pre-allocated memory for pivot sequence
const void* param
)
{
  int const sq_n = n * n;
  int const np1 = n+1;
  int info;
  int rc;  // return code ...

  dcopy_(&np1,&_d0,&_i0,y1,&_i1); // default, just to avoid uninitialized data

  for( int col = 0; col < n; col++ )
  {
    dcopy_(&n,&lab[col*n],&_i1,&mat[col*n],&_i1);
    daxpy_(&n,&_dn1,&lab[n*n],&_i1,&mat[col*n],&_i1);
  }

  // compute  beta  unless system is singular ... 
  dgetrf_(&n,&n,mat,&n,ipiv,&info);

  if( info != 0 ) // singular matrix ...
  {
    fprintf(stderr,"singular matrix\n");
    rc = 0;
  }
  else{
    // rhs for linear solve is negative of  f(v_n) ...
    dcopy_(&n,&lab[sq_n],&_i1,beta,&_i1);
    dscal_(&n,&_dn1,beta,&_i1);

    // solve  nxn  linear system
    dgetrs_("N",&n,&_i1,mat,&n,ipiv,beta,&np1,&info);
    assert( info == 0 );

    beta[n] = 1;
    for( int k = 0; k < n; k++ )
      beta[n] -= beta[k];

    rc = 1;
    for( int k = 0; k < n+1; k++ )
    {
      if( beta[k] < 0.0 )
        rc = 0;

      if( beta[k] > 1.0 )
        rc = 0;
    }

    if( rc == 1 )
    {
      // compute point on facet from b.c. coords
      for( int k = 0; k < n+1; k++ )
        daxpy_(&np1,&beta[k],&facet[k*(n+1)],&_i1,y1,&_i1);
    }
  }

  return rc;
}

/*
For  0 <= index <= n , select the
facet of  sigma0  that is opposite
vertex  index  and copy into  facet .
Assume memory for  facet  has already
been allocated.
Essentially, this amounts to deleting
column  index  of  sigma0 .
Also, copy corresponding residual values.
*/
static void get_facet
(
int const n,          // dimension of range ...
int const nu,         // nu >= 0
double* const sigma,  // [(n+1) x (n+2)]
double* const s_rsd,  // [n x (n+2)]; residuals for the simplex
double* const su,     // [nu x (n+2)]; residuals for the simplex
int const index,      // [0,n+1]; select a vertex
double* facet,        // [(n+1) x (n+1)]; output facet
double* f_rsd,        // [n x (n+1)]; residuals for the facet
double* fu            // [nu x (n+1)]; residuals for the facet
)
{
  int const np1 = n+1;
  int s_col; // [0,n+1]; s_col != index
  int f_col; // [0,n]
  s_col = 0;
  f_col = 0;
  while( s_col < n+2 )
  {
    if( s_col != index )
    {
      dcopy_(&np1,&sigma[s_col*(n+1)],&_i1,&facet[f_col*(n+1)],&_i1);
      if( s_rsd != 0 )
      {
        dcopy_(&n, &s_rsd[s_col*n], &_i1,&f_rsd[f_col*n], &_i1);
        dcopy_(&nu,&su[s_col*nu],   &_i1,&fu[f_col*nu],   &_i1);
      }
      f_col++;
    }
    s_col++;
  }
  assert( f_col == n+1 );
}

void codim1
(
int const n,         // dimension of range
int const nu,        // # user functions; >= 0
double* y,           // [n+1]; y = (x,\mu) point on the zero curve
double* u,           // [nu] interpolated values of user-functions
double const grain,  // size for simplices
double* cum_al,      // arc-length so-far
FILE* fp_trace,      // file for various messages
SimpconTask const task,
void (*user_func)(int const,int const,const double*,double*,double*,const void*),
const void* param    // user-defined; can be null; passed into  user_func
)
{
  assert( n >= 1 );
  // static allocation so they persist across calls ...
  static double* sigma0;  // [(n+1) x (n+2)]; initial simplex
  static double* sigma1;  // [(n+1) x (n+2)]; most recent simplex
  static double* tau;     // [(n+1) x (n+1)]; facet of sigma0
  static double* s_rsd0;  // [n x (n+2)]; residuals for  sigma0
  static double* su0;     // [nu x (n+2)]; residuals for  sigma0
  static double* s_rsd1;  // [n x (n+2)]; residuals for  sigma1
  static double* su1;     // [nu x (n+2)]; residuals for  sigma1
  static double* f_rsd;   // [n x (n+1)]; residuals for  tau
  static double* fu;      // [nu x (n+1)]; residuals for  tau
  static double* beta ;   // [n+1]; b.c. coords of point on  tau
  static double* y1a;     // [n+1]; point on a transversal facet of sigma0
  static double* y1b;     // [n+1]; point on a transversal facet of sigma0
  static double* u1a;     // [nu]; interpolated user function at y1a
  static double* u1b;     // [nu]; ditto at y1b
  static double* y_in;    // [n+1]; appx zero point on entry facet
  static double* beta_in; // [n+2];b.c. coords of  y_in
  static double* y_out;   // [n+1]; appx zero point on exit facet
  static double* y_buf;   // [n+1]; utility
  static double* rsd;     // [n]

  static int indexa;
  static int indexb;
  static int index_in;    // y_in  on facet  index_in  in sigma1 
  static int index_out;   // y_out  on facet  index_out  in  sigma1
  static int* code_list;  // [n+2]
  int n_code_list;
  static int index_a;
  static int index_b;
  int facet_select;
  int found_flag; // code from bc_include of a repeat
  static int* trans_code;
  static int index_o;
  static double* mat;     // [nxn]; used for linear solve in  transversal
  static int* ipiv;       // [n]; pivot sequence for said linear solve

  // take address of these for Fortran calls
  int const np1np2 = (n+1)*(n+2);
  int const np1np1 = (n+1)*(n+1);
  int const nunp2  = nu*(n+2);
  int const nsq  = n*n;
  int const nm1 = n-1;
  int const nnp2 = n*(n+2);
  int const nnp1 = n*(n+1);
  int const np1 = n+1;
  int k; // general purpose
  double y0_nrm;
  double grain10;
  double grain5 = 5*grain;
  double* yr;
  
  // re-entry point assuming previous call  
  switch( task )
  {
  case allocate_mem:        goto allocate_mem_task;
  case check_start_point:   goto check_start_point_task;
  case find_transverse:     goto find_transverse_task;
  case path_start_a:        goto path_start_a_task;
  case path_start_b:        goto path_start_b_task;
  case path_advance:        goto path_advance_task;
  case free_mem:            goto free_mem_task;
  default: assert(0);
  }

allocate_mem_task:
  // data structure allocations ...
  sigma0   = (double*)malloc((n+1)*(n+2)*sizeof(double));
  sigma1   = (double*)malloc((n+1)*(n+2)*sizeof(double));
  tau      = (double*)malloc((n+1)*(n+1)*sizeof(double)); // a facet
  beta     = (double*)malloc((n+1)*sizeof(double));
  s_rsd0   = (double*)malloc(n*(n+2)*sizeof(double));
  su0      = (double*)malloc(nu*(n+2)*sizeof(double));
  s_rsd1   = (double*)malloc(n*(n+2)*sizeof(double));
  su1      = (double*)malloc(nu*(n+2)*sizeof(double));
  f_rsd    = (double*)malloc(n*(n+1)*sizeof(double));
  fu       = (double*)malloc(nu*(n+1)*sizeof(double));
  rsd      = (double*)malloc(n*sizeof(double));
  y1a      = (double*)malloc((n+1)*sizeof(double));
  y1b      = (double*)malloc((n+1)*sizeof(double));
  u1a      = (double*)malloc(nu*sizeof(double));
  u1b      = (double*)malloc(nu*sizeof(double));
  y_in     = (double*)malloc((n+1)*sizeof(double));
  beta_in  = (double*)malloc((n+2)*sizeof(double));  
  y_out    = (double*)malloc((n+1)*sizeof(double));
  y_buf    = (double*)malloc((n+1)*sizeof(double));
  code_list= (int*)malloc((n+2)*sizeof(int));
  trans_code = (int*)malloc((n+3)*sizeof(int));
  mat      = (double*)malloc(n*n*sizeof(double));
  ipiv     = (int*)malloc(n*sizeof(int));
  return;

  // check starting point ...
check_start_point_task:
  user_func(n,nu,y,rsd,u,param);

  y0_nrm = dnrm2_(&n,rsd,&_i1);
  fprintf(fp_trace,"||f(y_0)|| = %e <= %e\n",y0_nrm,grain/100);

  if( y0_nrm > grain/100 )
    fprintf(fp_trace,"y_0  does not appear to be on the zero curve\n");
  else
    fprintf(fp_trace,"y_0  within tolerance\n");
  return;

find_transverse_task:
  // construct simplex centered on y_0;
  // then determine (exactly) two faces
  // transversal to the zero curve ...
  freudenthal(n,sigma0,n+1);

  // scale by  grain 
  dscal_(&np1np2,&grain,sigma0,&_i1);

  // offset to  y_0  as center
  {
    double* yr = (double*)malloc((n+2)*sizeof(double));
    for( int k = 0; k < n+2; k++ )
    {
      yr[k] = 0.1 * grain * (0.5 - (double)rand()/(double)RAND_MAX);
    }

    for( int col = 0; col < n+2; col++ )
    {
      daxpy_(&np1,&_d1,y ,&_i1,&sigma0[col*(n+1)],&_i1);
      daxpy_(&np1,&_d1,yr,&_i1,&sigma0[col*(n+1)],&_i1);
    }
    free(yr);
  }
  
  // label the simplex ...
  for( int k = 0; k < n+2; k++ )
    user_func(n,nu,&sigma0[k*(n+1)],&s_rsd0[k*n],&su0[k*nu],param);

  show("rsd\n",s_rsd0,n,n+2,n,"\n",fp_trace);

  /*=======================
  Find two facets of  sigma0  transversal to the zero curve
  =========================*/
  k = 0;
  while(k < n+2)
  {
    get_facet(n,nu,sigma0,s_rsd0,su0,k,tau,f_rsd,fu);
    trans_code[k] = transversal(n,tau,f_rsd,y1a,beta,mat,ipiv,param);
    k++;
  }
  trans_code[n+2] = 1; // sentinel 

  k = 0;
  while( trans_code[k] == 0 )
    k++;

  if( k == n+2 )
  {
    fprintf(fp_trace,"unable to find first transversal facet\n");
    goto free_return;
  }

  index_a = k;

  get_facet(n,nu,sigma0,s_rsd0,su0,index_a,tau,f_rsd,fu);
  (void)transversal(n,tau,f_rsd,y1a,beta,mat,ipiv,param);

  // compute u1a using  fu  and  beta
  dcopy_(&nu,&_d0,&_i0,u1a,&_i1);
  for( int k = 0; k < n+1; k++ )
    daxpy_(&nu,&beta[k],&fu[k*nu],&_i1,u1a,&_i1);

  show("y1a\n",y1a,1,n+1,1,"\n",fp_trace);
  show("u1a",u1a,1,nu,1,"\n",fp_trace);

  user_func(n,nu,y1a,rsd,u,param);
  fprintf(fp_trace,"\t||%e||\n",dnrm2_(&n,rsd,&_i1));

  k++;
  while( trans_code[k] == 0 )
  {
    k++;
  }

  if( k > n+1 )
  {
    fprintf(fp_trace,"unable to find second transversal facet\n");
    goto free_return;
  }

  index_b = k;

  get_facet(n,nu,sigma0,s_rsd0,su0,index_b,tau,f_rsd,fu);
  (void)transversal(n,tau,f_rsd,y1b,beta,mat,ipiv,param);
  // compute u1b using  fu  and  beta
  dcopy_(&nu,&_d0,&_i0,u1b,&_i1);
  for( int k = 0; k < n+1; k++ )
    daxpy_(&nu,&beta[k],&fu[k*nu],&_i1,u1b,&_i1);

  show("y1b\n",y1b,1,n+1,1,"\n",fp_trace);
  show("u1b",u1b,1,nu,1,"\n",fp_trace);
  user_func(n,nu,y1b,rsd,u,param);
  fprintf(fp_trace,"\t||%e||\n",dnrm2_(&n,rsd,&_i1));

  k++;
  while( trans_code[k] == 0 )
  {
    k++;
  }

  if( k != n+2 )
  {
    fprintf(fp_trace,"more than two transversal facets\n");
    goto free_return;
  }

  *cum_al = 0;
  // normal return; user must re-enter with  task == path_start_task
  return;

path_start_a_task:
  facet_select = index_a;
  goto _path_start_task;
path_start_b_task:
  facet_select = index_b;
_path_start_task: 
  // user has set  facet_select  to one of index_a or index_b
  if( facet_select == index_a )
  {
    dcopy_(&np1,y1a,&_i1,y_in,&_i1);
    dcopy_(&nu,u1a,&_i1,u,&_i1);
  }else if( facet_select == index_b )
  {
    dcopy_(&np1,y1b,&_i1,y_in,&_i1);
    dcopy_(&nu,u1b,&_i1,u,&_i1);
  }else
  {
    fprintf(fp_trace,"bad re-entry call; facet_select not correct\n");
    goto free_return;
  }

  // start arc-length computation ...
  dcopy_(&np1,y,&_i1,y_buf,&_i1);
  daxpy_(&np1,&_dn1,y_in,&_i1,y_buf,&_i1);
  *cum_al = dnrm2_(&np1,y_buf,&_i1);
  dcopy_(&np1,y_in,&_i1,y,&_i1);

  // take first step to  sigma1  and update vertex label
  pivot(n,nu,facet_select,sigma0,sigma1,s_rsd0,s_rsd1,su0,su1);
  user_func(n,nu,&sigma1[facet_select*(n+1)],&s_rsd1[facet_select*n],&su1[facet_select*nu],param);
  /*

                         sigma1
                  /\------------------- /
                 /  \                  /
                /    \                /
               /      \              / 
              /        \            /
   ...   y_p,idx_p   y_c,idx_c  y_a,idx_a
            /            \        /
           /              \      /
          /                \    /
         /                  \  /
        /--------------------\/
               sigma0

  */

  index_o = facet_select;
  return;
  
path_advance_task:
  // sigma1  is the most recent simplex at the end of the current path
  // x_in  is an approximate zero point on facet  index_o  of  sigma1
  // find (unique?) transversal face of  sigma1  different from  index_o

  trans_code[n+2] = 1; // sentinel
  k = 0;

  // brute-force test of all facets of  sigma1
  while( k < n+2 )
  {
    if( k != index_o )
    {
      get_facet(n,nu,sigma1,s_rsd1,su1,k,tau,f_rsd,fu);
      trans_code[k] = transversal(n,tau,f_rsd,y_out,beta,mat,ipiv,param);
    }else
      trans_code[k] = 0;
    k++;
  }

  index_o = 0;
  while( trans_code[index_o] == 0 )
  {
    index_o++;
  }

  if( index_o == n+2 )
  {
    fprintf(fp_trace,"exit facet not found in sigma1\n");
    goto free_return;
  }

  assert( index_o <= n+1 );
  k = index_o + 1;
  while( k < n+2 )
  {
    if( trans_code[k] == 1 )
    {
      fprintf(fp_trace,"simplex  sigma1  has more than one exit facet (!)\n");
      goto free_return;
    }

    k++;
  }
  assert( k == n+2 );

  // now, compute exit point  y_out ...
  get_facet(n,nu,sigma1,s_rsd1,su1,index_o,tau,f_rsd,fu);
  (void)transversal(n,tau,f_rsd,y_out,beta,mat,ipiv,param);

  dcopy_(&nu,&_d0,&_i0,u,&_i1);
  for( int k = 0; k < n+1; k++ )
    daxpy_(&nu,&beta[k],&fu[k*nu],&_i1,u,&_i1);

  // update cumulative arc-length ...
  dcopy_(&np1,y_in,&_i1,y_buf,&_i1);
  daxpy_(&np1,&_dn1,y_out,&_i1,y_buf,&_i1);
  *cum_al += dnrm2_(&np1,y_buf,&_i1);

  // output point
  dcopy_(&np1,y_out,&_i1,y,&_i1);

  // copy  sigma1  into  sigma0  then pivot
  // to create new  sigma1
  dcopy_(&np1np2,sigma1,&_i1,sigma0,&_i1);
  dcopy_(&nnp2,s_rsd1,&_i1,s_rsd0,&_i1);
  dcopy_(&nunp2,su1,&_i1,su0,&_i1);

  pivot(n,nu,index_o,sigma0,sigma1,s_rsd0,s_rsd1,su0,su1);
  // update one new vertex label
  user_func(n,nu,&sigma1[index_o*(n+1)],&s_rsd1[index_o*n],&su1[index_o*nu],param);
  
  // sigma1  is now at the end of the path so far ...
  // index_o  identifies the exit facet with point  y_out
  dcopy_(&np1,y_out,&_i1,y_in,&_i1);
  return;

free_return: free_mem_task:
  free(sigma0);
  free(sigma1);
  free(tau);
  free(beta);
  free(s_rsd0);
  free(su0);
  free(s_rsd1);
  free(su1);
  free(f_rsd);
  free(fu);
  free(rsd);
  free(y1a);
  free(y1b);
  free(u1a);
  free(u1b);
  free(y_in);
  free(beta_in);
  free(y_out);
  free(y_buf);
  free(code_list);
  free(trans_code);
  free(mat);
  free(ipiv);
}

#if 1
int main()
{
  int const nr = 2; // mapping from R^{nr+1} -> R^nr
  int const nu = 0; // # user functions
  double v[nr+1];
  v[0] = 0.459619; // point on the zero curve 
  v[1] = 0.954594;
  v[2] = 1.122073;
  
  double u[1];
  double const grain = 0.01;
  double cum_al;
  const void* param = (void*)0;
  codim1(nr,nu,v,u,grain,&cum_al,stdout,allocate_mem,helix,param);
  codim1(nr,nu,v,u,grain,&cum_al,stdout,check_start_point,helix,param);

  FILE* fp = fopen("helix.m","w");
  fprintf(fp,"xx=[\n");
  fprintf(fp,"%f %f %f\n",v[0],v[1],v[2]);
  codim1(nr,nu,v,u,grain,&cum_al,stdout,find_transverse,helix,param);
  codim1(nr,nu,v,u,grain,&cum_al,stdout,path_start_a,helix,param);
  fprintf(fp,"%f %f %f\n",v[0],v[1],v[2]);

  while(cum_al < 25 )
  {
    codim1(nr,nu,v,u,grain,&cum_al,stdout,path_advance,helix,param);
    fprintf(fp,"%f %f %f\n",v[0],v[1],v[2]);
  }
  fprintf(fp,"];plot3(xx(:,1),xx(:,2),xx(:,3));\n");

  fclose(fp);
  codim1(nr,nu,v,u,grain,&cum_al,stdout,free_mem,helix,param);
}
#endif
