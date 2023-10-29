#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <cfloat>
#include "codim2.h"
// LICENSE: CPL Version 1.0; see LICENSE file ...
/*
Compile with
    g++ codim2.cc -lblas -llapack -lm -o codim2
The example code at the end of the file does a torus
as a 2d surface in R^3. You need an  stl  viewer
such as  fstl (among many others).
Other interesting shapes include a sphere, a Klein bottle
and a right-angle wall ("hook"); The hook example
requires a boundary function (see below).

Let  f()  be a continuous mapping from R^nd into R^nr 
nd >= 3, nr >= 1 with  CoDim == nd - nr = 2 .
The set ("level-set") of points  y \in R^nd  such that  f(y) = 0 \in R^nr ,
and passing through a user-specified starting point  y_0 \in R^nd
generically forms a smooth two-dimensional manifold 
(a surface) in  R^nd and the surface is described as having 
"co-dimension 2" in the  ambient space  of dimension nd .
Given such a starting point  y_0  such that  f(y_0) = 0,
this code attempts to compute an approximation
to said suface starting at  y_0 as a connected collection of
nd-dimensional simplices (e.g., tetrahedra for the case of
f:R^3 -> R^1). The surface of enumerated, connected
simplices is an approximation to the actual surface is
the sense that each of the enumerated simplices intersects
the zero set (surface). The fidelity of the approximation improves
as the granularity of the enumerated simplices is reduced
at the cost of more compute time.Note that the compute time
goes up _quickly_ with a decrease in the  grain  parameter,
particularly for  nd  larger than 3!
Each of these simplices is cut by the manifold (more precisely,
a local linear approximation to the manifold) in a cross-section.
These cross-sections are reported to the user and could be, e.g.,
written to a file for some sort of later graphical rendering
(e.g., using  fstl ).

The code incorporates some features that make it
particularly suitable for {\it experimental} continuation in which
the evaluation of the residual ( f  above) is performed by
a piece of test equipment rather than a numerical
computation. Of course, numerical evaluation of the
residual can be done as well and some examples of
residual computation by code are provided as demos.

Point 1:
This code does NOT require partial derivatives
of  f() ! There are published codes for surface generation
but these codes typically require very accurate derivative
information.  In an experimental setting, the only practical
way to get such derivatives is by finite-differencing which
is slow and tends to accentuate noise. Even in a purely
computational setting (i.e.  f() above is computed, not
measured) there are situations -- such as retrofitting a
surface generation code to a legacy simulator -- in which
residual evaluation only without derivatives is advantageous.

Point 2:
The code explores the ambient space ONLY in the vicinity
of the zero surface (the manifold) starting in the vicinity 
of a user-specified "seed" point. It operates in an incremental fashion,
working outwards from recently-generated nearby simplices.
This property can be very important in an experimental context
in which an attempt to evaluate the residual  f()  above
too far off the zero surface might results in an experimental
device "latching" into a qualitatively different state or even
suffering physical damage.

Be advised that when performing such experimental or hardware-driven
sweeps, one should not expect ANYWHERE near the precision or
dynamic range of even single-precision computations, let alone
double precision. Double-precision computation usually supports
around 16 significant digits which -- in electronic language --
is something like 320 (!) dB. A typical experimental continuation
might provide -- on a good day -- 80 dB, which means something like
4 (!) significant digits. In addition, measured data will
always contain some noise and not be entirely re-producible.
The theme of this code is robustness over anything else.

It is assumed that a single residual evaluation (measurement)
is much, much slower than any reasonable matrix manipulations
(e.g., solving  ~nd  linear systems of dimension  ~nd ).
For moderate  nd  as described above (nd < 20),
these sort of computations are extremely fast on a modern computer.
The cost of one residual measurement will typically far outweigh
any such numerical manipulations, once residual values are available.

For the case of co-dimension 1 -- i.e., f:R^{n+1} -> R^n --
a termination criterion is specified easily by giving a
desired arc-length. The code starts a curve (not a surface)
at  y_0  and enumerates connected segments (moving in
a user-specified initial direction) until a user-specified
arc-length is achieved or exceeded by one segment.

Another possibility, even in the case of co-dimension 1, is
_cycling_. That is to say, the code enumerates an n-dimensional simplex
it has previously enumerated. The obvious example would be
a circle in the ambient space whose perimeter is shorter than the
user-specified arc-length. However, it is also possible for the
code to cycle as an error because the measurement of the residual
function contains noise or is otherwise not sufficiently reproducible.
Such a case would probably be treated as a failure and the user
would need to re-think the experimental setup to reduce noise or
increase the resolution of the measurement.

For co-dimension 2 the situation is a bit more complicated.
There are certainly applications of the code where a compact
connected surface -- such as a sphere or torus -- is the desired result.
In such a case, the code will enumerate connected simplices
starting around  y_0  until it runs out of "new" simplices.
In other words, every attempt to expand the frontier of
the surface steps onto a simplex that has previously been enumerated
and the code stops.

Another possibility is that the user prescribes a so-called
"boundary function" which is a mapping from R^nd into R^{>=0} .
If all vertices of a candidate simplex evaluate to 0 with the boundary
function, then the simplex is accepted. Otherwise, it is not enumerated.
A slightly more efficient criterion is to test only if the bary-center
of the simplex is inside the boundary. This mostly works as
expected, but can result in "raggedy edges" of the manifold.

It is better geometrically if the code can manipulate
simplices in which each dimension is more-or-less
on the same scale!!! This is the responsibility of the calling
program. For example, if coordiates are a mixture of
volages and currents, it might be advisable for the
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
*/

static int const _i0 = 0;
static int const _i1 = 1;
static double const _d0 = 0;
static double const _d1 = 1;
static double const _dn1 = -1;

// external routines, mostly Fortran (BLAS and Lapack)
extern "C" double dnrm2_(...);
extern "C" double ddot_(...);
extern "C" void dscal_(...);
extern "C" void dcopy_(...);
extern "C" void dswap_(...);
extern "C" void daxpy_(...);
extern "C" void dgetrs_(...);
extern "C" void dgetrf_(...);
extern "C" void dgesvd_(...);

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
*/

// two walls at right angles
static void hook
(
int const nd,
int const nu,
const double* y, // [nd]
double* rsd,     // [nr]
double* usf,     // [nu]
const void* param
)
{
  assert( nd == 3 );
  assert( nu == 0 );

  if( y[0] < 1 )
    rsd[0] = y[2];
  else
    rsd[0] = fabs(y[0]-1)-0.01;// (1+0.5*sin(2*M_PI*fabs(y[2]))); //y[2] - pow(y[0]-1,8);
}

static void hook_boundary
(
int const nd,
const double* y, // [nd]
double *score,   // accumulative
const void* param
)
{
  assert( nd == 3 );
  if( y[0] < 0 ) { *score = 1; return;}
  if( y[0] > 3 ) { *score = 1; return;}
  if( y[1] < 0 ) { *score = 1; return;}
  if( y[1] > 2 ) { *score = 1; return;}
  if( y[2] < 0 ) { *score = 1; return;}
  if( y[2] > 2 ) { *score = 1; return;}

  *score = 0;
}

// diagonal plane going through the origin in R^3
static void plane3
(
int const nd,
int const ufe,
const double* y, // [nd]
double* rsd,     // [nr]
double* usf,     // [nu]
const void* param
)
{
  assert( nd == 3 );
  rsd[0] = y[0] + 1.27*y[1];
  //usf[0] = y[0];
}

static void plane3_boundary
(
int const nd,
const double* y, // [nd]
double *score,   // accumulative
const void* param
)
{
  assert( nd == 3 );
  double const dist = max(fabs(y[1]),fabs(y[2]));

  if( dist <= 2 )
    *score += 0;
  else
    *score += exp(dist-2) - 1;
}

static void plane4
(
int const nd,
int const ufe,
const double* y, // [nd]
double* rsd,     // [nr]
double* usf,     // [nu]
const void* param
)
{
  assert( nd == 4 );
  rsd[0] = y[0]+y[1];
  rsd[1] = pow(y[2],2)+2*y[3];
  //usf[0] = y[0];//pow(y[1],2)+pow(y[2],2);
}

static void plane4_boundary
(
int const nd,
const double* y, // [nd]
double *score,   // accumulative
const void* param
)
{
  assert( nd == 4 );
  double const radius = 2;
  // box boundary ...
  double const dist = max( max(max(fabs(y[0]),fabs(y[1])),fabs(y[2])),fabs(y[3]));

  if( dist < radius )
    *score += 0;
  else
    *score += exp(dist-radius) - 1;
}

static void plane5
(
int const nd,
int const ufe,
const double* y, // [nd]
double* rsd,     // [nr]
double* usf,     // [nu]
const void* param
)
{
  assert( nd == 5 );
  rsd[0] = y[0]+y[1];
  rsd[1] = pow(y[2],2)+2*y[3];
  rsd[2] = pow(y[4],3) + y[0];
  //usf[0] = y[0];//pow(y[1],2)+pow(y[2],2);
}

static void plane5_boundary
(
int const nd,
const double* y, // [nd]
double *score,   // accumulative
const void* param
)
{
  assert( nd == 5 );
  double const radius = 2;
  // box boundary ...
  double const dist = max(fabs(y[4]),max( max(max(fabs(y[0]),fabs(y[1])),fabs(y[2])),fabs(y[3])));

  if( dist < radius )
    *score += 0;
  else
    *score += exp(dist-radius) - 1;
}

/*
This is a pseudo-realistic example of an electronic circuit with
two continuation parameters. The goal is to plot the current through
the driving source as a response w.r.t. the two parameters, the value
of the voltage source and the value of one of the resistors.

As a SPICE netlist:
V1 99 0 \mu_0
R1 99 1 1k+2k*\mu_1
R2 1 2 1k
D1 2 gnd DefaultDiodeModel
.end

The value of the voltage source is the first continuation parameter \mu_0
in the range [0,5] and the first resistor has value 1k + 1k*\mu_1 with \mu_1
in the range [0,1]. The _response_ is the current through the voltage source.
We would like to plot the response w.r.t. (\mu_0,\mu_1).
The boundary function imposes 0 <= \mu_0 <= 5 and 0 <= \mu_1 <= 1.
The usual sharp nonlinearity of the diode is somewhat softened by the
series resistor R2.
The response surface has a "knee" as the diode turns on,but then
has a concave ramp because the series resistance is also changing.

The mapping R^4 -> R^2 is:
(x1,x2,\mu_0,\mu_1) |--> (i_1,i_2) = (0,0)
where (i_1,i_2) represent Kirchoff's current laws at the
two internal nodes.

Note that in order to render the response on the same geometric
scale as the two continuation parameters, it is necessary to
scale the response. This is typical when mixing voltages and currents.
*/

static void rrd
(
int const nd,
int const nu,
const double* y, // [nd]
double* rsd,     // [nr]
double* usf,     // [nu]
const void* param
)
{
  assert( nd == 4 );
  assert( nu == 1 );
  double const mu0 = y[2];
  double const mu1 = y[3];
  double const R1 = 1e3 + mu1*2e3;
  double const R2 = 1e3;
  // diode equation ...
  double const D = 1e-15*(exp(40*y[1])-1);

  // impose Kirchoff's current law at two nodes
  rsd[0] = (mu0-y[0])/R1 - (y[0]-y[1])/R2;
  rsd[1] = (y[0]-y[1])/R2 - D;

  // response is current through source ...
  usf[0] = 2e3*(y[0]-y[1])/R2;
}

static void rrd_boundary
(
int const nd,
const double* y, // [nd]
double *score,   // accumulative
const void* param
)
{
  assert( nd == 4 );
  double const mu0 = y[2];
  double const mu1 = y[3];

  if( mu0 < 0 ) { *score = 1; return;};
  if( mu0 > 5 ) { *score = 1; return;};
  if( mu1 < 0 ) { *score = 1; return;};
  if( mu1 > 1 ) { *score = 1; return;};
  *score = 0;
  return;
}

//------------ compact examples ------------------
// set BoundryFunction = 0;
// unit sphere
static void u_sphere
(
int const nd,
int const ufe,
const double* y, // [nd]
double* rsd,     // [nr]
double* usf,
const void* param
)
{
  assert( nd == 3 );
  rsd[0] = pow(y[0],2) + pow(y[1],2) + pow(y[2],2) - 1;
  usf[0] = y[0];
}

static void torus
(
int const nd,
int const ufe,
const double* y, // [nd]
double* rsd,     // [nr]
double* usf,     // [nu]
const void* param
)
{
  assert( nd == 3 );
  // starting point is (R+r,0,0)
  double const r = 0.5;
  double const R = 2;
  rsd[0] = pow(R - sqrt(pow(y[0],2)+pow(y[1],2)),2) + pow(y[2],2) - pow(r,2);
  usf[0] = y[0];
}

/*
Immersion of a Klein bottle (!) in R^3
A possible starting point is (1,0,0)
Makes a cool picture ...
*/
static void klein
(
int const nd,
int const ufe,
const double* y, // [nd]
double* rsd,     // [nr]
double* usf,     // [nu]
const void* param
)
{
  // from Wolfram ...
  assert( nd == 3 );
  double const fact_a = (pow(y[0],2)+pow(y[1],2)+pow(y[2],2)+2*y[1]-1);
  double const fact_b = (pow(y[0],2)+pow(y[1],2)+pow(y[2],2)-2*y[1]-1);
  double const fact_c = pow(fact_b,2)-8*pow(y[2],2);
  rsd[0] = fact_a * fact_c + 16*y[0]*y[2]*fact_b;
  usf[0] = y[0];
}

void bary_center
(
int const nd,
const double* verts, //[nd*(nd+1)] verts of sigma
double* bc           //[nd] return bary-center of sigma
)
{
  double const rd = 1.0/(double)(nd+1); // avg by # vertices ...
  dcopy_(&nd,&_d0,&_i0,bc,&_i1);
  for( int k = 0; k < nd+1; k++ )
    daxpy_(&nd,&_d1,&verts[k*nd],&_i1,bc,&_i1);
  dscal_(&nd,&rd,bc,&_i1);
}

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
        fprintf(fp_o,"%+8.4f ",dat[c*lda+r]);
    fprintf(fp_o,";\n");
  }
  for( int c = 0; c < ncol; c++ )
      fprintf(fp_o,"%+8.4f ",dat[c*lda+(nrow-1)]);
  fprintf(fp_o,";%s",suffix);
  fflush(fp_o);
}

static void i_show
(
const char* title,
int* dat,
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
        fprintf(fp_o,"%d ",dat[c*lda+r]);
    fprintf(fp_o,"\n");
  }
  for( int c = 0; c < ncol; c++ )
      fprintf(fp_o,"%d ",dat[c*lda+(nrow-1)]);
  fprintf(fp_o,"%s",suffix);
  fflush(fp_o);
}

//------------ hash functions ----------------
size_t VertLabHash::
operator()(VertLab* const& x_p) const
{
  long int prime_prod = 1;
  int const nd = x_p->root_p->nd;

  for( int k = 0; k < nd; k++ )
  {
    prime_prod *= (iabs(x_p->vc[k])+1)*x_p->root_p->prime_tab[k];
  }
  return std::hash<int>()(prime_prod);
};

bool VertLabEqual::
operator()(VertLab*const &x, VertLab*const &y) const
{
  int const nd = x->root_p->nd;
  for( int k = 0; k < nd; k++ )
  {
    if( x->vc[k] != y->vc[k] )
      return false;
  }

  return true;
};

size_t SimplexHash::
operator()(Simplex* const& x_p) const
{
  long int prime_prod = 1;
  int const nd = x_p->root_p->nd;

  for( int k = 0; k < nd; k++ )
  {
    prime_prod *= (iabs(x_p->cnr[k])+1)*x_p->root_p->prime_tab[k];
    prime_prod *= (iabs(x_p->per[k])+1)*x_p->root_p->prime_tab[k];
  }
  return std::hash<int>()(prime_prod);
};

bool SimplexEqual::
operator()(Simplex*const &x, Simplex*const &y) const
{
  int const nd = x->root_p->nd;
  for( int k = 0; k < nd; k++ )
  {
    if( x->cnr[k] != y->cnr[k] )
      return false;

    if( x->per[k] != y->per[k] )
      return false;
  }

  return true;
};

size_t FacetHash::
operator()(Facet* const& x_p) const
{
  long int prime_prod = 1;
  int const nd = x_p->root_p->nd;
  int const nr = x_p->root_p->nr;

  for( int k = 0; k < nd*(nr+1); k++)
  {
    prime_prod *= (iabs(x_p->tau[k])+1)*x_p->root_p->prime_tab[k % nd];
  }

  return std::hash<int>()(prime_prod);
};

bool FacetEqual::
operator()(Facet*const &x_p, Facet*const &y_p) const
{
  int const nd = x_p->root_p->nd;
  int const nr = x_p->root_p->nr;

  for( int k = 0; k < nd*(nr+1); k++ )
  {
    if( x_p->tau[k] != y_p->tau[k] )
      return false;
  }

  return true;
};

VertLab::VertLab
(
CoDim2* root_p
)
:root_p(root_p)
,nd(root_p->nd)
,nr(root_p->nr)
,nu(root_p->nu)
{
  vc  = new int[nd];
  rsd = new double[nr];
  usf = new double[nu];
}

VertLab::~VertLab()
{
  delete[] vc;
  delete[] rsd;
  delete[] usf;
}

Facet::Facet
(
CoDim2* root_p,
int* inp_tau
)
:root_p(root_p)
,nd(root_p->nd)
,nr(root_p->nr)
,nu(root_p->nu)
{
  tau   = new int[nd*(nr+1)];
  coord = new double[nd];
  rsd   = new double[nr];
  usf   = new double[nu];
  memcpy(tau,inp_tau,(nd*(nr+1))*sizeof(int));
}

Facet::~Facet()
{
  delete[] tau;
  delete[] coord;
  delete[] rsd;
  delete[] usf;
}

/*
The  ConvexPoly  data structure is used to capture
the intersection of the manifold with a simplex.
*/
ConvexPoly::ConvexPoly
(
CoDim2* root_p
)
:root_p(root_p)
,n_verts(0)
{
  // see codim.h ...
  fp_list = new Facet*[root_p->nd+1];
  id_list = new int[2*(root_p->nd+1)];
}

ConvexPoly::~ConvexPoly()
{
  delete[] fp_list;
  delete[] id_list;
}

/*
This routine arranges the vertices of a convex poly
in a kind of "circular" order. Imagine a tetrahedron
in R^3 with vertices numbered 0,1,2,3.
A potential facet (in this case, an edge) is identified
by two vertices [i0,i1] -- the vertices that are _not_ included
in the chosen facet.
Suppose the manifold intersects this tetrahedron cutting
edges 0-3[1,2] (point a), 0-2[1,3] (point b), 1-2[0,3] (point c)
and 1-3[0,2] (point d). The points a-b-c-d are in circular
order and (abc) (acd) would partition the quadrilateral
into two non-overlapping triangles. Such would NOT be the
case if the vertices were ordered (abdc).

Each facet that intersects the manifold is identified
by two vertices [i0,i1], and the pairs are stored in the columns of  id_list .
This routine re-arranges  id_list  so that consecutive entries
have exactly one vertex in common. In the tetrahedron example 
above the (undesired) sequence (abdc) would result in  id_list
1 1 0 0
2 3 2 3
which fails to be consecutive (2nd to 3rd column).
Swapping the last two columns gives
1 1 0 0
2 3 3 2
which IS consecutive.
Each column of two indices must share exactly one common index with its successor.
The  _sequence()  routine is called internally before outputting
triangles in  .stl  format.
As an additional processing step, the  output_stl  routine partitions
a polygon with more than 3 vertices into non-overlapping triangles.
*/

void ConvexPoly::_sequence()
{
  if( n_verts <= 3 ) // easy out ...
    return;

  int k = 0;
  while( k < n_verts-1 )
  {
    // find entry with one common element
    int j = k+1;
    while( j < n_verts ) // must get a hit ...
    {
      // test for overlap
      if( (id_list[2*j+0] == id_list[2*k+0])
       || (id_list[2*j+1] == id_list[2*k+1]) 
       || (id_list[2*j+1] == id_list[2*k+0]) 
       || (id_list[2*j+0] == id_list[2*k+1]) )
        break;
      else
        j++;
    }
    assert( j < n_verts );
    // swap entries j and k+1
    int temp = id_list[2*(k+1)+0];
    id_list[2*(k+1)+0] = id_list[2*j+0];
    id_list[2*j+0] = temp;

    temp = id_list[2*(k+1)+1];
    id_list[2*(k+1)+1] = id_list[2*j+1];
    id_list[2*j+1] = temp;

    Facet* tp = fp_list[k+1];
    fp_list[k+1] = fp_list[j];
    fp_list[j] = tp;
    k++;
  }
}

/*
Render a projection onto 3d as a triangulated surface.
The user selects projection coords 0 <= (j0,j1,j2) < nd;
The normal vectors are dummy; this could probably be improved.
*/
void ConvexPoly::output_stl
(
int const j0,
int const j1,
int const j2,
FILE* fp_stl
)
{
  if( n_verts < 3 )
    return;

  _sequence();

  if( n_verts == 5 )
  {
    fprintf(fp_stl,"facet normal 1 0 0\nouter loop\n");
    fprintf(fp_stl,"vertex %f %f %f\n",fp_list[0]->coord[j0],fp_list[0]->coord[j1],fp_list[0]->coord[j2]);
    fprintf(fp_stl,"vertex %f %f %f\n",fp_list[1]->coord[j0],fp_list[1]->coord[j1],fp_list[1]->coord[j2]);
    fprintf(fp_stl,"vertex %f %f %f\n",fp_list[4]->coord[j0],fp_list[4]->coord[j1],fp_list[4]->coord[j2]);
    fprintf(fp_stl,"endloop\nendfacet\n");

    fprintf(fp_stl,"facet normal 1 0 0\nouter loop\n");
    fprintf(fp_stl,"vertex %f %f %f\n",fp_list[1]->coord[j0],fp_list[1]->coord[j1],fp_list[1]->coord[j2]);
    fprintf(fp_stl,"vertex %f %f %f\n",fp_list[2]->coord[j0],fp_list[2]->coord[j1],fp_list[2]->coord[j2]);
    fprintf(fp_stl,"vertex %f %f %f\n",fp_list[3]->coord[j0],fp_list[3]->coord[j1],fp_list[3]->coord[j2]);
    fprintf(fp_stl,"endloop\nendfacet\n");

    fprintf(fp_stl,"facet normal 1 0 0\nouter loop\n");
    fprintf(fp_stl,"vertex %f %f %f\n",fp_list[1]->coord[j0],fp_list[1]->coord[j1],fp_list[1]->coord[j2]);
    fprintf(fp_stl,"vertex %f %f %f\n",fp_list[3]->coord[j0],fp_list[3]->coord[j1],fp_list[3]->coord[j2]);
    fprintf(fp_stl,"vertex %f %f %f\n",fp_list[4]->coord[j0],fp_list[4]->coord[j1],fp_list[4]->coord[j2]);
    fprintf(fp_stl,"endloop\nendfacet\n");

    fflush(fp_stl);
    return;
  }  

  int bnd = n_verts-1;
  do
  {
    fprintf(fp_stl,"facet normal 1 0 0\nouter loop\n");
    fprintf(fp_stl,"vertex %f %f %f\n",fp_list[0]->coord[j0],fp_list[0]->coord[j1],fp_list[0]->coord[j2]);
    fprintf(fp_stl,"vertex %f %f %f\n",fp_list[1]->coord[j0],fp_list[1]->coord[j1],fp_list[1]->coord[j2]);
    fprintf(fp_stl,"vertex %f %f %f\n",fp_list[2]->coord[j0],fp_list[2]->coord[j1],fp_list[2]->coord[j2]);
    fprintf(fp_stl,"endloop\nendfacet\n");

    Facet* temp = fp_list[1];
    fp_list[1] = fp_list[bnd];
    fp_list[bnd] = temp;

    bnd--;
  } while( bnd >= 2 );
  fflush(fp_stl);
};

void ConvexPoly::response_stl
(
int const usel, // [0,nu-1]
FILE* fp_stl
)
{
  int const nu = root_p->nu;
  int const nd = root_p->nd;
  
  assert( 0 <= usel );
  assert( usel < nu );
  if( n_verts < 3 )
    return;
  _sequence();
  
  if( n_verts == 5 )
  {
    fprintf(fp_stl,"facet normal 1 0 0\nouter loop\n");
    fprintf(fp_stl,"vertex %f %f %f\n",fp_list[0]->coord[nd-2],fp_list[0]->coord[nd-1],fp_list[0]->usf[usel]);
    fprintf(fp_stl,"vertex %f %f %f\n",fp_list[1]->coord[nd-2],fp_list[1]->coord[nd-1],fp_list[1]->usf[usel]);
    fprintf(fp_stl,"vertex %f %f %f\n",fp_list[4]->coord[nd-2],fp_list[4]->coord[nd-1],fp_list[4]->usf[usel]);
    fprintf(fp_stl,"endloop\nendfacet\n");

    fprintf(fp_stl,"facet normal 1 0 0\nouter loop\n");
    fprintf(fp_stl,"vertex %f %f %f\n",fp_list[1]->coord[nd-2],fp_list[1]->coord[nd-1],fp_list[1]->usf[usel]);
    fprintf(fp_stl,"vertex %f %f %f\n",fp_list[2]->coord[nd-2],fp_list[2]->coord[nd-1],fp_list[2]->usf[usel]);
    fprintf(fp_stl,"vertex %f %f %f\n",fp_list[3]->coord[nd-2],fp_list[3]->coord[nd-1],fp_list[3]->usf[usel]);
    fprintf(fp_stl,"endloop\nendfacet\n");

    fprintf(fp_stl,"facet normal 1 0 0\nouter loop\n");
    fprintf(fp_stl,"vertex %f %f %f\n",fp_list[1]->coord[nd-2],fp_list[1]->coord[nd-1],fp_list[1]->usf[usel]);
    fprintf(fp_stl,"vertex %f %f %f\n",fp_list[3]->coord[nd-2],fp_list[3]->coord[nd-1],fp_list[3]->usf[usel]);
    fprintf(fp_stl,"vertex %f %f %f\n",fp_list[4]->coord[nd-2],fp_list[4]->coord[nd-1],fp_list[4]->usf[usel]);
    fprintf(fp_stl,"endloop\nendfacet\n");

    fflush(fp_stl);
    return;
  }  

  int bnd = n_verts-1;
  do
  {
    fprintf(fp_stl,"facet normal 1 0 0\nouter loop\n");
    fprintf(fp_stl,"vertex %f %f %f\n",fp_list[0]->coord[nd-2],fp_list[0]->coord[nd-1],fp_list[0]->usf[usel]);
    fprintf(fp_stl,"vertex %f %f %f\n",fp_list[1]->coord[nd-2],fp_list[1]->coord[nd-1],fp_list[1]->usf[usel]);
    fprintf(fp_stl,"vertex %f %f %f\n",fp_list[2]->coord[nd-2],fp_list[2]->coord[nd-1],fp_list[2]->usf[usel]);
    fprintf(fp_stl,"endloop\nendfacet\n");

    Facet* temp = fp_list[1];
    fp_list[1] = fp_list[bnd];
    fp_list[bnd] = temp;

    bnd--;
  } while( bnd >= 2 );
  fflush(fp_stl);
};

Simplex::Simplex
(
CoDim2* root_p
)
:root_p(root_p)
,nd(root_p->nd)
,nr(root_p->nr)
,nu(root_p->nu)
,vcs_dirty(1)
{
  cnr = new int[nd];
  per = new int[nd];
  vcs = new int[nd*(nd+1)];
  serial = root_p->serial;
  root_p->serial++;
}

Simplex::~Simplex()
{
  delete[] cnr;
  delete[] per;
  delete[] vcs;
}

static void rotate_right
(
int const m,
int* a // [m]
)
{
  assert( m >= 1 );
  int const temp = a[m-1];
  for( int k = m-1; k > 0; k-- )
    a[k] = a[k-1];
  a[0] = temp;
}

static void rotate_left
(
int const m,
int* a // [m]
)
{
  assert( m >= 1 );
  int const temp = a[0];
  for( int k = 0; k < m-1; k++ )
    a[k] = a[k+1];
  a[m-1] = temp;
}

static void swap
(
int const m,
int const i0,
int const i1,
int* a
)
{
  assert( 0 <= i0 );
  assert( i0 < m );
  assert( 0 <= i1 );
  assert( i1 < m );
  int const temp = a[i0];
  a[i0] = a[i1];
  a[i1] = temp;
}

// apply recursion (2) from Henderson
void Simplex::update_vcs()
{
  int const nd = root_p->nd;
  if( vcs_dirty == 0 )
    return;

  for( int k = 0; k < nd; k++ )
    vcs[0*nd+k] = cnr[k];

  for( int index = 1; index < nd+1; index++ )
  {
    memcpy(&vcs[index*nd],&vcs[(index-1)*nd],nd*sizeof(int));
    vcs[index*nd + per[index-1]] += 1;
  }

  vcs_dirty = 0;
};

// compute "physical" vertices from integer grid
void Simplex::phys_v
(
double* vertices // [nd x (nd+1)]
)
{
  update_vcs();

  for( int k = 0; k < nd+1; k++ )
  {
    for( int j = 0; j < nd; j++ )
      vertices[k*nd+j] = (double)vcs[k*nd+j]*root_p->grain + root_p->offset[j];
  }
}

void Simplex::phys_v
(
const int* tau,  // [nd x (nr+1)]
double* vertices // [nd x (nr+1)]
)
{
  for( int k = 0; k < nr+1; k++ )
  {
    for( int j = 0; j < nd; j++ )
      vertices[k*nd+j] = (double)tau[k*nd+j]*root_p->grain + root_p->offset[j];
  }
}

/*
Pivot  this  around vertex  index  ;
create new  simplex  and point to it.
Specify point of evaluation for new vertex and
where to load the residual.
*/
void Simplex::pivot
(
int const p_index,
Simplex* *o_neigh_p, // local allocation of a new Simplex
int *new_v           // column of neigh_p->vcs which is new
)
{
  assert( 0 <= p_index );
  assert( p_index < root_p->nd+1 );

  Simplex* neigh_p = new Simplex(root_p);
  *o_neigh_p = neigh_p;
  int v_index;

  memcpy(neigh_p->cnr,cnr,nd*sizeof(int));
  memcpy(neigh_p->per,per,nd*sizeof(int));

  if( p_index == 0 )
  {
    neigh_p->cnr[per[0]]++;
    rotate_left(nd,neigh_p->per);
    *new_v = nd;
  }else if( p_index == nd )
  {
    neigh_p->cnr[per[nd-1]]--;
    rotate_right(nd,neigh_p->per);
    *new_v = 0;
  }else if( (0 < p_index) && (p_index < nd) )
  {
    swap(nd,p_index-1,p_index,neigh_p->per);
    *new_v = p_index;
  }else assert(0);
}

/*
A Facet of a Simplex  (itself a simplex in R^nr)  is identified by a pair
of vertices of the simplex.
*/
void Simplex::get_facet
(
int const index0,
int const index1,
int* tau         // [nd x (nr+1)]
)
{
  int s_col; // [0,nd]; s_col != index0,index1
  int f_col; // [0,nr]
  assert( 0 <= index0 );
  assert( index0 <= nd );
  assert( 0 <= index1 );
  assert( index1 <= nd );
  assert( index0 != index1 );
  update_vcs();

  s_col = 0;
  f_col = 0;
  while( s_col < nd+1 )
  {
    if( (s_col != index0) && (s_col != index1) )
    {
      memcpy(&tau[f_col*nd],&vcs[s_col*nd],nd*sizeof(int));
      f_col++;
    }
    s_col++;
  }
  assert( f_col == nr+1 ); // should be this many vertices in the facet ...
}

void Simplex::show()
{
  for( int k = 0; k < nd; k++ )
    fprintf(stdout,"%d ",cnr[k]);
  fprintf(stdout,"\n");

  for( int k = 0; k < nd; k++ )
    fprintf(stdout,"%d ",per[k]);
  fprintf(stdout,"\n\n");

  ::i_show("vcs\n",vcs,nd,nd+1,nd,"\n",stdout);
}

/*
Determine if facet [tau] intersects manifold
If so, allocate a  Facet  structure.
Return 0 if no intersection.
It is the responsibility of the calling code to record
the new  Facet ...
*/
Facet* Simplex::intersection_check
(
int const i0,
int const i1,
int* tau // [nd x (nr+1])]
)
{
  double* rsd    = new double[nr*(nr+1)];
  double* usf    = new double[nu*(nr+1)];
  double* A      = new double[nr*nr];
  int* ipiv      = new int[nr];
  int info;
  int const nrp1 = nr+1;

  update_vcs();
  // load  rsd  by extracting the residual for
  // each vertex of the facet
  for( int col = 0; col < nr+1; col++ )
  {
    VertLab label_srch(root_p);
    memcpy(label_srch.vc,&tau[col*nd],nd*sizeof(int));
    // lookup ...
    auto it = root_p->vert_lab_tab.find(&label_srch);
    assert( it != root_p->vert_lab_tab.end());
    memcpy(&rsd[col*nr],(*it)->rsd,nr*sizeof(double));
    memcpy(&usf[col*nu],(*it)->usf,nu*sizeof(double));
  }

  // form matrix ...
  for( int col = 0; col < nr; col++ )
  {
    dcopy_(&nr,&rsd[col*nr],&_i1,&A[col*nr],&_i1);    
    daxpy_(&nr,&_dn1,&rsd[nr*nr],&_i1,&A[col*nr],&_i1);
  }

  // factor and test if singular ...
  dgetrf_(&nr,&nr,A,&nr,ipiv,&info);
  if( info != 0 ) // singular
  {
    delete[] rsd;
    delete[] usf;
    delete[] A;
    delete[] ipiv;
  
    return (Facet*)0;
  }

  double* beta = new double[nr+1];
  // b.c. coords ...
  beta[nr] = 1; // adjust this later ...

  // rhs for linear solve is _negative_ of last residual ...
  dcopy_(&nr,&rsd[nr*nr],&_i1,beta,&_i1);
  dscal_(&nr,&_dn1,beta,&_i1); // negate

  // solve  nr x nr linear system
  dgetrs_("N",&nr,&_i1,A,&nr,ipiv,beta,&nrp1,&info);
  assert( info == 0 );

  // adjust bary coords ...
  for( int k = 0; k < nr; k++ )
    beta[nr] -= beta[k];

  // if any exterior to [0,1] declare failure
  for( int k = 0; k < nr+1; k++ )
  {
    if( (beta[k] >= 0) && (beta[k] <= 1) ) 
      continue;

    delete[] rsd;
    delete[] usf;
    delete[] A;
    delete[] ipiv;
    delete[] beta;

    return (Facet*)0;
  }

  double* vertices = new double[nd*(nr+1)];
  phys_v(tau,vertices);
  Facet* tau_p = new Facet(root_p,tau);

  // compute appx intersection point as convex comb of vertices
  // interpolate for user functions ...
  dcopy_(&nd,&_d0,&_i0,tau_p->coord,&_i1);
  dcopy_(&nu,&_d0,&_i0,tau_p->usf,&_i1);
  dcopy_(&nr,&_d0,&_i0,tau_p->rsd,&_i1);
  for( int k = 0; k < nr+1; k++ )
  {
    daxpy_(&nd,&beta[k],&vertices[k*nd],&_i1,tau_p->coord,&_i1);
    daxpy_(&nr,&beta[k],     &rsd[k*nr],&_i1,  tau_p->rsd,&_i1);
    daxpy_(&nu,&beta[k],     &usf[k*nu],&_i1,  tau_p->usf,&_i1);
  }

  delete[] rsd;
  delete[] usf;
  delete[] A;
  delete[] ipiv;
  delete[] beta;
  delete[] vertices;

  return tau_p;
}

/*
Find the intersection between a simplex and the
local linear approx to the manifold. In general,
said intersection will be a convex polyhedron of
points on a common hyper-plane. For example, even in
R^3, a plane can intersect a tetrahedron in 3 or 4(!)
points, depending on the angle of the slicing.

This routine finds the vertices of the polyhedral
intersection and also looks for unenumerated
("new") simplices that share a common facet with
a point of this intersection.

It returns the set of such new simplices (which
might, of course, be empty) obtained by pivoting
and adds them to the hash table of enumerated simplices.
When pivoting results in a new simplex, the code
also updates the output label for the unique new vertex.
*/

void Simplex::get_cross_sect
(
ConvexPoly& cvx,
unordered_set<Simplex*,SimplexHash,SimplexEqual>& neighbors
)
{
  int const tot_comb = root_p->tot_comb;
  int* fv_tab = root_p->fv_tab;
  int* tau                 = new int[nd*(nr+1)];
  double* facet_vertices   = new double[nd*(nr+1)];
  double* simplex_vertices = new double[nd*(nd+1)];

  void* param = root_p->param;
  double const boundary_epsilon = root_p->boundary_epsilon;
  bool interp_sw = root_p->interpolate_user_functions;

  update_vcs();
  cvx.n_verts = 0;
  assert( neighbors.size() == 0 ); // start empty ...

  // search over all possible facets of the simplex
  for( int k = 0; k < tot_comb; k++ )
  {
    int const i0 = fv_tab[0*tot_comb+k];
    int const i1 = fv_tab[1*tot_comb+k];
    Facet* tau_p;
    get_facet(i0,i1,tau); // get facet by ruling OUT (i0,i1)

    // check for a repeat ...
    Facet srch(root_p,tau);
    auto it = root_p->fac_tab.find(&srch);
    // already in table, must intersect
    // but might be from a different simplex!
    if( it != root_p->fac_tab.end() )
    {
      tau_p = *it;
      assert( tau_p != (Facet*)0 );
    }else{
      tau_p = intersection_check(i0,i1,tau);
      if(tau_p != 0)
      {
        assert( root_p->fac_tab.find(tau_p) == root_p->fac_tab.end() );
        root_p->fac_tab.insert(tau_p);
      }
    }

    if(tau_p != 0) // got a hit, consider all neighboring simplices ...
    {
      assert( cvx.n_verts < nd+1 );
      cvx.fp_list[cvx.n_verts] = tau_p;
      cvx.id_list[2*cvx.n_verts+0] = i0;
      cvx.id_list[2*cvx.n_verts+1] = i1;
      cvx.n_verts++;

      // test remaining nd-1 neighboring simplices ...
      for( int j = 2; j < nd+1; j++ )
      {
        Simplex* piv;
        int new_v; // [0,nd]

        // select a simplex neighbor and check usability
        /*
        In order to be accepted, a candidate simplex must
        1. Pass the boundary check
        2. Not be previously enumerated in the finished simplices
        3. Not be a repeat in the candidate simplices
        4. If raster scanning, last two cnr coords are inside the raster region
        */
        
        pivot(root_p->fv_tab[k+j*tot_comb],&piv,&new_v);
        assert( 0 <= new_v );
        assert( new_v <= nd );

        if( (root_p->prev_enum.count(piv) == 0)&&
            (root_p->frontier.count(piv) == 0)&&
            (neighbors.count(piv)==0) )
        {
          // for efficiency, only perform boundary check
          // if candidate passes other checks ...
          piv->phys_v(simplex_vertices);

          // final test for a candidate neighboring simplex ...
          if(root_p->boundary_score(simplex_vertices) <= boundary_epsilon)
          {
            VertLab* ru_p = new VertLab(root_p);
            memcpy(ru_p->vc,&piv->vcs[new_v*nd],nd*sizeof(int));
            // test for repeat ...
            if( root_p->vert_lab_tab.count(ru_p) == 0 )
            {
              // label new vertex
              (*(root_p->ResidualFunction))(nd,nu,&simplex_vertices[new_v*nd],ru_p->rsd,ru_p->usf,root_p->param);
              root_p->vert_lab_tab.insert(ru_p);
            }else{
              delete ru_p;
            }

            neighbors.insert(piv);
          }else{
            delete piv;
          }
        }else{
          delete piv;
        }
      }
    }
  }

  delete[] tau;
  delete[] facet_vertices;
  delete[] simplex_vertices;
}

CoDim2::CoDim2
(
int const nd,
int const nu,
double const grain
)
: nd(nd)
, nr(nd-2)
, nu(nu)
, grain(grain)
, y0(new double[nd])
, offset(new double[nd])
, fv_tab(new int[((nd+1)*nd/2)*(nd+1)])
, fp_inform(stdout) // default ...
, ResidualFunction(0)
, BoundaryFunction(0)
, param((void*)0)
, prime_tab(new long[nd])
, boundary_epsilon(1e-10)
, interpolate_user_functions(false)
, max_front_size(0)
, serial(0)
{
  // first two positions: all possible combinations of nd+1 choose 2 ...
  // remaining (nd+1)-2 positions: everything else
  tot_comb = (nd+1)*nd/2; // chose(#vertices,2)
  int sel_len = 0; // row index
  for( int k = 0; k < nd+1; k++ )
    for( int j = k+1; j < nd+1; j++ )
    {
      fv_tab[sel_len+0*tot_comb] = k;
      fv_tab[sel_len+1*tot_comb] = j;
      int row_fill = 0; // column index
      for( int m = 0; m < nd+1; m++ )
      {
        if( (m != k) && (m != j) )
        {
          assert( row_fill < nd-1 );
          fv_tab[sel_len+(row_fill+2)*tot_comb] = m;
          row_fill++;
        }
      }
      assert( row_fill == nd-1 );
      sel_len++;
    }
  assert( sel_len == tot_comb );

  // possibly extend prime_tab if  nd > 7 ...
  for( int k = 0; k < nd; k++ )
    prime_tab[k] = _prime_tab[k%7];
}

CoDim2::~CoDim2()
{
  delete[] y0;
  delete[] offset;
  delete[] prime_tab;
  delete[] fv_tab;

  // delete storage for frontier simplices ...
  // (often empty)
  for( auto it = frontier.begin(); it != frontier.end(); ++it)
  {
    delete (*it);
  }

  // delete storage for enumerated simplices ...
  for( auto it = prev_enum.begin(); it != prev_enum.end(); ++it)
  {
    delete (*it);
  }

  // delete storage for VertLab
  for( auto it = vert_lab_tab.begin(); it != vert_lab_tab.end(); ++it)
  {
    delete (*it);
  }

  // delete storage for Facets
  for( auto it = fac_tab.begin(); it != fac_tab.end(); ++it)
  {
    delete (*it);
  }
}

/*
Sum score over all vertices ...
All vertices must be inside ...
*/
double CoDim2::boundary_score
(
double* vertices // [nd x (nd+1)]
) const
{
  if( BoundaryFunction == 0 )
    return 0;

  double score = 0;
  for( int k = 0; k < nd+1; k++ )
    (*BoundaryFunction)(nd,&vertices[k*nd],&score,param);

  return score;
}

static double range_exceed
(
const double lo,
const double hi,
const double x
)
{
  assert( lo < hi );
  if( (lo <= x) && (x <= hi) )
    return 0;

  if( x < lo )
    return exp(lo - x) - 1;

  if( hi < x )
    return exp(x - hi) - 1;

  assert(0);
}

void CoDim2::start
(
const double* the_y0 // [nd]
)
{
  double* eval_rsd = new double[nr];
  double* eval_usf = new double[nu];
  double* vertices = new double[nd*(nd+1)];

  if( ResidualFunction == 0 )
  {
    fprintf(fp_inform,"ResidualFunction not assigned\n");
    exit(0);
  }
  memcpy(y0,the_y0,nd*sizeof(double));

  Simplex* seed_p = new Simplex(this);

  assert( ResidualFunction != 0 );
  (*ResidualFunction)(nd,nu,y0,eval_rsd,eval_usf,param);
  if( dnrm2_(&nr,eval_rsd,&_i1) > grain / 100 )
  {
    fprintf(fp_inform,"Waring: ||f(y0)|| > grain / 100\n");
  }

  // create integer coords ...
  for( int k = 0; k < nd; k++ )
  {
    seed_p->cnr[k] = 0;
    seed_p->per[k] = k;
  }

  dcopy_(&nd,&_d0,&_i0,offset,&_i1);
  seed_p->phys_v(vertices);
  // generate first with offset zero ...
  bary_center(nd,vertices,offset);

  // move seed simplex so its b.c. is *almost* on  y0
  for( int k = 0; k < nd; k++ )
  {
    offset[k] = y0[k] - offset[k];
    offset[k] += 0.1*grain * (0.5-(double)rand()/(double)RAND_MAX);
  }
  seed_p->phys_v(vertices);

  if( (BoundaryFunction != 0)&&(boundary_score(vertices) > boundary_epsilon) )
  {
    fprintf(fp_inform,"warning: initial simplex fails boundary check\n");
  }

  // label *seed_p simplex ...
  for( int k = 0; k < nd+1; k++ )
  {
    (*ResidualFunction)(nd,nu,&vertices[k*nd],eval_rsd,eval_usf,param);
    VertLab* new_p = new VertLab(this);
    memcpy(new_p->vc,&seed_p->vcs[k*nd],nd*sizeof(int));
    dcopy_(&nr,eval_rsd,&_i1,new_p->rsd,&_i1);
    dcopy_(&nu,eval_usf,&_i1,new_p->usf,&_i1);
    vert_lab_tab.insert(new_p);
  }

  frontier.insert(seed_p);

  delete[] eval_rsd;
  delete[] eval_usf;
  delete[] vertices;
}

#if 1
/*
grab element in frontier
get its cross section
move to prev_enum, add neighbors to frontier
*/
void CoDim2::enumerate
(
ConvexPoly& cvx
)
{
  unordered_set<Simplex*,SimplexHash,SimplexEqual> neighbors;
  assert( frontier.size() > 0 );
  auto it = frontier.begin();
  Simplex* choice_p = *it;
  assert( choice_p != 0 );
  assert( neighbors.size() == 0 ); // start empty ...
  frontier.erase(choice_p);
  prev_enum.insert(choice_p);
  choice_p->get_cross_sect(cvx,neighbors);
  frontier.merge(neighbors);
}
#else
void CoDim2::enumerate
(
ConvexPoly& cvx
)
{
  unordered_set<Simplex*,SimplexHash,SimplexEqual> neighbors;
  assert( frontier.size() > 0 );
  double min_score = DBL_MAX;  
  Simplex* choice_p;
  // find choice with best score ...
  for( auto it = frontier.begin(); it != frontier.end(); ++it )
  {
    double temp_score = (double)(*it)->serial; //raster_score(*it);
    if( temp_score < min_score )
    {
      choice_p =  *it;
      min_score = temp_score;
    }
  }
  assert( choice_p != 0 );
  assert( neighbors.size() == 0 ); // start empty ...

  frontier.erase(choice_p);
  prev_enum.insert(choice_p);
  choice_p->get_cross_sect(cvx,neighbors);

  frontier.merge(neighbors);
}

#endif

void CoDim2::progress()
{
  if( (int)frontier.size() > max_front_size )
    max_front_size = (int)frontier.size();

  fprintf(fp_inform,"%10d %10ld",(int)prev_enum.size()
                   ,frontier.size());
  fprintf(fp_inform,"\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b");
}

void CoDim2::statistics()
{
  fprintf(fp_inform,"%10d %10ld\n",(int)prev_enum.size()
                   ,frontier.size());
}

#if 0
int main()
{
  int const nd = 3;
  int const nu = 0;
  double const grain = 0.05;
  double y0[nd] = {2.5,0,0};
  CoDim2 SD(nd,nu,grain);
  SD.ResidualFunction = torus;
  SD.BoundaryFunction = 0;
  SD.start(y0);
  FILE* fp = fopen("torus.stl","w");
  fprintf(fp,"solid example\n");
  while(SD.frontier.size() > 0)
  {
    ConvexPoly cvx(&SD);
    assert(SD.nu == 0);
    SD.enumerate(cvx);
    cvx.output_stl(0,1,2,fp);
    SD.progress();
  }
  fprintf(fp,"endsolid example\n");
  fclose(fp);

  return 0;
}
#else
int main()
{
  int const nd = 4;
  int const nu = 1;
  double const grain = 0.02;
  double y0[nd] = {0,0,0,0};
  CoDim2 SD(nd,nu,grain);
  SD.ResidualFunction = rrd;
  SD.BoundaryFunction = rrd_boundary;
  SD.start(y0);
  FILE* fp = fopen("rrd.stl","w");
  fprintf(fp,"solid example\n");
  while(SD.frontier.size() > 0)
  {
    ConvexPoly cvx(&SD);
    SD.enumerate(cvx);
    cvx.response_stl(0,fp);
    SD.progress();
  }
  fprintf(fp,"endsolid example\n");
  fclose(fp);

  return 0;
}
#endif
