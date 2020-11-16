/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

ComputeStyle(smd/force/interact,ComputeSmdForceInteract)

#else

#ifndef LMP_COMPUTE_SMD_FORCE_INTERACT_H
#define LMP_COMPUTE_SMD_FORCE_INTERACT_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeSmdForceInteract : public Compute {
 public:
  ComputeSmdForceInteract(class LAMMPS *, int, char **);
  ~ComputeSmdForceInteract();
  void init();
  void init_list(int, class NeighList *);
  void compute_local();
  double memory_usage();
  int array_max_index(int * , int );

 private:
  int nvalues,ncount,cutstyle;

  int *pstyle;              // style of each requested output
  int *pindex;              // for pI, index of the output (0 to M-1)
  int singleflag;

  int nmax;
  double *vlocal;
  double **alocal;
/* $$$$ */
  double ***flocal;
  int *main_type;
  int ntypes;

  class NeighList *list;

  int compute_pairs(int);
  void reallocate(int);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Invalid keyword in compute smd/force/interact command

Self-explanatory.

E: Compute smd/force/interact requires atom attribute radius

UNDOCUMENTED

E: No pair style is defined for compute smd/force/interact

Self-explanatory.

E: Pair style does not support compute smd/force/interact

The pair style does not have a single() function, so it can
not be invoked by compute smd/force/interact.

E: Pair style does not have extra field requested by compute smd/force/interact

The pair style does not support the pN value requested by the compute
smd/force/interact command.

*/
