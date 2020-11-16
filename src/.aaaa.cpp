/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "compute_smd_force_interact.h"
#include <cmath>
#include <cstring>
#include <cstdlib>
#include "atom.h"
#include "update.h"
#include "force.h"
#include "pair.h"
#include "neighbor.h"
#include "neigh_request.h"
#include "neigh_list.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

#define DELTA 10000

enum{DIST,ENG,FORCE,FX,FY,FZ,PN,FHX,FHY,FHZ};
enum{TYPE,RADIUS};

/* ---------------------------------------------------------------------- */

ComputeSmdForceInteract::ComputeSmdForceInteract(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg),
  pstyle(NULL), pindex(NULL), vlocal(NULL), alocal(NULL), flocal(NULL), main_type(NULL)
{
  if (narg < 4) error->all(FLERR,"Illegal compute smd/force/interact command");

  local_flag = 1;
  /* number of quantities required by the compute */
  nvalues = narg - 3;
/*$$$$*/
printf("NARG\t%d\n",narg);
   ntypes = atom->ntypes;
  pstyle = new int[nvalues];
  pindex = new int[nvalues];
printf("NTYPES\t%d\n",ntypes);
main_type = new int[ntypes];
  nvalues = 0;
  int iarg = 3;
  while (iarg < narg) {
printf("%s\n",arg[iarg]);
    if (strcmp(arg[iarg],"dist") == 0) pstyle[nvalues++] = DIST;
    else if (strcmp(arg[iarg],"eng") == 0) pstyle[nvalues++] = ENG;
    else if (strcmp(arg[iarg],"force") == 0) pstyle[nvalues++] = FORCE;
    else if (strcmp(arg[iarg],"fx") == 0) pstyle[nvalues++] = FX;
    else if (strcmp(arg[iarg],"fy") == 0) pstyle[nvalues++] = FY;
    else if (strcmp(arg[iarg],"fz") == 0) pstyle[nvalues++] = FZ;
    else if (strcmp(arg[iarg],"fhx") == 0) pstyle[nvalues++] = FHX;
    else if (strcmp(arg[iarg],"fhy") == 0) pstyle[nvalues++] = FHY;
    else if (strcmp(arg[iarg],"fhz") == 0) pstyle[nvalues++] = FHZ;
    else if (arg[iarg][0] == 'p') {
      int n = atoi(&arg[iarg][1]);
      if (n <= 0) error->all(FLERR,
                             "Invalid keyword in compute smd/force/interact command");
      pstyle[nvalues] = PN;
      pindex[nvalues++] = n-1;
// $$$$
//    }
//    else if (strcmp(arg[iarg],"f_hertz") == 0){
//	pstyle[nvalues++] = F_HERTZ;
    }else break;

    iarg++;
  }

  // optional args

  cutstyle = TYPE;

  while (iarg < narg) {
    if (strcmp(arg[iarg],"cutoff") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal compute smd/force/interact command");
      if (strcmp(arg[iarg+1],"type") == 0) cutstyle = TYPE;
      else if (strcmp(arg[iarg+1],"radius") == 0) cutstyle = RADIUS;
      else error->all(FLERR,"Illegal compute smd/force/interact command");
      iarg += 2;
    } else error->all(FLERR,"Illegal compute smd/force/interact command");
  }

  // error check

  if (cutstyle == RADIUS && !atom->radius_flag)
    error->all(FLERR,"Compute smd/force/interact requires atom attribute radius");

  // set singleflag if need to call pair->single()

  singleflag = 0;
  for (int i = 0; i < nvalues; i++)
    if (pstyle[i] != DIST) singleflag = 1;

  if (nvalues == 1) size_local_cols = 0;
  else size_local_cols = nvalues;

  nmax = 0;
//  ntypes = NULL;
  vlocal = NULL;
  alocal = NULL;
  flocal = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeSmdForceInteract::~ComputeSmdForceInteract()
{
//  memory->destroy(ntypes);
  memory->destroy(vlocal);
  memory->destroy(alocal);
//  memory->destroy(flocal);
  delete [] pstyle;
  delete [] pindex;
  delete [] main_type;
}

/* ---------------------------------------------------------------------- */


/*only for 1d int array of positive numbers,returns the INDEX of the biggest value: e.g. if max value is array[i] -> then array_max output is i */
int ComputeSmdForceInteract::array_max_index(int *array, int length)
{
printf("CALL : array_max_index(\n");
  int x_max=0;
  int max_index=0;
  for(int i=0; i<length; i++){
	printf("Array_[%d]\t = %d\n",i,array[i]);
	if (array[i]>x_max){
	printf("NEW MAX : Array_[%d]\t = %d\n",i,array[i]);
		x_max=array[i];
		max_index = i;
		}
	}
return(max_index);
}

/* ---------------------------------------------------------------------- */

void ComputeSmdForceInteract::init()
{
  if (singleflag && force->pair == NULL)
    error->all(FLERR,"No pair style is defined for compute smd/force/interact");
  if (singleflag && force->pair->single_enable == 0)
    error->all(FLERR,"Pair style does not support compute smd/force/interact");

  for (int i = 0; i < nvalues; i++)
    if (pstyle[i] == PN && pindex[i] >= force->pair->single_extra)
      error->all(FLERR,"Pair style does not have extra field"
                 " requested by compute smd/force/interact");

  // need an occasional half neighbor list
  // set size to same value as request made by force->pair
  // this should enable it to always be a copy list (e.g. for granular pstyle)

  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->compute = 1;
  neighbor->requests[irequest]->occasional = 1;
  NeighRequest *pairrequest = neighbor->find_request((void *) force->pair);
  if (pairrequest) neighbor->requests[irequest]->size = pairrequest->size;
}

/* ---------------------------------------------------------------------- */

void ComputeSmdForceInteract::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void ComputeSmdForceInteract::compute_local()
{
  invoked_local = update->ntimestep;

  // count local entries and compute pair info
/*$$$$*/
printf("**************************\nCOMPUTE LOCAL  1\n");
  ncount = compute_pairs(0);

printf("**************************\nCOMPUTE LOCAL  2\n");
  if (ncount > nmax) reallocate(ncount);
  size_local_rows = ncount;
/*$$$$*/
printf("**************************\nCOMPUTE LOCAL  3\n");
  compute_pairs(2);
}

/* ----------------------------------------------------------------------
   count pairs and compute pair info on this proc
   only count pair once if newton_pair is off
   both atom I,J must be in group
   if flag is set, compute requested info about pair
------------------------------------------------------------------------- */

int ComputeSmdForceInteract::compute_pairs(int flag)
{
  int i,j,m,n,ii,jj,inum,jnum,itype,jtype;
  tagint itag,jtag;
  double xtmp,ytmp,ztmp,delx,dely,delz;
  double rsq,radsum,eng,fpair,factor_coul,factor_lj;
  int *ilist,*jlist,*numneigh,**firstneigh;
  double *ptr;

  double **x = atom->x;
  /* $$$$ radius to contact_radius, specific for SMD hertz potential */
  double *radius = atom->contact_radius;
  tagint *tag = atom->tag;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  double *special_coul = force->special_coul;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;


//printf("\tNTYPES\t%d\n",ntypes);

  // invoke half neighbor list (will copy or build if necessary)

  if (flag == 0) neighbor->build_one(list);

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms
  // skip if I or J are not in group
  // for newton = 0 and J = ghost atom,
  //   need to insure I,J pair is only output by one proc
  //   use same itag,jtag logic as in Neighbor::neigh_half_nsq()
  // for flag = 0, just count pair interactions within force cutoff
  // for flag = 1, calculate requested output fields

  Pair *pair = force->pair;
  double **cutsq = force->pair->cutsq;

  m = 0;
/* Zeroizing alocal array maybe better to move under flag==0*/
if(alocal){
printf("PRINTING ALOCAL\n");
for(int iii=0; iii<(ntypes*(ntypes+1)/2); iii++){
	for(int jjj = 0 ; jjj<nvalues  ; jjj++){
		printf("%d\t%d\t%lf\n",iii,jjj,alocal[iii][jjj]);
	alocal[iii][jjj]=0.;
		}
	}
}

/*start loop on i particle*/
  for (ii = 0; ii < inum; ii++) {

/* $$$$ */
for(int iii=0; iii<ntypes; iii++){
printf("main_type[iii] = %lf\n",main_type[iii]);
	main_type[iii]=0;}

    i = ilist[ii];
    if (!(mask[i] & groupbit)) continue;

    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itag = tag[i];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

/*$$$$*/
printf("##################\nTYPE of atom i\t%d\n",itype);
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      factor_coul = special_coul[sbmask(j)];
      j &= NEIGHMASK;

      if (!(mask[j] & groupbit)) continue;

      // itag = jtag is possible for long cutoffs that include images of self

      if (newton_pair == 0 && j >= nlocal) {
        jtag = tag[j];
        if (itag > jtag) {
          if ((itag+jtag) % 2 == 0) continue;
        } else if (itag < jtag) {
          if ((itag+jtag) % 2 == 1) continue;
        } else {
          if (x[j][2] < ztmp) continue;
          if (x[j][2] == ztmp) {
            if (x[j][1] < ytmp) continue;
            if (x[j][1] == ytmp && x[j][0] < xtmp) continue;
          }
        }
      }

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (cutstyle == TYPE) {
printf("cutstyle == TYPE\n");
        if (rsq >= cutsq[itype][jtype])
		 continue;
      } else {
//printf("ELSE\n");
        radsum = radius[i] + radius[j];
printf("FOR TYPES %d-%d\tsqrt(rsq) = %lf\tr_i+r_j=%lf\tr_cut=%lf\n",itype,jtype,sqrt(rsq),radsum,sqrt(cutsq[i][j]));
        if (rsq >= radsum*radsum){
	printf("continue	r=%lf>r_c=%lf\n",sqrt(rsq),radsum);
	 continue;}
      }
	/* $$$$ */
	/* If interparticle distance less than contact distance, increase +1 to the contact with molecule of jtype*/
	if(itype!=jtype && rsq<cutsq[i][j]){
printf("CONTACT TYPE\t%d-%d\t+1\n",itype,jtype);
	main_type[jtype-1]++;
}
      if (flag==1) {
printf("\nCOMPUTE PAIRS FLAG IS 1\n");
        if (singleflag)
          eng = pair->single(i,j,itype,jtype,rsq,factor_coul,factor_lj,fpair);
        else eng = fpair = 0.0;

        if (nvalues == 1) ptr = &vlocal[m];
        else ptr = alocal[m];

        for (n = 0; n < nvalues; n++) {
          switch (pstyle[n]) {
          case DIST:
            ptr[n] = sqrt(rsq);
            break;
          case ENG:
            ptr[n] = eng;
            break;
          case FORCE:
            ptr[n] = sqrt(rsq)*fpair;
            break;
          case FX:
            ptr[n] = delx*fpair;
            break;
          case FY:
            ptr[n] = dely*fpair;
            break;
          case FZ:
            ptr[n] = delz*fpair;
            break;
          case PN:
            ptr[n] = pair->svector[pindex[n]];
            break;
          }
        }
      } // end of if(flag==1)

      m++;
    } // end of j loop
printf("END OF J LOOP\n");

for(int iii=0; iii<ntypes; iii++)
	if(itype==iii+1 && main_type[iii]!=0) printf("WARNING: self contact detected for type\t%d\t,detected %d\ttimes\n",itype,main_type[iii]);

	/*$$$$*/
	//printf("MAIN TYPE\n");
			printf( "MAIN TYPE ARRAY\t for atom %d of type %d\n",ii,itype);
		for(int iii=0; iii<ntypes; iii++)
			printf( "\t%d\n",main_type[iii] );

		printf( "MAX FOR INDEX:\t%d\n", array_max_index(main_type,ntypes) );
//			printf("\n");

/* $$$$ */
/* 1 HOW TO SAVE IN PTR
   2 WHAT HAPPEN WITH PTR THEN
   3 WHAT WHEN I PRINT
*/
	if(flag==2){
/* the first ntypes pairs are 1-1, 1-2, .., 1-ntypes
   then, in order 2-2, 2-3, .., 2-ntype, 3-3, .., ..,
   (ntypes-1)-(ntypes-1), (ntypes-1)-ntypes, ntypes-ntypes
   In total thy are 0.5*ntypes*(ntypes+1) */
/*  Computation of main contact*/
/*	M CAN BE negative; THIS is saving the values in the incorrect place	*/
	jtype = array_max_index(main_type,ntypes)+1;
printf("FLAG == 2\titype = %d\tjtype = %d\n",itype,jtype);
	m = jtype-itype;
	if ( m < 0 ) m = -m; 
if(itype!=1)
	m += 0.5*(2*ntypes-itype+2)*(itype-1);
//        if (nvalues == 1) ptr = &vlocal[m];
//        else 
	ptr = alocal[m];

        for (n = 0; n < nvalues; n++) {
          switch (pstyle[n]) {
          case FHX:
            ptr[n] += atom->smd_force_h[i][0];
            break;
          case FHY:
            ptr[n] += atom->smd_force_h[i][1];
            break;
          case FHZ:
            ptr[n] += atom->smd_force_h[i][2];
            break;
          }
	}
      } // end of flag==2 case


  }  // end of i loop
printf("NUMBER OF MOLECULAR PAIRS\n",ntypes*(ntypes+1)/2);
/*the return is the number of molecule pairs*/
  return (ntypes*(ntypes+1)/2);
}

/* ---------------------------------------------------------------------- */

void ComputeSmdForceInteract::reallocate(int n)
{
  // grow vector_local or array_local

  while (nmax < n) nmax += DELTA;


//    memory->destroy(flocal);
//    memory->create(flocal,ntypes,3*ntypes,"smd/force/interact:farray_local");

  if (nvalues == 1) {
    memory->destroy(vlocal);
    memory->create(vlocal,nmax,"smd/force/interact:vector_local");
    vector_local = vlocal;
//  } 
/*$$$$*/
//    else if(){
//    memory->destroy(flocal);
//    memory->create(flocal,ntypes,3*ntypes,"smd/force/interact:farray_local");
//    array_local = flocal;
    } else {
/* The data is saved in alocal array, this is a number_of_pairs x number_of_observables array*/
    memory->destroy(alocal);
    memory->create(alocal,nmax,nvalues,"smd/force/interact:array_local");
    array_local = alocal;
printf("MEMORY array_local allocated\n");
  }
}

/* ----------------------------------------------------------------------
   memory usage of local data
------------------------------------------------------------------------- */

double ComputeSmdForceInteract::memory_usage()
{
  double bytes = nmax*nvalues * sizeof(double);
  return bytes;
}
