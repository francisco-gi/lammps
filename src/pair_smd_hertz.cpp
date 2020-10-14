/* ----------------------------------------------------------------------
 *
 *                    *** Smooth Mach Dynamics ***
 *
 * This file is part of the USER-SMD package for LAMMPS.
 * Copyright (2014) Georg C. Ganzenmueller, georg.ganzenmueller@emi.fhg.de
 * Fraunhofer Ernst-Mach Institute for High-Speed Dynamics, EMI,
 * Eckerstrasse 4, D-79104 Freiburg i.Br, Germany.
 *
 * ----------------------------------------------------------------------- */

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

/* ----------------------------------------------------------------------
   Contributing author: Mike Parks (SNL)
------------------------------------------------------------------------- */

#include "pair_smd_hertz.h"
#include <mpi.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include "atom.h"
#include "domain.h"
#include "force.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

#define SQRT2 1.414213562e0

/* ---------------------------------------------------------------------- */

PairHertz::PairHertz(LAMMPS *lmp) :
                Pair(lmp) {

        onerad_dynamic = onerad_frozen = maxrad_dynamic = maxrad_frozen = NULL;
        bulkmodulus = NULL;
        kn = NULL;
        scale = 1.0;
}

/* ---------------------------------------------------------------------- */

PairHertz::~PairHertz() {

//printf("\n\n\nPairHertz::~PairHertz()\n\t smd_force_h->[0][2] = %lf\n",(atom->smd_force_h)[0][2]);
//int i,ii,inum,*ilist; //,*numneigh;
//        inum = list->inum;
//        ilist = list->ilist;
//        numneigh = list->numneigh;
//        firstneigh = list->firstneigh;


        // loop over neighbors of my atoms
//        for (ii = 0; ii < inum; ii++) {
//                i = ilist[ii];
//printf("i is %d\n",i);
//i = 0;
//while( i < 251*5)
//{
//		printf("%lf\t%lf\t%lf\n",(atom->smd_force_h)[i][0],(atom->smd_force_h)[i][1],(atom->smd_force_h)[i][2]);
//		i++;
//	}
//printf("i is %d\n",i);
        if (allocated) {
                memory->destroy(setflag);
                memory->destroy(cutsq);
                memory->destroy(bulkmodulus);
                memory->destroy(kn);

                delete[] onerad_dynamic;
                delete[] onerad_frozen;
                delete[] maxrad_dynamic;
                delete[] maxrad_frozen;
        }
}

/* ---------------------------------------------------------------------- */

void PairHertz::compute(int eflag, int vflag) {
        int i, j, ii, jj, inum, jnum, itype, jtype;
        double xtmp, ytmp, ztmp, delx, dely, delz;
        double rsq, r, evdwl, fpair;
        int *ilist, *jlist, *numneigh, **firstneigh;
        double rcut, r_geom, delta, ri, rj, dt_crit;
        double *rmass = atom->rmass;

        evdwl = 0.0;
        ev_init(eflag, vflag);

        double **f = atom->f;
        double **x = atom->x;
        double **x0 = atom->x0;
        int *type = atom->type;
        int nlocal = atom->nlocal;
        double *radius = atom->contact_radius;
        double *sph_radius = atom->radius;
        double rcutSq;
        double delx0, dely0, delz0, rSq0, sphCut;

	// $$$$
        double **f_h = atom->smd_force_h;
	printf("\n\n\n\n\n\n\n\n\n**********************************************************\n");	
	printf("PAIR_SMD_HERTZ\n\nlocal atoms are %d\tf_h[2] = %lf\n",atom->nlocal,f_h[atom->nlocal][2]);
	printf("\n**********************************************************\n\n\n\n\n\n\n\n\n");	

        int newton_pair = force->newton_pair;
        int periodic = (domain->xperiodic || domain->yperiodic || domain->zperiodic);

        inum = list->inum;
        ilist = list->ilist;
        numneigh = list->numneigh;
        firstneigh = list->firstneigh;

        stable_time_increment = 1.0e22;

        // loop over neighbors of my atoms
        for (ii = 0; ii < inum; ii++) {
                i = ilist[ii];
                xtmp = x[i][0];
                ytmp = x[i][1];
                ztmp = x[i][2];
                itype = type[i];
                ri = scale * radius[i];
                jlist = firstneigh[i];
                jnum = numneigh[i];

                for (jj = 0; jj < jnum; jj++) {
                        j = jlist[jj];
                        j &= NEIGHMASK;

                        jtype = type[j];

                        delx = xtmp - x[j][0];
                        dely = ytmp - x[j][1];
                        delz = ztmp - x[j][2];
						// // double delxx = delx;
//						double delyy = dely;
//						double delzz = delz;
//						
//						// If condition added by me.
//						if (periodic) {
//							printf("\n\nPBC-CHECK\n\n");
//							domain->minimum_image(delx, dely, delz);
//						}
				
                        // double rsqq = delxx * delxx + delyy * delyy + delzz * delzz;
                        rsq = delx * delx + dely * dely + delz * delz;
						// If you print this you can see there are a lot of differences
						// if ( rsqq > rsq )
						// {
							// printf("\nDifference between distances pre and post modification\n%lf\n",rsqq-rsq);
						// }
                        rj = scale * radius[j];
                        rcut = ri + rj;
                        rcutSq = rcut * rcut;

						// if ( (rsq < rcutSq) && (rsqq > rcutSq) )
						// {
							// printf("\n************************************\nWARNING : Differences in interacting pairs\n************************************\n");
						// }

                        if (rsq < rcutSq) {
								
                                /*
                                 * self contact option:
                                 * if pair of particles was initially close enough to interact via a bulk continuum mechanism (e.g. SPH), exclude pair from contact forces.
                                 * this approach should work well if no updates of the reference configuration are performed.
                                 */

                                if (itype == jtype) {
                                        delx0 = x0[j][0] - x0[i][0];
                                        dely0 = x0[j][1] - x0[i][1];
                                        delz0 = x0[j][2] - x0[i][2];
                                        if (periodic) {
                                                domain->minimum_image(delx0, dely0, delz0);
                                        }
                                        rSq0 = delx0 * delx0 + dely0 * dely0 + delz0 * delz0; // initial distance
                                        sphCut = sph_radius[i] + sph_radius[j];
                                        if (rSq0 < sphCut * sphCut) {
                                                rcut = 0.5 * rcut;
                                                rcutSq = rcut * rcut;
                                                if (rsq > rcutSq) {
                                                        continue;
                                                }
                                        }
                                }

                                r = sqrt(rsq);
                                //printf("hertz interaction, r=%f, cut=%f, h=%f\n", r, rcut, sqrt(rSq0));

                                // Hertzian short-range forces
                                delta = rcut - r; // overlap distance
                                r_geom = ri * rj / rcut;
                                //assuming poisson ratio = 1/4 for 3d
                                fpair = 1.066666667e0 * bulkmodulus[itype][jtype] * delta * sqrt(delta * r_geom); //  units: N
                                evdwl = fpair * 0.4e0 * delta; // GCG 25 April: this expression conserves total energy
                                dt_crit = 3.14 * sqrt(0.5 * (rmass[i] + rmass[j]) / (fpair / delta));

                                stable_time_increment = MIN(stable_time_increment, dt_crit);
                                if (r > 2.0e-16) {
                                        fpair /= r; // divide by r and multiply with non-normalized distance vector
                                } else {
                                        fpair = 0.0;
                                }

                                /*
                                 * contact viscosity -- needs to be done, see GRANULAR package for normal & shear damping
                                 * for now: no damping and thus no viscous energy deltaE
                                 */

                                if (evflag) {
                                        ev_tally(i, j, nlocal, newton_pair, evdwl, 0.0, fpair, delx, dely, delz);
                                }

                                f[i][0] += delx * fpair;
                                f[i][1] += dely * fpair;
                                f[i][2] += delz * fpair;

                                if (newton_pair || j < nlocal) {
                                        f[j][0] -= delx * fpair;
                                        f[j][1] -= dely * fpair;
                                        f[j][2] -= delz * fpair;
                                }

                        } // end 'if' particles interacting
                } // end of loop 'for' jj<jnum neighbours of i
//printf("\n j list finished with number %d\t",j);
//printf("i number %d\n",i);
//printf("\n jj is %d out of jnum %d\t",jj,jnum);
//printf(" ii is  %d out of inum %d\n",ii,inum);
               f_h[i][0] = f[i][0];
               f_h[i][1] = f[i][1];
               f_h[i][2] = f[i][2];
if (i==inum-1)
	printf( "FORCES\t%lf\t%lf\t%lf\n" ,f_h[0][0], f_h[0][1] , f_h[0][2] );
        }

//      double stable_time_increment_all = 0.0;
//      MPI_Allreduce(&stable_time_increment, &stable_time_increment_all, 1, MPI_DOUBLE, MPI_MIN, world);
//      if (comm->me == 0) {
//              printf("stable time step for pair smd/hertz is %f\n", stable_time_increment_all);
//      }
}

/* ----------------------------------------------------------------------
 allocate all arrays
 ------------------------------------------------------------------------- */

void PairHertz::allocate() {
        allocated = 1;
        int n = atom->ntypes;

        memory->create(setflag, n + 1, n + 1, "pair:setflag");
        for (int i = 1; i <= n; i++)
                for (int j = i; j <= n; j++)
                        setflag[i][j] = 0;

        memory->create(bulkmodulus, n + 1, n + 1, "pair:kspring");
        memory->create(kn, n + 1, n + 1, "pair:kn");

        memory->create(cutsq, n + 1, n + 1, "pair:cutsq"); // always needs to be allocated, even with granular neighborlist

        onerad_dynamic = new double[n + 1];
        onerad_frozen = new double[n + 1];
        maxrad_dynamic = new double[n + 1];
        maxrad_frozen = new double[n + 1];
}

/* ----------------------------------------------------------------------
 global settings
 ------------------------------------------------------------------------- */

void PairHertz::settings(int narg, char **arg) {
        if (narg != 1)
                error->all(FLERR, "Illegal number of args for pair_style hertz");

        scale = force->numeric(FLERR, arg[0]);
        if (comm->me == 0) {
                printf("\n>>========>>========>>========>>========>>========>>========>>========>>========\n");
                printf("SMD/HERTZ CONTACT SETTINGS:\n");
                printf("... effective contact radius is scaled by %f\n", scale);
                printf(">>========>>========>>========>>========>>========>>========>>========>>========\n");
        }

}

/* ----------------------------------------------------------------------
 set coeffs for one or more type pairs
 ------------------------------------------------------------------------- */

void PairHertz::coeff(int narg, char **arg) {
        if (narg != 3)
                error->all(FLERR, "Incorrect args for pair coefficients");
        if (!allocated)
                allocate();

        int ilo, ihi, jlo, jhi;
        force->bounds(FLERR,arg[0], atom->ntypes, ilo, ihi);
        force->bounds(FLERR,arg[1], atom->ntypes, jlo, jhi);

        double bulkmodulus_one = atof(arg[2]);

        // set short-range force constant
        double kn_one = 0.0;
        if (domain->dimension == 3) {
                kn_one = (16. / 15.) * bulkmodulus_one; //assuming poisson ratio = 1/4 for 3d
        } else {
                kn_one = 0.251856195 * (2. / 3.) * bulkmodulus_one; //assuming poisson ratio = 1/3 for 2d
        }

        int count = 0;
        for (int i = ilo; i <= ihi; i++) {
                for (int j = MAX(jlo, i); j <= jhi; j++) {
                        bulkmodulus[i][j] = bulkmodulus_one;
                        kn[i][j] = kn_one;
                        setflag[i][j] = 1;
                        count++;
                }
        }

        if (count == 0)
                error->all(FLERR, "Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
 init for one type pair i,j and corresponding j,i
 ------------------------------------------------------------------------- */

double PairHertz::init_one(int i, int j) {

        if (!allocated)
                allocate();

        if (setflag[i][j] == 0)
                error->all(FLERR, "All pair coeffs are not set");

        bulkmodulus[j][i] = bulkmodulus[i][j];
        kn[j][i] = kn[i][j];

        // cutoff = sum of max I,J radii for
        // dynamic/dynamic & dynamic/frozen interactions, but not frozen/frozen

        double cutoff = maxrad_dynamic[i] + maxrad_dynamic[j];
        cutoff = MAX(cutoff, maxrad_frozen[i] + maxrad_dynamic[j]);
        cutoff = MAX(cutoff, maxrad_dynamic[i] + maxrad_frozen[j]);

        if (comm->me == 0) {
                printf("cutoff for pair smd/hertz = %f\n", cutoff);
        }
        return cutoff;
}

/* ----------------------------------------------------------------------
 init specific to this pair style
 ------------------------------------------------------------------------- */

void PairHertz::init_style() {
        int i;

        // error checks

        if (!atom->contact_radius_flag)
                error->all(FLERR, "Pair style smd/hertz requires atom style with contact_radius");

        int irequest = neighbor->request(this);
        neighbor->requests[irequest]->size = 1;

        // set maxrad_dynamic and maxrad_frozen for each type
        // include future Fix pour particles as dynamic

        for (i = 1; i <= atom->ntypes; i++)
                onerad_dynamic[i] = onerad_frozen[i] = 0.0;

        double *radius = atom->radius;
        int *type = atom->type;
        int nlocal = atom->nlocal;

        for (i = 0; i < nlocal; i++) {
                onerad_dynamic[type[i]] = MAX(onerad_dynamic[type[i]], radius[i]);
        }

        MPI_Allreduce(&onerad_dynamic[1], &maxrad_dynamic[1], atom->ntypes, MPI_DOUBLE, MPI_MAX, world);
        MPI_Allreduce(&onerad_frozen[1], &maxrad_frozen[1], atom->ntypes, MPI_DOUBLE, MPI_MAX, world);
}

/* ----------------------------------------------------------------------
 neighbor callback to inform pair style of neighbor list to use
 optional granular history list
 ------------------------------------------------------------------------- */

void PairHertz::init_list(int id, NeighList *ptr) {
        if (id == 0)
                list = ptr;
}

/* ----------------------------------------------------------------------
 memory usage of local atom-based arrays
 ------------------------------------------------------------------------- */

double PairHertz::memory_usage() {

        return 0.0;
}

void *PairHertz::extract(const char *str, int &/*i*/) {
        //printf("in PairTriSurf::extract\n");
        if (strcmp(str, "smd/hertz/stable_time_increment_ptr") == 0) {
                return (void *) &stable_time_increment;
        }

        return NULL;

}
