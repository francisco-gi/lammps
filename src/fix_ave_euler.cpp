/* ----------------------------------------------------------------------
    This is the

    ██╗     ██╗ ██████╗  ██████╗  ██████╗ ██╗  ██╗████████╗███████╗
    ██║     ██║██╔════╝ ██╔════╝ ██╔════╝ ██║  ██║╚══██╔══╝██╔════╝
    ██║     ██║██║   ██║██║   ██║██║   ██║██╔══██║   ██║   ╚════██║
    ███████╗██║╚██████╔╝╚██████╔╝╚██████╔╝██║  ██║   ██║   ███████║
    ╚══════╝╚═╝ ╚═════╝  ╚═════╝  ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚══════╝®

    DEM simulation engine, released by
    DCS Computing Gmbh, Linz, Austria
    http://www.dcs-computing.com, office@dcs-computing.com

    LIGGGHTS® is part of CFDEM®project:
    http://www.liggghts.com | http://www.cfdem.com

    Core developer and main author:
    Christoph Kloss, christoph.kloss@dcs-computing.com

    LIGGGHTS® is open-source, distributed under the terms of the GNU Public
    License, version 2 or later. It is distributed in the hope that it will
    be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
    of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. You should have
    received a copy of the GNU General Public License along with LIGGGHTS®.
    If not, see http://www.gnu.org/licenses . See also top-level README
    and LICENSE files.

    LIGGGHTS® and CFDEM® are registered trade marks of DCS Computing GmbH,
    the producer of the LIGGGHTS® software and the CFDEM®coupling software
    See http://www.cfdem.com/terms-trademark-policy for details.

-------------------------------------------------------------------------
    Contributing author and copyright for this file:
    (if not contributing author is listed, this file has been contributed
    by the core developer)

    Copyright 2012-     DCS Computing GmbH, Linz
    Copyright 2009-2012 JKU Linz
------------------------------------------------------------------------- */

#include <stdlib.h>
#include <string.h>
#include <cmath>
#include <iostream>
#include "mpi_liggghts.h"
#include "fix_ave_euler.h"
//#include "fix_multisphere.h"
#include "compute_stress_atom.h"
#include "math_extra.h"			// modified from math_extra_liggghts.h to math_extra.h
#include "atom.h"
#include "force.h"
#include "domain.h"
#include "modify.h"
#include "neighbor.h"
#include "region.h"
#include "update.h"
#include "random_park.h"
#include "memory.h"
#include "error.h"
#include "comm.h"
//
#include "pair.h"	// because we need access to members of the class
#include "smd_kernels.h"	// because we need access to members of the class
#include <Eigen/Eigen>

#include <typeinfo>

#define BIG 1000000000
#define INVOKED_PERATOM 8 

using namespace LAMMPS_NS;
using namespace FixConst;
//
using namespace SMD_Kernels;
using namespace Eigen;

//int main() {
//  int i;
//  std::cout << typeid(i).name();
//  return 0;
//}

static Matrix3d Deviator(Matrix3d M) {
	Matrix3d eye;
	eye.setIdentity();
	eye *= M.trace() / 3.0;
	return M - eye;
}         


/* ---------------------------------------------------------------------- */

FixAveEuler::FixAveEuler(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
  parallel_(true),
  exec_every_(1),
  box_change_size_(false),
  box_change_domain_(false),
  cell_size_ideal_rel_(3.),
  cell_size_ideal_(0.),
  ncells_(0),
  ncells_max_(0),
  ncellptr_max_(0),
  cellhead_(NULL),
  cellptr_(NULL),
  idregion_(NULL),
  region_(NULL),
  center_(NULL),
  v_av_(NULL),
  vol_fr_(NULL),
  weight_(NULL),
  w(NULL),
  radius_(NULL),
  ncount_(NULL),
  mass_(NULL),
  stress_(NULL),
  compute_stress_(NULL),
  rho(NULL),
  def_grad_cell(NULL),
  random_(0)
{
  // this fix produces a global array

//int count_1=0;

//while( count_1 < narg)
//{
//  std::cout << arg[count_1] << std::endl;
//  count_1++;
//}

//  printf("\n\nInitialization\n\n");
  array_flag = 1;
  size_array_rows = BIG;
  size_array_cols = 3 + 1 + 3;

  triclinic_ = domain->triclinic;  

  // random number generator, seed is hardcoded
  // correction str to int
  random_ = new RanPark(lmp,15485863);

  // parse args
  if (narg < 6) error->all(FLERR,"Illegal fix ave/pic command");
  int iarg = 3;


  if(strcmp(arg[iarg++],"nevery"))
  {
    error->fix_error(FLERR,this,"expecting keyword 'nevery'");
//    printf("\nnevery detected\n");
  }

  exec_every_ = force->inumeric(FLERR,arg[iarg++]);
  
  if(exec_every_ < 1)
    error->fix_error(FLERR,this,"'nevery' > 0 required");
  nevery = exec_every_;


  if(strcmp(arg[iarg++],"cell_size_relative"))
  {
    error->fix_error(FLERR,this,"expecting keyword 'cell_size_relative'");
  }

  cell_size_ideal_rel_ = force->numeric(FLERR,arg[iarg++]);

//std::cout << std::endl << "**********************************************\nRELATIVE CELL SIZE\n" << "cell_size_ideal_rel_\t" << cell_size_ideal_rel_ << std::endl;


//std::cout << std::endl << "cell_size_ideal_rel_\t" << cell_size_ideal_rel_ << std::endl;
 
  if(cell_size_ideal_rel_ < 1.)
  {
//    error->fix_error(FLERR,this,"'cell_size_relative' > 1 required");
      printf("'cell_size_relative' = %lf\n",cell_size_ideal_rel_);
  }

  if(strcmp(arg[iarg++],"parallel"))
  {
	error->fix_error(FLERR,this,"expecting keyword 'parallel'");
  }
  if(strcmp(arg[iarg],"yes") == 0)
  {
	parallel_ = true;
  } 
  else if(strcmp(arg[iarg],"no") == 0)
  {
	parallel_ = false;
  }
  else
  {
	error->fix_error(FLERR,this,"expecting 'yes' or 'no' after 'parallel'");
  }
  iarg++;

  while(iarg < narg)
  {
     if (strcmp(arg[iarg],"basevolume_region") == 0) {
       if (iarg+2 > narg) error->fix_error(FLERR,this,"");
       int iregion = domain->find_region(arg[iarg+1]);
       if (iregion == -1)
         error->fix_error(FLERR,this,"region ID does not exist");
       int n = strlen(arg[iarg+1]) + 1;
       idregion_ = new char[n];
       strcpy(idregion_,arg[iarg+1]);
       region_ = domain->regions[iregion];
       iarg += 2;
     } else {
       char *errmsg = new char[strlen(arg[iarg])+50];
       sprintf(errmsg,"unknown keyword or wrong keyword order: %s", arg[iarg]);
       error->fix_error(FLERR,this,errmsg);
       delete []errmsg;
     }
  }



//  std::cout << typeid(cell_size_ideal_rel_).name();
//std::cout << std::endl << "cell_size_ideal_rel_\t" << cell_size_ideal_rel_ << std::endl;
//printf("Cell size ideal relative : %ld\n",cell_size_ideal_rel);
//  printf("\n\nFixAveEuler::FixAveEuler - Completed\n\n");
}

/* ---------------------------------------------------------------------- */

FixAveEuler::~FixAveEuler()
{

//printf("\n\nFixAveEuler::~FixAveEuler \n\n");
  
  memory->destroy(cellhead_);
  memory->destroy(cellptr_);
  if(idregion_) delete []idregion_;
  memory->destroy(center_);
  memory->destroy(v_av_);
  memory->destroy(vol_fr_);
  memory->destroy(weight_);
  memory->destroy(w);
  memory->destroy(radius_);
  memory->destroy(ncount_);
  memory->destroy(mass_);
  memory->destroy(rho);
  memory->destroy(stress_);
  memory->destroy(def_grad_cell);
  if (random_) delete random_;

//printf("\n\nFixAveEuler::~FixAveEuler - Completed\n\n");

}

/* ---------------------------------------------------------------------- */

void FixAveEuler::post_constructor()
{
//  printf("fix ave euler - post_constructor()\n");
  //  stress computation, just for pairwise contribution
  if(!compute_stress_)
  {
//    printf("_______________________________________\n");
//    printf("IN POST_CREATE  -  COMP_STRESS_ not initialized\n");
//    printf("_______________________________________\n");
        
        const char* arg[4];
        arg[0]="stress_faveu";
        arg[1]="all";
        arg[2]="stress/atom";
        arg[3]="pair";

//    printf("In post_constructor adding compute\n");
//        modify->add_compute(4,(char**)arg);

//    printf("In post_constructor Compute stress\n");
//        compute_stress_ = static_cast<ComputeStressAtom*>(modify->compute[modify->find_compute(arg[0])]);
  }

  // ensure that the compute is calculated in the first time step
//    printf("In post_constructor - update timestep\n");
  bigint nfirst = (update->ntimestep/nevery)*nevery + nevery;
//    printf("In post_constructor Compute stress add step\n");
//  compute_stress_->addstep(nfirst);
}

/* ---------------------------------------------------------------------- */

int FixAveEuler::setmask()
{
//  printf("fix ave euler - setmask()\n");
  int mask = 0;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixAveEuler::init()
{
//printf("fix ave euler - init()\n");
  if(!atom->radius_flag)
    error->fix_error(FLERR,this,"requires atom attribute radius");
  if(!atom->rmass_flag)
    error->fix_error(FLERR,this,"requires atom attribute mass");

  // does not work with MS
  /*
   * FixMultisphere* fix_ms = static_cast<FixMultisphere*>(modify->find_fix_style("multisphere",0));
  if(fix_ms)
      error->fix_error(FLERR,this,"does not work with multisphere");
  */
  // check if box constant
  box_change_size_ = false;
  if(domain->box_change_size)
  {
    box_change_size_ = true;
  }
  box_change_domain_ = false;
  if(domain->box_change_domain)
  {
    box_change_domain_ = true;
  }
  if (region_)
  {
    int iregion = domain->find_region(idregion_);
    if (iregion == -1)
     error->fix_error(FLERR,this,"regions used by this command must not be deleted");
    region_ = domain->regions[iregion];
  }

  // error checks

  if (!parallel_ && 1 == domain->triclinic)
    error->fix_error(FLERR,this,"triclinic boxes only support 'parallel=yes'");
}

/* ----------------------------------------------------------------------
   setup initial bins
   only does averaging if nvalid = current timestep
------------------------------------------------------------------------- */

void FixAveEuler::setup(int vflag)
{
//printf("fix ave euler - setup\n");
    setup_bins();
    end_of_step();
//printf("\nFixAveEuler::setup  -  Concluded\n");
}

/* ----------------------------------------------------------------------
   setup 3d bins and their extent and coordinates
   called at setup() and when averaging occurs if box size changes
   similar to FixAveSpatial::setup_bins() and PairDSMC::init_style()

   bins are subbox - skin/2 so owned particles cannot move outside
   bins - so do not have to extrapolate
------------------------------------------------------------------------- */

void FixAveEuler::setup_bins()
{

// printf("fix ave euler - setup_bins\n"); 
   int ibin;
//std::cout << __cplusplus << std::endl ;

//std::cout << typeid(cell_size_ideal_rel_).name();

//std::cout << std::endl << "cell_size_ideal_rel_\t" << cell_size_ideal_rel_ << std::endl;


//  printf("Cell size ideal relative : %lf\n",cell_size_ideal_rel_);
  
   // calc ideal cell size as multiple of max cutoff
    cell_size_ideal_ = cell_size_ideal_rel_ * (neighbor->cutneighmax-neighbor->skin);

//printf("cell_size_ideal_ = %lf\n",cell_size_ideal_ );
//  printf("neighbor->cutneighmax-neighbor->skin : %lf\n",neighbor->cutneighmax-neighbor->skin); 
//  printf("Cell size ideal rel: %lf\n",cell_size_ideal_rel_);
//  printf("Cell size ideal: %lf\n",cell_size_ideal_);
 
    for(int dim = 0; dim < 3; dim++)
    {
      // GET BOUNDS
      if (triclinic_ == 0) {
//	printf("TRICLINIC == 0 \n");
//	std::cout << parallel_ << std::endl;
        lo_[dim] = parallel_ ? domain->sublo[dim] : domain->boxlo[dim];
        hi_[dim] = parallel_ ? domain->subhi[dim] : domain->boxhi[dim];
//	std::cout << parallel_ << std::endl;
      } else {
//	printf("TRICLINIC != 0 \n");
        lo_lamda_[dim] = domain->sublo_lamda[dim];
        hi_lamda_[dim] = domain->subhi_lamda[dim];
        cell_size_ideal_lamda_[dim] = cell_size_ideal_/domain->h[dim];
      }
//    std::cout << std::endl << "domain->box and sub\thi\tlo\t" <<  domain->boxhi[dim] << "\t" <<   domain->boxlo[dim] << "\t" << domain->subhi[dim] << "\t" << domain->sublo[dim] << std::endl;
//    std::cout << std::endl << "diff\thi\tlo\t" <<  hi_[dim]-lo_[dim] << "\t" << hi_[dim] << "\t" << lo_[dim] << std::endl;
//    printf("L_%d = %lf\t%lf - %lf\n",dim,hi_[dim]-lo_[dim],hi_[dim],lo_[dim]);
    }
    

    if (triclinic_) {
      domain->lamda2x(lo_lamda_,lo_);
      domain->lamda2x(hi_lamda_,hi_);
    }
    // Calculation of the extension of the cells for the three dimensions
    // round down (makes cell size larger)
    // at least one cell
    for(int dim = 0; dim < 3; dim++)
    {
      if (triclinic_) {
          ncells_dim_[dim] = static_cast<int>((hi_lamda_[dim]-lo_lamda_[dim])/cell_size_ideal_lamda_[dim]);
          if (ncells_dim_[dim] < 1) {
            ncells_dim_[dim] = 1;
            error->warning(FLERR,"Number of cells for fix_ave_euler was less than 1");
          }
          cell_size_lamda_[dim] = (hi_lamda_[dim]-lo_lamda_[dim])/static_cast<double>(ncells_dim_[dim]);
          cell_size_[dim] = cell_size_lamda_[dim]*domain->h[dim];
      } else {
	  // number of cells approximated by defect, hence larger cells
          ncells_dim_[dim] = static_cast<int>((hi_[dim]-lo_[dim])/cell_size_ideal_); // n cells in dimension dim 

//std::cout << "ncells_dim_:\t"  << ncells_dim_[dim] << std::endl;

          if (ncells_dim_[dim] < 1) {
            ncells_dim_[dim] = 1;
            
            error->warning(FLERR,"Number of cells for fix_ave_euler was less than 1");
          }
          cell_size_[dim] = (hi_[dim]-lo_[dim])/static_cast<double>(ncells_dim_[dim]);
//std::cout << "cell_size_:\t"  << cell_size_[dim] << std::endl;

      }
    } // loop over dimensions x,y,z

if(logfile)
	fprintf(logfile,"cell_size_ideal = %lf\t x,y,z= %lf\t%lf\t%lf\n",cell_size_ideal_,cell_size_[0],cell_size_[1],cell_size_[2]);  

    for(int dim = 0; dim < 3; dim++)
    {
        cell_size_inv_[dim] = 1./cell_size_[dim];
        if (triclinic_) cell_size_lamda_inv_[dim] = 1./cell_size_lamda_[dim];

    }
// total number of cells and volume 
    ncells_ = ncells_dim_[0]*ncells_dim_[1]*ncells_dim_[2];
    
    cell_volume_ = cell_size_[0]*cell_size_[1]*cell_size_[2];
    
    // (re) allocate spatial bin arrays
    if (ncells_ > ncells_max_)
    {
        ncells_max_ = ncells_;
        memory->grow(cellhead_,ncells_max_,"ave/euler:cellhead_");
        memory->grow(center_,ncells_max_,3,"ave/euler:center_");
        memory->grow(v_av_,  ncells_max_,3,"ave/euler:v_av_");
        memory->grow(vol_fr_,ncells_max_,  "ave/euler:vol_fr_");
        memory->grow(weight_,ncells_max_,  "ave/euler:weight_");
        memory->grow(w,ncells_max_,  "ave/euler:w");
        memory->grow(radius_,ncells_max_,  "ave/euler:radius_");
        memory->grow(ncount_,ncells_max_,    "ave/euler:ncount_");
        memory->grow(mass_,ncells_max_,    "ave/euler:mass_");
        memory->grow(stress_,ncells_max_,7,"ave/euler:stress_");
        memory->grow(rho,ncells_max_,    "ave/euler:rho");
        memory->grow(def_grad_cell,ncells_max_,9,"ave/euler:def_grad_cell");
    }
//printf("\nSpatial Arrays bin reallocated\n");
    // calculate center coordinates for cells
    for(int ix = 0; ix < ncells_dim_[0]; ix++)
    {
        for(int iy = 0; iy < ncells_dim_[1]; iy++)
        {
            for(int iz = 0; iz < ncells_dim_[2]; iz++)
            {
                ibin = iz*ncells_dim_[1]*ncells_dim_[0] + iy*ncells_dim_[0] + ix;

                if (triclinic_) {
                  center_[ibin][0] = lo_lamda_[0] + (static_cast<double>(ix)+0.5) * cell_size_lamda_[0];
                  center_[ibin][1] = lo_lamda_[1] + (static_cast<double>(iy)+0.5) * cell_size_lamda_[1];
                  center_[ibin][2] = lo_lamda_[2] + (static_cast<double>(iz)+0.5) * cell_size_lamda_[2];
                  domain->lamda2x(center_[ibin],center_[ibin]);

                } else {
                    center_[ibin][0] = lo_[0] + (static_cast<double>(ix)+0.5) * cell_size_[0];
                    center_[ibin][1] = lo_[1] + (static_cast<double>(iy)+0.5) * cell_size_[1];
                    center_[ibin][2] = lo_[2] + (static_cast<double>(iz)+0.5) * cell_size_[2];
                }
            }
        }
    }
//printf("\nCells center coordinates  computed\n");
    // calculate weight_[icell]
    if(!region_)
    {
//printf("REGION NOT DEFINED, ALL WEIGHTS = 1\n");
        for(int icell = 0; icell < ncells_max_; icell++)
            weight_[icell] = 1.;
    }
    // MC calculation if region_ shape is known by lammps, then it has the attribute match()
    if(region_)
    {
//printf("REGION DEFINED\n");
        double x_try[3];
        int ibin;
        int ntry = ncells_ * ntry_per_cell(); // number of MC tries - in .h file defined :  ntry_per_cell()=50
        double contribution = 1./static_cast<double>(ntry_per_cell());  // contrib of each try

        for(int icell = 0; icell < ncells_max_; icell++)
            weight_[icell] = 0.;

        for(int itry = 0; itry < ntry; itry++)
        {
            x_try[0] = lo_[0]+(hi_[0]-lo_[0])*random_->uniform();
            x_try[1] = lo_[1]+(hi_[1]-lo_[1])*random_->uniform();
            x_try[2] = lo_[2]+(hi_[2]-lo_[2])*random_->uniform();
            if(region_->match(x_try[0],x_try[1],x_try[2]))
            {
                ibin = coord2bin(x_try);
                // only do this for points in my subbox
                if(ibin >= 0)
                    weight_[ibin] += contribution;
            }
        }
//printf("\nCell weight calculated\n");

        // allreduce weights
        MPI_Sum_Vector(weight_,ncells_,world);

        // limit weight to 1
        for(int icell = 0; icell < ncells_max_; icell++)
            if(weight_[icell] > 1.) weight_[icell] = 1.; printf("\nLimiting weight\n");
    }

    // print to screen and log
    
    if (comm->me == 0)
    {
//        if (screen)
//	{
//	fprintf(screen,"Euler cell size on proc %d = %f (%d x %d x %d grid)\n",
//            comm->me,cell_size_ideal_,ncells_dim_[0],ncells_dim_[1],ncells_dim_[2]);
//	}
        if (logfile) fprintf(logfile,"Euler cell size on proc %d = %f (%d x %d x %d grid)\n",
            comm->me,cell_size_ideal_,ncells_dim_[0],ncells_dim_[1],ncells_dim_[2]);
    }
//   printf("\nFixAveEuler::setup_bins  -  Terminated\n"); 
}

/* ---------------------------------------------------------------------- */

void FixAveEuler::end_of_step()
{
//   printf("\nFixAveEuler::end_of_step\n"); 
    // have to adapt grid if box size changes
    if(box_change_size_ || (parallel_ && box_change_domain_))
    {
        if(region_)
            error->warning(FLERR,"Fix ave/euler using 'basevolume_region'"
                                "and changing simulation or load-balancing is huge over-head");
        setup_bins();
    }

//   printf("\nFixAveEuler::end_of_step  2\n"); 
    // bin atoms
    bin_atoms();

//   printf("\nFixAveEuler::end_of_step  3\n"); 
calculate_eu_sph();
    // calculate Eulerian grid properties
    // performs allreduce if necessary
//    calculate_eu();

//   printf("\nFixAveEuler::end_of_step  -  Concluded\n"); 
}

/* ---------------------------------------------------------------------- */

int FixAveEuler::ncells_pack()
{


//printf("\n\nFixAveEuler::ncells_pack\n\n");

    // in parallel mode, each proc will pack
    if (parallel_)
        return ncells_;

    // in serial mode, only proc 0 will pack
    if(0 == comm->me)
        return ncells_;
    else
        return 0;
}

/* ----------------------------------------------------------------------
   bin owned and ghost atoms
   this also implies we do not need to wrap around PBCs
   bin ghost atoms only if inside my grid
------------------------------------------------------------------------- */

void FixAveEuler::bin_atoms()
{
//printf("\n\nFixAveEuler::bin_atoms\n\n");

  int i,ibin;
  double **x = atom->x;
  int *mask = atom->mask;
  int nall = atom->nlocal + atom->nghost;

  for (i = 0; i < ncells_max_; i++)
    cellhead_[i] = -1;

  // re-alloc cellptr_ if necessary
  if(nall > ncellptr_max_)
  {
      ncellptr_max_ = nall;
      memory->grow(cellptr_,ncellptr_max_,"ave/pic:cellptr_");
  }

  // bin in reverse order so linked list will be in forward order
  // also use ghost atoms
  // skip if any atom is out of (sub) box

  for (i = nall-1; i >= 0; i--)
  {
      if(! (mask[i] & groupbit)) continue;

	/* bin of atom i*/
      ibin = coord2bin(x[i]);

      // particles outside grid may return values ibin < 0 || ibin >= ncells_
      // these are ignores
      if (ibin < 0 || ibin >= ncells_) {
        continue;
      }

      cellptr_[i] = cellhead_[ibin];
      cellhead_[ibin] = i;
  }
//printf("\n\nFixAveEuler::bin_atoms  -  Completed\n\n");

}

/*	Periodic Boundary conditions on cells, if not periodic and cell outside box it returns -1 */

void FixAveEuler::pbc(int &nx, int &ny, int &nz)
{
	if(domain->xperiodic)
	{
	if( nx>=ncells_dim_[0])
	    nx-=ncells_dim_[0];
	if( nx < 0 )
	    nx+=ncells_dim_[0];
	}

	else
	{
	if( nx>=ncells_dim_[0])
	    nx  = -1;
	if( nx < 0 )
	    nx = -1;
	}


	if(domain->yperiodic)
	{
	if( ny>=ncells_dim_[1])
	    ny-=ncells_dim_[1];
	if( ny < 0 )
	    ny+=ncells_dim_[1];
	}

	else
	{
	if( ny>=ncells_dim_[1])
	    ny  = -1;
	if( ny < 0 )
	    ny = -1;
	}


	if(domain->zperiodic)
	{
	if( nz>=ncells_dim_[2])
	    nz-=ncells_dim_[2];
	if( nz < 0 )
	    nz+=ncells_dim_[2];
	}

	else
	{
	if( nz>=ncells_dim_[2])
	    nz  = -1;
	if( nz < 0 )
	    nz = -1;
	}
}

/* ----------------------------------------------------------------------
	This member computes eulerian data according to SPH paradigm.
	CAVEAT:
	- This is working properly only for periodic boundary conditions. In case of non periodic boundaries the code will try to assign the values to non exixsting cells outside the simulation box if a particle dists from a boundary less than the kernel radius.
	- It works only in 3 dimensions, as the original fix.
	- It was written thinking only to NON-triclinic box.

WORK TO DO
	1 compute position of jcell
	2 distance jcell-particle, hence kernel, hence contribution to vol frac, etc...
	3 check if nall is correct to be local+ghost
	4 carefullly check names
	5 add MPI_COMMANDS
	6 check values, make them printable via dump
 ----------------------------------------------------------------------*/

void FixAveEuler::calculate_eu_sph()
{
    int itmp=0;
    Matrix3d *F = (Matrix3d *) force->pair->extract("smd/tlsph/Fincr_ptr", itmp); 
	if (F == NULL) {
	error->all(FLERR, "fix_ave_euler could not access deformation tensor. Are the matching pair styles present?");}
    double *det_def_grad = (double *) force->pair->extract("smd/tlsph/detF_ptr", itmp); 
	if (det_def_grad == NULL) {
	error->all(FLERR, "fix_ave_euler could not access smd/tlsph/detF_ptr. Are the matching pair styles present?");}

//printf("DET  BEGIN :   %lf\n",det_def_grad[0]);

    Matrix3d *T = (Matrix3d *) force->pair->extract("smd/tlsph/stressTensor_ptr", itmp);
	if (T == NULL) {
	error->all(FLERR, "fix_ave_euler could not access stress tensors. Are the matching pair styles present?");}
    int nall = atom->nlocal + atom->nghost;
for(int i=0; i<nall; i++) 
	det_def_grad[i] = F[i].determinant();
//printf("DET  BEGIN :   %lf\n",det_def_grad[0]);
    double * const * const x = atom->x;
    double * const * const v = atom->v;
    double * const  vfrac = atom->vfrac;
//    double * const * const v = atom->v;
    int *mask = atom->mask;
    double * const  sph_radius = atom->radius;
    double * const radius = atom->contact_radius;
    double * const rmass = atom->rmass;
    const double * const volume = atom->vfrac;
//    double ** def_grad_part = atom->smd_data_9;
    double dx,dy,dz;
    double wf,wfd,r;
    int jcellx,jcelly,jcellz;
    int icellx,icelly,icellz;

double y[3];
int icell_arr[3];
double def_grad_part[9];
double stress_part[7];
double new_vol;
double von_mises_stress;
Matrix3d stress_deviator;

/* half number of cells along each direction for which the kernel of a particle is not zero. Equivalent to radius in numbver if cells */
    double ncells_sph[3];
    // wrap compute with clear/add
    modify->clearstep_compute();

    for(int i = 0; i < ncells_; i++)
    {
        ncount_[i] = 0;
        vectorZeroize3D(v_av_[i]);
        vol_fr_[i] = 0.;
        radius_[i] = 0.;
        mass_[i] = 0.;
        vectorZeroizeN(stress_[i],7);
        rho[i] = 0.;
        vectorZeroizeN(def_grad_cell[i],9);
        vectorZeroizeN(stress_[i],7);
    }
//printf("cell_size[0] =  %lf\n",cell_size_[0]); 
//printf("SPH_radius[0] =  %lf\n",sph_radius[0]); 
//printf("ncells_sph = %lf\t%d\n\n", static_cast<int>( sph_radius[0]*cell_size_inv_[0] + 1.),  ( sph_radius[0]*cell_size_inv_[0] + 1.) );


	/* start atom loop*/

  for (int iatom= 0; iatom < nall; iatom++)
  {
	def_grad_part[0] = F[iatom](0,0);
	def_grad_part[1] = F[iatom](0,1);
	def_grad_part[2] = F[iatom](0,2);
	def_grad_part[3] = F[iatom](1,0);
	def_grad_part[4] = F[iatom](1,1);
	def_grad_part[5] = F[iatom](1,2);
	def_grad_part[6] = F[iatom](2,0);
	def_grad_part[7] = F[iatom](2,1);
	def_grad_part[8] = F[iatom](2,2);

	stress_deviator = Deviator(T[iatom]);
	von_mises_stress =   sqrt(3. / 2.) * stress_deviator.norm();
	stress_part[0] = von_mises_stress;
	stress_part[1] = T[iatom](0,0);	// xx
	stress_part[2] = T[iatom](1,1);	// yy
	stress_part[3] = T[iatom](2,2);	// zz
	stress_part[4] = T[iatom](0,1);	// xy
	stress_part[5] = T[iatom](0,2);	// xz
	stress_part[6] = T[iatom](1,2);	// yz

        if(! (mask[iatom] & groupbit)) continue;

	for(int dim = 0 ;  dim < 3 ; dim++)
	{
	  ncells_sph[dim] = static_cast<int>( sph_radius[iatom]*cell_size_inv_[dim] + 1.);
	}

//	printf("\n\n===================\niatom x y z\n");
//	printf("%d\t%lf\t%lf\t%lf\t\n",iatom,x[iatom][0],x[iatom][1],x[iatom][2]);
	/* cell at the center of the volume we are considering */
	int icell = coord2bin(x[iatom]);

	// particles outside grid may return values ibin < 0 || ibin >= ncells_
        // these are ignores
        if (icell < 0 || icell >= ncells_) {
	continue;
	}

//	printf("icell %d\n",icell);

	/* Loop over neighbour cells in SPH radius */
	 for( double  i = -ncells_sph[0]-1 ; i < ncells_sph[0]+1 ; i++){
		dx =  i*cell_size_[0];
	   for( double  j = -ncells_sph[1]-1  ; j < ncells_sph[1]+1 ; j++){
		dy =  j*cell_size_[1];
	     for( double k = -ncells_sph[2]-1 ; k < ncells_sph[1]+1 ; k++){
		dz =  k*cell_size_[2];
		/* check if cell inside SPH radius */
		if ( (dx*dx +dy*dy + dz*dz) > (sph_radius[iatom]*sph_radius[iatom]) )
			continue;

		/* determining x,y,z indeces of i cell containing i particle */
		icellz = static_cast<int>( icell/(ncells_dim_[0]*ncells_dim_[1]) );
		icelly = static_cast<int>( ( icell-icellz*ncells_dim_[1]*ncells_dim_[0] )/ncells_dim_[0] );
		icellx = icell - ncells_dim_[0]*ncells_dim_[1]*icellz - ncells_dim_[0]*icelly ;

		/* determination of indices of cell j */
		jcellx = icellx + (int)i;
		jcelly = icelly + (int)j;
		jcellz = icellz + (int)k;
		pbc(jcellx,jcelly,jcellz);
		/* If cell outside box simulation and no pbc along that direction*/
		if(jcellx==-1 || jcelly==-1 || jcellz==-1)
			continue;
		int jcell = ncells_dim_[0]*ncells_dim_[1]*jcellz + ncells_dim_[0]*jcelly + jcellx;
		
		/* computing distance particle-cell_center, using dx,dy,dz for a second purpose */
		dx = x[iatom][0] - center_[jcell][0];
		dy = x[iatom][1] - center_[jcell][1];
		dz = x[iatom][2] - center_[jcell][2];
//		printf("%lf\t- %lf\t= %lf\n",x[iatom][2],center_[jcell][2],dz);
		r = dx*dx + dy*dy + dz*dz;
		spiky_kernel_and_derivative(sph_radius[iatom],r,domain->dimension,wf,wfd);
//printf("WF = %lf\n", wf);		
		/* check compatibility with SMD */
		new_vol = vfrac[iatom]*det_def_grad[iatom];
//printf("VOL = %lf\n", vfrac[iatom] );
//printf("Det(F) = %lf\n", det_def_grad[iatom] );
//printf("NEW VOL = %lf\n", new_vol);		
		rho[jcell] += wf*rmass[iatom];
		mass_[jcell] += wf*rmass[iatom]*(new_vol); // volume in current configuration
		vol_fr_[jcell] += wf*new_vol*new_vol;
		ncount_[jcell] += wf*new_vol;

//		vol_fr_[jcell] = 1111;

		for(int l = 0; l < 9; l++)
		{
			def_grad_cell[jcell][l] += wf*def_grad_part[l]*new_vol;
			if( l < 7 )
			stress_[jcell][l] += wf*new_vol*stress_part[l];
//			if( l < 7 )
//			stress_[jcell][l] = 5555.0 ;
		}
	     } // end of z loop
 	   }  // end of y loop
	 }  // end of z loop

//printf("%d\t%d\t%d\n",icellx,icelly,icellz);
//printf("%d\t%d\t%d\n",icellx,icelly,icellz);

  }  // end of particle loop

    // allreduce contributions so far if not parallel
    if(!parallel_ && ncells_ > 0)
    {
        MPI_Sum_Vector(&(def_grad_cell[0][0]),9*ncells_,world);
        MPI_Sum_Vector(vol_fr_,ncells_,world);
        MPI_Sum_Vector(rho,ncells_,world);
        MPI_Sum_Vector(mass_,ncells_,world);
        MPI_Sum_Vector(ncount_,ncells_,world);
        MPI_Sum_Vector(&(stress_[0][0]),7*ncells_,world);
    }
}

/*	Returns lower side of the cell */
void FixAveEuler::bin2coord(int bin, double *x)
{
	int icell[3];
	icell[2] = static_cast<int>( bin/(ncells_dim_[0]*ncells_dim_[1]) );
	icell[1] = static_cast<int>( ( bin-icell[2]*ncells_dim_[1]*ncells_dim_[0] )/ncells_dim_[0] );
	icell[0] = bin - ncells_dim_[0]*ncells_dim_[1]*icell[2] - ncells_dim_[0]*icell[1] ;
	
	for(int dim = 0; dim < domain->dimension; dim++ )
	{
	  x[dim] = icell[dim]*cell_size_[dim];
	}
}

/*	Returns lower side of the cube */
void FixAveEuler::bin2coord(int *icell, double *x)
{
	for(int dim = 0; dim < domain->dimension; dim++ )
	{
	  x[dim] = icell[dim]*cell_size_[dim];
	}
}

/* ----------------------------------------------------------------------
   map coord to grid, also return ix,iy,iz indices in each dim
------------------------------------------------------------------------- */

inline int FixAveEuler::coord2bin(double *x)
{

//printf("\n\nFixAveEuler::coord2bin\n\n");

  int i,iCell[3];
  double float_iCell[3];

  if (triclinic_) {
    double tmp_x[3];
    domain->x2lamda(x,tmp_x);
    for (i=0;i<3;i++) {
      float_iCell[i] = (tmp_x[i]-lo_lamda_[i])*cell_size_lamda_inv_[i];
      iCell[i] = static_cast<int> (float_iCell[i] >= 0 ? float_iCell[i] : float_iCell[i]-1);
    }
  } else {
    for (i=0;i<3;i++) {

      // skip particles outside my subdomain
      
      if(x[i] <= domain->sublo[i] || x[i] >= domain->subhi[i])
        return -1;
      float_iCell[i] = (x[i]-lo_[i])*cell_size_inv_[i];
      iCell[i] = static_cast<int> (float_iCell[i]);
    }
  }
//printf("FixAveEuler::coord2bin  -  Returning\n");
//printf("%d\t%d\t%d\n",iCell[0],iCell[1],iCell[2]);
  return iCell[2]*ncells_dim_[1]*ncells_dim_[0] + iCell[1]*ncells_dim_[0] + iCell[0];
}

/* ----------------------------------------------------------------------
   calculate Eulerian data, use interpolation function
------------------------------------------------------------------------- */

void FixAveEuler::calculate_eu()
{
    //int ncount;
//  printf("FixAveEuler::calculate_eu  - initialization\n");
    double * const * const v = atom->v;
    double * const radius = atom->radius;
    double * const rmass = atom->rmass;

    double prefactor_vol_fr = 4./3.*M_PI/cell_volume_;
    double prefactor_stress = 1./cell_volume_;
    double vel_x_mass[3];
    #ifdef SUPERQUADRIC_ACTIVE_FLAG
    const double * const volume = atom->volume;
    const int superquadric_flag = atom->superquadric_flag;
    #endif
//printf("modify->clearstep_compute()\n");
    // wrap compute with clear/add
    modify->clearstep_compute();

//std::cout << INVOKED_PERATOM << std::endl;
//std::cout << compute_stress_ << std::endl;
//std::cout << compute_stress_->invoked_flag << std::endl;
    // invoke compute if not previously invoked
//printf("FixAveEuler::calculate_eu  -  Invoke compute\n");
//    
////std::cout << compute_stress_->invoked_flag << "\t" << INVOKED_PERATOM << std::endl;
//
//    if (!(compute_stress_->invoked_flag & INVOKED_PERATOM)) {
//printf("No compute_stress_->invoked_flag & INVOKED_PERATOM\n");
//std::cout << compute_stress_->invoked_flag << "\t" << INVOKED_PERATOM << std::endl;
//        compute_stress_->compute_peratom();
//        compute_stress_->invoked_flag |= INVOKED_PERATOM;
//    }
//
//printf("FixAveEuler::calculate_eu  -  forward comm\n");
//    // forward comm per-particle stress from compute so neighs have it
//    comm->forward_comm_compute(compute_stress_);
//
//    // need to get pointer here since compute_peratom() may realloc
//printf("FixAveEuler::calculate_eu  -  getting pointer to stress\n");
//    double **stress_atom = compute_stress_->array_atom;
//
    // loop all binned particles
    // each particle can contribute to the cell that it has been binned
    // optionally plus its 26 neighs

//printf("FixAveEuler::calculate_eu  -  Loop on all binned particles\n");
    for(int icell = 0; icell < ncells_; icell++)
    {
        ncount_[icell] = 0;
        vectorZeroize3D(v_av_[icell]);
        vol_fr_[icell] = 0.;
        radius_[icell] = 0.;
        mass_[icell] = 0.;
        vectorZeroizeN(stress_[icell],7);

        // skip if no particles in cell
        
        if(-1 == cellhead_[icell])
            continue;

        // add contributions of particle - v and volume fraction
        // v is favre-averaged (mass-averaged)
        // radius is number-averaged

        for(int iatom = cellhead_[icell/*+stencil*/]; iatom >= 0; iatom = cellptr_[iatom])
        {
//std::cout << "iatom\t"  <<iatom << std::endl;
//std::cout << "cellptr_[iatom]\t" << cellptr_[iatom] << std::endl;

            vectorScalarMult3D(v[iatom],rmass[iatom],vel_x_mass);
            vectorAdd3D(v_av_[icell],vel_x_mass,v_av_[icell]);
            double r = radius[iatom];
            #ifdef SUPERQUADRIC_ACTIVE_FLAG
            if(superquadric_flag)
                r = cbrt(0.75 * volume[iatom] / M_PI);
            #endif

            vol_fr_[icell] += r*r*r;
            radius_[icell] += r;
            mass_[icell] += rmass[iatom];
            ncount_[icell]++;
        }  // end of particle loop

    } // end of cell loop

    // allreduce contributions so far if not parallel
    if(!parallel_ && ncells_ > 0)
    {
        MPI_Sum_Vector(&(v_av_[0][0]),3*ncells_,world);
        MPI_Sum_Vector(vol_fr_,ncells_,world);
        MPI_Sum_Vector(radius_,ncells_,world);
        MPI_Sum_Vector(mass_,ncells_,world);
        MPI_Sum_Vector(ncount_,ncells_,world);
    }

    // perform further calculations

    double eps_ntry = 1./static_cast<double>(ntry_per_cell());
    for(int icell = 0; icell < ncells_; icell++)
    {
        // calculate average vel and radius
        if(ncount_[icell]) vectorScalarDiv3D(v_av_[icell],mass_[icell]);
        if(ncount_[icell]) radius_[icell]/=static_cast<double>(ncount_[icell]);

        // calculate volume fraction
        //safety check, add an epsilon to weight if any particle ended up in that cell
        if(vol_fr_[icell] > 0. && MathExtra::compDouble(weight_[icell],0.,1e-6))
           weight_[icell] = eps_ntry;
        if(weight_[icell] < eps_ntry )
            vol_fr_[icell] = 0.;
        else
            vol_fr_[icell] *= prefactor_vol_fr/weight_[icell];
        
        // add contribution of particle - stress
        // need v before can calculate stress
        // stress is molecular diffusion + contact forces

//        for(int iatom = cellhead_[icell/*+stencil*/]; iatom >= 0; iatom = cellptr_[iatom])
//        {
//            stress_[icell][1] += -rmass[iatom]*(v[iatom][0]-v_av_[icell][0])*(v[iatom][0]-v_av_[icell][0]) + stress_atom[iatom][0];
//            stress_[icell][2] += -rmass[iatom]*(v[iatom][1]-v_av_[icell][1])*(v[iatom][1]-v_av_[icell][1]) + stress_atom[iatom][1];
//            stress_[icell][3] += -rmass[iatom]*(v[iatom][2]-v_av_[icell][2])*(v[iatom][2]-v_av_[icell][2]) + stress_atom[iatom][2];
//            stress_[icell][4] += -rmass[iatom]*(v[iatom][0]-v_av_[icell][0])*(v[iatom][1]-v_av_[icell][1]) + stress_atom[iatom][3];
//            stress_[icell][5] += -rmass[iatom]*(v[iatom][0]-v_av_[icell][0])*(v[iatom][2]-v_av_[icell][2]) + stress_atom[iatom][4];
//            stress_[icell][6] += -rmass[iatom]*(v[iatom][1]-v_av_[icell][1])*(v[iatom][2]-v_av_[icell][2]) + stress_atom[iatom][5]; 
//        }
//        stress_[icell][0] = -0.333333333333333*(stress_[icell][1]+stress_[icell][2]+stress_[icell][3]);
        if(weight_[icell] < eps_ntry)
            vectorZeroizeN(stress_[icell],7);
        else
            vectorScalarMultN(7,stress_[icell],prefactor_stress/weight_[icell]);
    }

    // allreduce stress if not parallel
//printf("FixAveEuler::calculate_eu  -  MPI_SUM\n");
    if(!parallel_ && ncells_ > 0)
    {
        MPI_Sum_Vector(&(stress_[0][0]),7*ncells_,world);

        // recalc pressure based on allreduced stress
//        for(int icell = 0; icell < ncells_; icell++)
//            stress_[icell][0] = -0.333333333333333*(stress_[icell][1]+stress_[icell][2]+stress_[icell][3]);
    }

    // wrap with clear/add
    modify->addstep_compute(update->ntimestep + exec_every_);
//printf("FixAveEuler::calculate_eu  -  End\n");
}

/* ----------------------------------------------------------------------
   return I,J array value
   if I exceeds current bins, return 0.0 instead of generating an error
   column 1,2,3 = bin coords, next column = vol fr,
   remaining columns = vel, stress, radius
------------------------------------------------------------------------- */

double FixAveEuler::compute_array(int i, int j)
{

//printf("\n\nFixAveEuler::compute_array\n\n");

  if(i >= ncells_) return 0.0;

  else if(j < 3) return center_[i][j];
  else if(j == 3) return vol_fr_[i];
  else if(j < 7) return v_av_[i][j-4];
  else if(j == 7) return stress_[i][0];
  else if(j < 14) return stress_[i][j-7];
  else if(j < 15) return radius_[i];
  else return 0.0;
}
