import mdtraj as mdtraj
# for some reason we have to import mdtraj before openmm, not sure why...
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
import numpy as np
from numpy import linalg as LA
from copy import deepcopy
import sys
sys.setrecursionlimit(2000)

class hist_charges:
    def __init__(self, dz, zdim, zlist, chargelist, sheet_area):
        self.sheet_area = sheet_area
        self.dz = dz
        self.zdim = zdim
        self.bins = [i*self.dz for i in range(0, int(self.zdim/self.dz))]
        self.zlist = zlist
        self.chargelist = chargelist
        self.Qcount_i = []
    def q_count(self):
        for bin_i in range(len(self.bins)-1):
            bin0 = self.bins[bin_i]
            bin1 = self.bins[bin_i+1]
            ztotal = [self.chargelist[i] for i,x in enumerate(self.zlist) if bin0 < x <= bin1]
            #avg_count = sum(ztotal)/len(ztotal)/(self.sheet_area * self.dz) if len(ztotal) != 0 else 0
            avg_count = sum(ztotal)/(self.sheet_area * self.dz)

            self.Qcount_i.append( avg_count )

        return self.Qcount_i

class get_charge:
    def __init__(self, atomlist):
        self.atomlist = atomlist
        self.position_z = []
        self.charges = []
        #C1_grpcA =  frame.xyz[i, 1 , 2]
    def pointcharge(self):
        for atom_i in range(len(self.atomlist)):
            atom_idx = self.atomlist[atom_i]
            C1_grpcA =  frame.xyz[i, 1 , 2]
            pos_z = frame.xyz[i, atom_idx, 2] - float(C1_grpcA)
            (q, sig, eps) = nbondedForce.getParticleParameters(atom_idx)
            self.position_z.append(pos_z)
            self.charges.append( q._value )
 
        return self.position_z, self.charges


namecat = 'BMIM'
namean = 'BF4'
namesolv= 'trfl'
namegrp = 'grpc'

temperature=300
#here, we load pdb with drude oscillators
traj='md_nvt300.dcd'
top='start_drudes300.pdb'

framestart = 5000
frameCount = 5000
frameend = framestart + frameCount
#framestart = 49000
frame = mdtraj.load(traj, top=top)
#frameend = frame.n_frames

# use OpenMM library to get charges
# here we load pdb without drude oscillators
pdb = PDBFile('start_nodrudes300.pdb')
pdb.topology.loadBondDefinitions('graph_residue_c.xml')
pdb.topology.loadBondDefinitions('graph_residue_n.xml')
#pdb.topology.loadBondDefinitions('graph_residue_s.xml')
pdb.topology.loadBondDefinitions('sapt_residues.xml')
pdb.topology.createStandardBonds();
modeller = Modeller(pdb.topology, pdb.positions)
#forcefield = ForceField('graph_c_freeze.xml','graph_c_freeze.xml','sapt_noDB.xml')
forcefield = ForceField('graph_c_freeze.xml','graph_n_freeze.xml','sapt_noDB_2sheets.xml')
#forcefield = ForceField('graph_c_freeze.xml','graph_n_freeze.xml','graph_s_freeze.xml','sapt_noDB_2sheets.xml')
modeller.addExtraParticles(forcefield)
system = forcefield.createSystem(modeller.topology, nonbondedCutoff=1.4*nanometer, constraints=None, rigidWater=True)
nbondedForce = [f for f in [system.getForce(i) for i in range(system.getNumForces())] if type(f) == NonbondedForce][0]

# this is not important, this is just to create openmm object that we can use to access topology
integ_md = DrudeLangevinIntegrator(temperature, 1/picosecond, 1*kelvin, 1/picosecond, 0.001*picoseconds)
platform = Platform.getPlatformByName('CPU')
simmd = Simulation(modeller.topology, system, integ_md, platform)
#vec_x = u.trajectory[startFrame].triclinic_dimensions[0]
#vec_y = u.trajectory[startFrame].triclinic_dimensions[1]
#area = LA.norm( np.cross(vec_x, vec_y) )
boxVecs = simmd.topology.getPeriodicBoxVectors()
crossBox = np.cross(boxVecs[0], boxVecs[1])
sheet_area = np.dot(crossBox, crossBox)**0.5 / nanometer**2
print(sheet_area)

charges=[]
# get indices solvent molecules
cation = []
anion = []
solvent = []
for res in simmd.topology.residues():
    if res.name == namecat:
        for atom in res._atoms:
            cation.append(atom.index)
    if res.name == namean:
        for atom in res._atoms:
            anion.append(atom.index)
    if res.name == namesolv:
        for atom in res._atoms:
            solvent.append(atom.index)

solution =  deepcopy(cation)
solution.extend(deepcopy(anion))
solution.extend(deepcopy(solvent))
print(len(solution))
#print(len(solvent))
#print(charges, len(charges))

dz = 0.01
cell_dist = 14.
zbins = [0.0 for i in range(1, int(cell_dist/dz))]
allQcat_i = np.array(zbins)
allQan_i = np.array(zbins)
allQsolv_i = np.array(zbins)
allQsolution_i = np.array(zbins)
count_i = np.array(zbins)
#frame = mdtraj.load(traj, top=top)
for i in range(framestart, frameend ):
    Q_cation = get_charge( cation )
    cat_pos_z, cat_charges = Q_cation.pointcharge()
    Q_anion = get_charge( anion )
    an_pos_z, an_charges = Q_anion.pointcharge()
    Q_solvent = get_charge( solvent )
    solv_pos_z, solv_charges = Q_solvent.pointcharge()
    Q_solution = get_charge( solution )
    solution_pos_z, solution_charges = Q_solution.pointcharge()
#    solv_pos_z = []
#    solv_charges = []
#    for j in range(len(solvent)):
#        solv_idx = solvent[j]
#        pos_z = frame.xyz[i,solv_idx,2]
#        (q, sig, eps) = nbondedForce.getParticleParameters(solv_idx)
#        solv_pos_z.append(pos_z)
#        solv_charges.append( q._value )
    hist_cat_i = hist_charges(dz, cell_dist, cat_pos_z, cat_charges, sheet_area)
    Qcat_count_i =  hist_cat_i.q_count()
    allQcat_i = allQcat_i + np.array(Qcat_count_i)

    hist_an_i = hist_charges(dz, cell_dist, an_pos_z, an_charges, sheet_area)
    Qan_count_i =  hist_an_i.q_count()
    allQan_i = allQan_i + np.array(Qan_count_i)

    hist_solv_i = hist_charges(dz, cell_dist, solv_pos_z, solv_charges, sheet_area)
    Qsolv_count_i =  hist_solv_i.q_count()
    allQsolv_i = allQsolv_i + np.array(Qsolv_count_i)

    hist_solution_i = hist_charges(dz, cell_dist, solution_pos_z, solution_charges, sheet_area )
    Qsolution_count_i =  hist_solution_i.q_count()
    allQsolution_i = allQsolution_i + np.array( Qsolution_count_i )


with open("hist_q_cat.dat", "w") as ofile_cat:
    for i in range(0,len(allQcat_i)):
        Q_cat = allQcat_i[i]/frameCount
        print('{0:5.8f}  {1:5.8f}'.format( i*dz, Q_cat), file = ofile_cat)
        #line_q = str("{0:5.8f}".format(float(i*dz)) + "  " + str("{0:5.8f}".format(float(Q_cat))) + "\n"
        #ofile_cat.write(line_q)

with open("hist_q_an.dat", "w") as ofile_an:
    for i in range(0,len(allQan_i)):
        Q_an = allQan_i[i]/frameCount
        print('{0:5.8f}  {1:5.8f}'.format( i*dz, Q_an), file = ofile_an)
        #line_q = str("{0:5.8f}".format(float(i*dz)) + "  " + str("{0:5.8f}".format(float(Q_an))) + "\n"
        #ofile_an.write(line_q)

with open("hist_q_solv.dat", "w") as ofile_solv:
    for i in range(0,len(allQsolv_i)):
        Q_solv = allQsolv_i[i]/frameCount
        print('{0:5.8f}  {1:5.8f}'.format( i*dz, Q_solv), file = ofile_solv)
        #print( i*dz, Q_solv)
        #line_q = str("{0:5.8f}".format(float(i*dz)) + "  " + str("{0:5.8f}".format(float(Q_solv))) + "\n"
        #ofile_solv.write(line_q)
        
with open("hist_q_solution.dat", "w") as ofile_solution:
    for i in range(0,len(allQsolution_i)):
        Q_solution = allQsolution_i[i]/frameCount
        print('{0:5.8f}  {1:5.8f}'.format( i*dz, Q_solution), file = ofile_solution)
        #print( i*dz, Q_solv)
        #line_q = str("{0:5.8f}".format(float(i*dz)) + "  " + str("{0:5.8f}".format(float(Q_solution))) + "\n"
        #ofile_solution.write(line_q)

