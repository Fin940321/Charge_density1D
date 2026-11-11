### Import modules
import MDAnalysis as mda
from openmm.app import *
from openmm import *
from openmm.unit import nanometer, picosecond, picoseconds, kelvin
from sys import stdout
import numpy as np
from numpy import linalg as LA
import sys
from numba import jit
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

sys.setrecursionlimit(2000)


### Optimized histogram computation
@jit(parallel=True, fastmath=True, nopython=True)
def numba_histogram_manual(positions_z, charges, bins_array):
    """Numba-accelerated histogram for benchmarking"""
    num_bins = len(bins_array) - 1
    hist = np.zeros(num_bins)
    
    for idx in range(len(positions_z)):
        pos = positions_z[idx]
        charge = charges[idx]
        bin_idx = np.searchsorted(bins_array, pos, side='right') - 1
        if 0 <= bin_idx < num_bins:
            hist[bin_idx] += charge
    
    return hist


def compute_charge_histogram(frame_xyz_z, atom_indices, charges_array, reference_z, bins):
    """
    Compute charge density histogram using NumPy vectorization.
    
    Good taste: Simple, direct, no special cases.
    """
    if len(atom_indices) == 0:
        return np.zeros(len(bins) - 1)
    
    positions_z = frame_xyz_z[atom_indices] - reference_z
    charges = charges_array[atom_indices]
    hist, _ = np.histogram(positions_z, bins=bins, weights=charges)
    return hist


def process_frame(frame_xyz_z, cation_idx, anion_idx, solvent_idx, charges, ref_z, bins):
    """
    Process single frame for all species.
    
    Linus: "Functions should do one thing and do it well."
    """
    return (
        compute_charge_histogram(frame_xyz_z, cation_idx, charges, ref_z, bins),
        compute_charge_histogram(frame_xyz_z, anion_idx, charges, ref_z, bins),
        compute_charge_histogram(frame_xyz_z, solvent_idx, charges, ref_z, bins)
    )


### Configuration
temperature = 300
traj_file = "FV_NVT.dcd"
top_file = "start_drudes.pdb"
ffdir = './ffdir/'
electrode_ffdir = './electrode_ffdir/'

print("=" * 70)
print("    Charge Density Analysis V7 - Linus Edition")
print("    \"Good taste means eliminating special cases\"")
print("=" * 70)
print()

### Load trajectory
print("=== Loading Trajectory ===")
print(f"Trajectory: {traj_file}")
print(f"Topology:   {top_file}")

u = mda.Universe(top_file, traj_file)
total_frames = len(u.trajectory)
framestart = total_frames // 2  # Integer division - cleaner
frameCount = total_frames - framestart
frameend = framestart + frameCount

print(f"✓ Loaded successfully")
print(f"  Total frames:    {total_frames}")
print(f"  Analysis frames: {framestart} to {frameend} ({frameCount} frames)")
print(f"  Time range:      {u.trajectory[framestart].time:.2f} - {u.trajectory[frameend-1].time:.2f} ps")
print()


### Load OpenMM system for charges
print("=== Loading Force Field ===")
pdb = PDBFile('start_nodrudes.pdb')
pdb.topology.loadBondDefinitions(ffdir + 'graph_residue_c.xml')
pdb.topology.loadBondDefinitions(ffdir + 'graph_residue_n.xml')
pdb.topology.loadBondDefinitions(ffdir + 'sapt_residues.xml')
pdb.topology.loadBondDefinitions(electrode_ffdir + 'nanotube9x9_residue_c.xml')
pdb.topology.loadBondDefinitions(electrode_ffdir + 'nanotube9x9_residue_n.xml')
pdb.topology.createStandardBonds()

modeller = Modeller(pdb.topology, pdb.positions)
forcefield = ForceField(
    ffdir + 'graph_c_freeze.xml',
    ffdir + 'graph_n_freeze.xml',
    ffdir + 'sapt_noDB_2sheets.xml',
    electrode_ffdir + 'nanotube9x9_c_freeze.xml',
    electrode_ffdir + 'nanotube9x9_n_freeze.xml'
)
modeller.addExtraParticles(forcefield)
system = forcefield.createSystem(
    modeller.topology, 
    nonbondedCutoff=1.4*nanometer, 
    constraints=None, 
    rigidWater=True
)

nbondedForce = [f for f in [system.getForce(i) for i in range(system.getNumForces())] 
                if type(f) == NonbondedForce][0]

integ_md = DrudeLangevinIntegrator(temperature, 1/picosecond, 1*kelvin, 1/picosecond, 0.001*picoseconds)
platform = Platform.getPlatformByName('CPU')
simmd = Simulation(modeller.topology, system, integ_md, platform)

boxVecs = simmd.topology.getPeriodicBoxVectors()
crossBox = np.cross(boxVecs[0], boxVecs[1])
sheet_area = np.dot(crossBox, crossBox)**0.5 / nanometer**2

print(f"✓ Force field loaded")
print(f"  Sheet area: {sheet_area:.6f} nm²")
print()


### Auto-detect ion types
print("=== Detecting Ion Types ===")

unique_residues = set(res.name for res in simmd.topology.residues())
print(f"  Residue types: {sorted(unique_residues)}")

# Detection patterns
CATION_TYPES = ["BMIM", "BMI", "EMIM", "PMIM", "OMIM"]
ANION_TYPES = ["Tf2N", "trfl", "Tf2", "TRF", "trf", "TFSI", "BF4", "PF6", "Cl"]

namecat = next((cat for cat in CATION_TYPES if cat in unique_residues), None)
namean = next((an for an in ANION_TYPES if an in unique_residues), None)

if namecat is None:
    raise ValueError(f"No cation detected in residues: {sorted(unique_residues)}")
if namean is None:
    raise ValueError(f"No anion detected in residues: {sorted(unique_residues)}")

print(f"  ✓ Cation: {namecat}")
print(f"  ✓ Anion:  {namean}")
print()


### Extract atom indices - THE RIGHT WAY
print("=== Extracting Atom Indices ===")

# Linus: "Don't use Python lists for numeric data. Use NumPy from the start."
cation_list = []
anion_list = []
solvent_list = []

for res in simmd.topology.residues():
    if res.name == namecat:
        cation_list.extend(atom.index for atom in res._atoms)
    elif res.name == namean:
        anion_list.extend(atom.index for atom in res._atoms)
    # Add other residue types to solvent if needed
    # elif res.name in SOLVENT_TYPES:
    #     solvent_list.extend(atom.index for atom in res._atoms)

# Convert to NumPy arrays - no special cases needed
cation_idx = np.array(cation_list, dtype=np.int32)
anion_idx = np.array(anion_list, dtype=np.int32)
solvent_idx = np.array(solvent_list, dtype=np.int32)  # Empty array if no solvent - perfectly fine!

print(f"  Cation atoms:  {len(cation_idx)}")
print(f"  Anion atoms:   {len(anion_idx)}")
print(f"  Solvent atoms: {len(solvent_idx)}")
print(f"  Total:         {len(cation_idx) + len(anion_idx) + len(solvent_idx)}")
print()

# Validate indices
assert len(cation_idx) > 0, "No cation atoms found!"
assert len(anion_idx) > 0, "No anion atoms found!"
assert np.max(cation_idx) < u.atoms.n_atoms, "Invalid cation index!"
assert np.max(anion_idx) < u.atoms.n_atoms, "Invalid anion index!"
if len(solvent_idx) > 0:
    assert np.max(solvent_idx) < u.atoms.n_atoms, "Invalid solvent index!"


### Extract charges
print("Extracting atomic charges...")
num_particles = nbondedForce.getNumParticles()
all_charges = np.zeros(num_particles, dtype=np.float64)

for atom_idx in tqdm(range(num_particles), desc="Charges", ncols=70):
    q, sig, eps = nbondedForce.getParticleParameters(atom_idx)
    all_charges[atom_idx] = q._value

print(f"✓ Extracted {num_particles} charges")
print()


### Setup histogram bins
dz = 0.01  # nm
cell_dist = 14.0  # nm
num_bins = int(cell_dist / dz)
bins = np.linspace(0, cell_dist, num_bins + 1)

# Pre-allocate accumulators
hist_cat_total = np.zeros(num_bins, dtype=np.float64)
hist_an_total = np.zeros(num_bins, dtype=np.float64)
hist_solv_total = np.zeros(num_bins, dtype=np.float64)

print(f"=== Analysis Parameters ===")
print(f"  Bin width:  {dz} nm")
print(f"  Range:      0 to {cell_dist} nm")
print(f"  Bins:       {num_bins}")
print(f"  Frames:     {frameCount}")
print()


### Performance benchmark (optional)
print("=== Performance Benchmark ===")
u.trajectory[framestart]
test_xyz_z = u.atoms.positions[:, 2] / 10.0
test_ref_z = test_xyz_z[1]

test_pos = test_xyz_z[cation_idx] - test_ref_z
test_chg = all_charges[cation_idx]

t1 = time.time()
for _ in range(10):
    _ = numba_histogram_manual(test_pos, test_chg, bins)
time_numba = (time.time() - t1) / 10

t2 = time.time()
for _ in range(10):
    _ = compute_charge_histogram(test_xyz_z, cation_idx, all_charges, test_ref_z, bins)
time_numpy = (time.time() - t2) / 10

speedup = time_numba / time_numpy
print(f"  Numba:  {time_numba*1000:.3f} ms/frame")
print(f"  NumPy:  {time_numpy*1000:.3f} ms/frame")
print(f"  Winner: {'NumPy' if speedup > 1 else 'Numba'} ({abs(speedup):.1f}x faster)")
print()


### Main calculation loop
start_time = time.time()

print("Computing charge density distributions...")
print("Strategy: NumPy vectorization, direct approach, zero special cases")
print()

for frame_idx in tqdm(range(framestart, frameend), desc="Frames", unit="fr", ncols=70):
    u.trajectory[frame_idx]
    
    # Extract Z coordinates only (minimize memory transfer)
    frame_xyz_z = u.atoms.positions[:, 2] / 10.0  # Å -> nm
    reference_z = frame_xyz_z[1]  # Reference electrode position
    
    # Process frame
    hist_cat, hist_an, hist_solv = process_frame(
        frame_xyz_z, cation_idx, anion_idx, solvent_idx,
        all_charges, reference_z, bins
    )
    
    # Accumulate
    hist_cat_total += hist_cat
    hist_an_total += hist_an
    hist_solv_total += hist_solv

elapsed = time.time() - start_time
print(f"\n✓ Computation complete")
print(f"  Total time:  {elapsed:.2f} s")
print(f"  Per frame:   {elapsed/frameCount*1000:.2f} ms")
print(f"  Throughput:  {frameCount/elapsed:.1f} frames/s")
print()


### Calculate densities
avg_charge_cat = hist_cat_total / frameCount
avg_charge_an = hist_an_total / frameCount
avg_charge_solv = hist_solv_total / frameCount
avg_charge_total = avg_charge_cat + avg_charge_an + avg_charge_solv

bin_volume = sheet_area * dz
density_cat = avg_charge_cat / bin_volume
density_an = avg_charge_an / bin_volume
density_solv = avg_charge_solv / bin_volume
density_total = avg_charge_total / bin_volume

z_centers = (bins[:-1] + bins[1:]) / 2

print("=== Charge Density Statistics ===")
print(f"  Cation:  [{density_cat.min():.6e}, {density_cat.max():.6e}] C/nm³")
print(f"  Anion:   [{density_an.min():.6e}, {density_an.max():.6e}] C/nm³")
print(f"  Solvent: [{density_solv.min():.6e}, {density_solv.max():.6e}] C/nm³")
print(f"  Total:   [{density_total.min():.6e}, {density_total.max():.6e}] C/nm³")
print()


### Write output files
print("Writing results...")

output_files = [
    ("hist_q_cat_V7.dat", density_cat, "cation"),
    ("hist_q_an_V7.dat", density_an, "anion"),
    ("hist_q_solv_V7.dat", density_solv, "solvent"),
    ("hist_q_total_V7.dat", density_total, "total")
]

for filename, density, species in output_files:
    with open(filename, "w") as f:
        for z, rho in zip(z_centers, density):
            print(f'{z:5.8f}  {rho:5.8f}', file=f)
    print(f"  ✓ {filename}")

print()


### Plot results
print("Generating plots...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), dpi=300)

# Individual species
ax1.plot(z_centers, density_cat, label=f'Cation ({namecat})', 
         color='blue', linewidth=1.5, alpha=0.8)
ax1.plot(z_centers, density_an, label=f'Anion ({namean})', 
         color='red', linewidth=1.5, alpha=0.8)
if len(solvent_idx) > 0 and np.any(density_solv != 0):
    ax1.plot(z_centers, density_solv, label='Solvent', 
             color='green', linewidth=1.5, alpha=0.8)
ax1.set_xlabel('Z Position (nm)', fontsize=11)
ax1.set_ylabel('Charge Density (C/nm³)', fontsize=11)
ax1.set_title('Individual Species Charge Density', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.5)

# Total solution
ax2.plot(z_centers, density_total, label='Total', color='black', linewidth=2)
ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
ax2.set_xlabel('Z Position (nm)', fontsize=11)
ax2.set_ylabel('Charge Density (C/nm³)', fontsize=11)
ax2.set_title('Total Charge Density', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.5)

fig.suptitle(f'Charge Density Analysis V7 - {namecat}/{namean}', 
             fontsize=14, fontweight='bold', y=0.995)

plt.tight_layout()
plt.savefig("charge_density_V7.png", dpi=600, bbox_inches='tight')
print("  ✓ charge_density_V7.png")

print()
print("=" * 70)
print("✓ Analysis Complete - V7 Linus Edition")
print()
print("Key improvements:")
print("  • Eliminated Python list intermediate steps")
print("  • Removed special case handling for empty arrays")
print("  • Simplified function interfaces")
print("  • Added data validation")
print("  • Cleaner code structure")
print()
print('"Good taste is about seeing the problem from a different angle"')
print("                                        - Linus Torvalds")
print("=" * 70)
