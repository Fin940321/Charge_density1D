### Import modules
import MDAnalysis as mda
from openmm.app import *
from openmm import *
from openmm.unit import nanometer, picosecond, picoseconds, kelvin
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt


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
electrode_pdb = "start_nodrudes.pdb"  # For electrode detection
ffdir = './ffdir/'
electrode_ffdir = './electrode_ffdir/'

print("=" * 70)
print("    Charge Density Analysis V8 - Auto-Detection Edition")
print("    \"Good taste means eliminating magic numbers\"")
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


### Detect electrode boundaries (NEW in V8)
print("=== Detecting Electrode Boundaries ===")
print(f"  Loading electrode structure: {electrode_pdb}")

# Load electrode structure with MDAnalysis
u_electrode = mda.Universe(electrode_pdb)
# Select only grpc (electrode carbon positions, not force field grph)
electrode = u_electrode.select_atoms("resname grpc")

if len(electrode) == 0:
    raise ValueError(
        f"No electrode atoms found with resname 'grpc'\n"
        f"Available residues: {set(res.resname for res in u_electrode.residues)}"
    )

# Get Z positions of electrode atoms (in Angstroms)
z_positions_electrode = electrode.positions[:, 2]
z_min_angstrom = z_positions_electrode.min()
z_max_angstrom = z_positions_electrode.max()
z_range_angstrom = z_max_angstrom - z_min_angstrom

# Convert to nanometers (Å → nm)
z_min_nm = z_min_angstrom / 10.0
z_max_nm = z_max_angstrom / 10.0
cell_dist = z_range_angstrom / 10.0  # This replaces the hard-coded 14.0 nm

print(f"  ✓ Electrode atoms (grpc): {len(electrode)}")
print(f"  ✓ Z range (Å):            {z_min_angstrom:.2f} to {z_max_angstrom:.2f} Å")
print(f"  ✓ Z range (nm):           {z_min_nm:.4f} to {z_max_nm:.4f} nm")
print(f"  ✓ Electrode spacing:      {cell_dist:.4f} nm")
print()


### Load OpenMM system for charges
print("=== Loading Force Field ===")
pdb = PDBFile(electrode_pdb)
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


### Setup histogram bins (NEW calculation in V8)
print("=== Setting Up Histogram Bins ===")

# Calculate number of bins to maintain ~0.01 nm bin width
target_dz = 0.01  # nm (desired bin width)
num_bins = int(cell_dist / target_dz)

# Recalculate actual dz from the detected electrode spacing
# This follows the same approach as ion_density1D_V3.py
dz = (z_max_nm - z_min_nm) / num_bins

# Create bins from 0 to cell_dist
bins = np.linspace(0, cell_dist, num_bins + 1)

print(f"  ✓ Target bin width: {target_dz} nm")
print(f"  ✓ Actual bin width: {dz:.6f} nm")
print(f"  ✓ Number of bins:   {num_bins}")
print(f"  ✓ Z range:          0 to {cell_dist:.4f} nm")
print()

# Pre-allocate accumulators
hist_cat_total = np.zeros(num_bins, dtype=np.float64)
hist_an_total = np.zeros(num_bins, dtype=np.float64)
hist_solv_total = np.zeros(num_bins, dtype=np.float64)

print(f"=== Analysis Parameters ===")
print(f"  Bin width:       {dz:.6f} nm")
print(f"  Range:           0 to {cell_dist:.4f} nm")
print(f"  Bins:            {num_bins}")
print(f"  Frames:          {frameCount}")
print(f"  Sheet area:      {sheet_area:.6f} nm²")
print(f"  Bin volume:      {sheet_area * dz:.6e} nm³")
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
# Convert to actual Z coordinates (add z_min back) then to Angstrom
z_centers_angstrom = (z_centers + z_min_nm) * 10.0  # Convert nm to Å and shift to actual position

print("=== Charge Density Statistics ===")
print(f"  Cation:  [{density_cat.min():.6e}, {density_cat.max():.6e}] C/nm³")
print(f"  Anion:   [{density_an.min():.6e}, {density_an.max():.6e}] C/nm³")
print(f"  Solvent: [{density_solv.min():.6e}, {density_solv.max():.6e}] C/nm³")
print(f"  Total:   [{density_total.min():.6e}, {density_total.max():.6e}] C/nm³")
print()


### Write output files
print("Writing results...")

output_files = [
    ("hist_q_cat_V8.dat", density_cat, "cation"),
    ("hist_q_an_V8.dat", density_an, "anion"),
    ("hist_q_solv_V8.dat", density_solv, "solvent"),
    ("hist_q_total_V8.dat", density_total, "total")
]

for filename, density, species in output_files:
    with open(filename, "w") as f:
        for z, rho in zip(z_centers_angstrom, density):
            print(f'{z:5.8f}  {rho:5.8f}', file=f)
    print(f"  ✓ {filename}")

print()


### Plot results
print("Generating plots...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), dpi=300)

# Individual species
ax1.plot(z_centers_angstrom, density_cat, label=f'Cation ({namecat})', 
         color='blue', linewidth=1.5, alpha=0.8)
ax1.plot(z_centers_angstrom, density_an, label=f'Anion ({namean})', 
         color='red', linewidth=1.5, alpha=0.8)
if len(solvent_idx) > 0 and np.any(density_solv != 0):
    ax1.plot(z_centers_angstrom, density_solv, label='Solvent', 
             color='green', linewidth=1.5, alpha=0.8)
ax1.set_xlabel('Z Position (Å)', fontsize=11)
ax1.set_ylabel('Charge Density (C/nm³)', fontsize=11)
ax1.set_title('Individual Species Charge Density', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.5)

# Total solution
ax2.plot(z_centers_angstrom, density_total, label='Total', color='black', linewidth=2)
ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
ax2.set_xlabel('Z Position (Å)', fontsize=11)
ax2.set_ylabel('Charge Density (C/nm³)', fontsize=11)
ax2.set_title('Total Charge Density', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.5)

fig.suptitle(f'Charge Density Analysis - {namecat}/{namean}', 
             fontsize=14, fontweight='bold', y=0.995)

plt.tight_layout()
plt.savefig("charge_density_V8.png", dpi=600, bbox_inches='tight')
print("  ✓ charge_density_V8.png")

print()
print("=" * 70)
print("✓ Analysis Complete - V8 Auto-Detection Edition")
print()
print("New features in V8:")
print("  • Auto-detect electrode spacing from PDB structure")
print("  • Automatic unit conversion (Å → nm)")
print("  • Dynamic bin width calculation: dz = (z_max - z_min) / bins")
print("  • No more hard-coded cell_dist = 14.0 nm")
print("  • Same accuracy, more flexibility")
print()
print('"Good taste means eliminating magic numbers"')
print("                    - Linus Torvalds (adapted)")
print("=" * 70)
