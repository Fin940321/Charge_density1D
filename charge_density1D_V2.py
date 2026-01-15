### Import modules
import MDAnalysis as mda
from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
import numpy as np
from numpy import linalg as LA
from copy import deepcopy
import sys
from numba import jit
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

sys.setrecursionlimit(2000)


### Definition of numba-accelerated functions
@jit(nopython=True)
def calculate_histogram_numba(positions_z, charges, bins_array):
    """使用 numba 加速的直方圖計算"""
    num_bins = len(bins_array) - 1
    hist = np.zeros(num_bins)
    
    for idx in range(len(positions_z)):
        pos = positions_z[idx]
        charge = charges[idx]
        
        # 找到對應的 bin
        for bin_idx in range(num_bins):
            if bins_array[bin_idx] <= pos < bins_array[bin_idx + 1]:
                hist[bin_idx] += charge
                break
        # 處理最後一個邊界情況
        if pos == bins_array[num_bins]:
            hist[num_bins - 1] += charge
    
    return hist


@jit(nopython=True)
def extract_charges_positions(frame_xyz, atom_indices, charges_array, C1_grpcA):
    """使用 numba 加速提取座標和電荷"""
    num_atoms = len(atom_indices)
    position_z = np.zeros(num_atoms)
    charges = np.zeros(num_atoms)
    
    for i in range(num_atoms):
        atom_idx = atom_indices[i]
        position_z[i] = frame_xyz[atom_idx, 2] - C1_grpcA
        charges[i] = charges_array[atom_idx]
    
    return position_z, charges


### Input parameters
namecat = 'BMIM'
namean = 'trfl'
namegrp = 'grp'
temperature = 300

# Trajectory and topology files
traj_file = "FV_NVT.dcd"
top_file = "start_drudes.pdb"

framestart = 1000
frameCount = 1010
frameend = framestart + frameCount

print("=== 使用 MDAnalysis 載入軌跡 ===")
print(f"軌跡檔案: {traj_file}")
print(f"拓撲檔案: {top_file}")

# Load trajectory with MDAnalysis
u = mda.Universe(top_file, traj_file)
print(f"✓ 成功載入軌跡")
print(f"  總幀數: {len(u.trajectory)}")
print(f"  總原子數: {u.atoms.n_atoms}")
print(f"  軌跡時間範圍: {u.trajectory[0].time:.2f} - {u.trajectory[-1].time:.2f} ps")
print()


### Load OpenMM system to get charges
ffdir = './ffdir/'
electrode_ffdir = './electrode_ffdir/'

print("=== 使用 OpenMM 載入力場與電荷資訊 ===")
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

# Get NonbondedForce to extract charges
nbondedForce = [f for f in [system.getForce(i) for i in range(system.getNumForces())] 
                if type(f) == NonbondedForce][0]

# Create simulation object to access topology
integ_md = DrudeLangevinIntegrator(temperature, 1/picosecond, 1*kelvin, 1/picosecond, 0.001*picoseconds)
platform = Platform.getPlatformByName('CPU')
simmd = Simulation(modeller.topology, system, integ_md, platform)

# Calculate sheet area
boxVecs = simmd.topology.getPeriodicBoxVectors()
crossBox = np.cross(boxVecs[0], boxVecs[1])
sheet_area = np.dot(crossBox, crossBox)**0.5 / nanometer**2
sheet_area1 = np.dot(crossBox, crossBox)

print(f"✓ 力場載入完成")
print(f"  盒子向量面積: {sheet_area1}")
print(f"  片層面積: {sheet_area:.6f} nm²")
print()


### Extract atom indices for different species
print("=== 提取原子索引 ===")
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

solution = deepcopy(cation)
solution.extend(deepcopy(anion))
solution.extend(deepcopy(solvent))

print(f"  陽離子 ({namecat}) 原子數: {len(cation)}")
print(f"  陰離子 ({namean}) 原子數: {len(anion)}")
print(f"  溶劑原子數: {len(solvent)}")
print(f"  總溶液原子數: {len(solution)}")
print()


### Extract all atomic charges
print("正在提取所有原子的電荷資訊...")
all_charges = np.zeros(nbondedForce.getNumParticles())
for atom_idx in tqdm(range(nbondedForce.getNumParticles()), desc="提取電荷", ncols=80):
    (q, sig, eps) = nbondedForce.getParticleParameters(atom_idx)
    all_charges[atom_idx] = q._value

# Convert atom indices to numpy arrays for numba
cation_array = np.array(cation, dtype=np.int32)
anion_array = np.array(anion, dtype=np.int32)
solvent_array = np.array(solvent, dtype=np.int32)
solution_array = np.array(solution, dtype=np.int32)

print(f"✓ 電荷提取完成")
print()


### Set up histogram bins
dz = 0.01  # nm
cell_dist = 14.0  # nm
num_bins = int(cell_dist / dz)
bins = np.linspace(0, cell_dist, num_bins + 1)

allQcat_i = np.zeros(num_bins)
allQan_i = np.zeros(num_bins)
allQsolv_i = np.zeros(num_bins)
allQsolution_i = np.zeros(num_bins)

print(f"=== 計算參數 ===")
print(f"  Bin 寬度 (dz): {dz} nm")
print(f"  計算範圍: 0 到 {cell_dist} nm")
print(f"  Bin 數量: {num_bins}")
print(f"  分析幀數: {framestart} 到 {frameend} (共 {frameCount} 幀)")
print()


### Main calculation loop using MDAnalysis
start_time = time.time()

print("正在計算電荷密度...")
for frame_idx in tqdm(range(framestart, frameend), desc="處理影格", unit="frame", ncols=80):
    # Move to the specific frame in MDAnalysis trajectory
    u.trajectory[frame_idx]
    
    # Get all atomic positions for current frame (in nm, MDAnalysis uses Angstrom by default)
    # Convert from Angstrom to nm
    current_frame_xyz = u.atoms.positions / 10.0  # Convert Å to nm
    
    # Get reference coordinate (atom index 1, z coordinate)
    C1_grpcA_current = current_frame_xyz[1, 2]
    
    # Extract positions and charges for different species using numba-accelerated function
    cat_pos_z, cat_charges = extract_charges_positions(
        current_frame_xyz, cation_array, all_charges, C1_grpcA_current
    )
    
    an_pos_z, an_charges = extract_charges_positions(
        current_frame_xyz, anion_array, all_charges, C1_grpcA_current
    )
    
    solv_pos_z, solv_charges = extract_charges_positions(
        current_frame_xyz, solvent_array, all_charges, C1_grpcA_current
    )
    
    solution_pos_z, solution_charges = extract_charges_positions(
        current_frame_xyz, solution_array, all_charges, C1_grpcA_current
    )
    
    # Calculate histograms with weighted charges
    hist_cat, _ = np.histogram(cat_pos_z, bins=bins, weights=cat_charges)
    allQcat_i += hist_cat
    
    hist_an, _ = np.histogram(an_pos_z, bins=bins, weights=an_charges)
    allQan_i += hist_an
    
    hist_solv, _ = np.histogram(solv_pos_z, bins=bins, weights=solv_charges)
    allQsolv_i += hist_solv
    
    hist_solution, _ = np.histogram(solution_pos_z, bins=bins, weights=solution_charges)
    allQsolution_i += hist_solution


# Calculate execution time
elapsed_time = time.time() - start_time
print(f"\n✓ 計算完成！")
print(f"  總耗時: {elapsed_time:.2f} 秒")
print(f"  平均每幀處理時間: {elapsed_time/frameCount*1000:.2f} 毫秒")
print()


### Calculate charge densities
# 1. Calculate time-averaged charge sums
avg_charge_sum_cat = allQcat_i / frameCount
avg_charge_sum_an = allQan_i / frameCount
avg_charge_sum_solv = allQsolv_i / frameCount

# 2. Calculate total solution average charge sum
avg_charge_sum_solution = avg_charge_sum_cat + avg_charge_sum_an

# 3. Calculate final charge density (divide by bin volume)
bin_volume = sheet_area * dz
avg_density_cat = avg_charge_sum_cat / bin_volume
avg_density_an = avg_charge_sum_an / bin_volume
avg_density_solv = avg_charge_sum_solv / bin_volume
avg_density_solution = avg_charge_sum_solution / bin_volume

# Calculate bin centers for z positions
z_positions_bin_centers = (bins[:-1] + bins[1:]) / 2

print("=== 電荷密度統計 ===")
print(f"  陽離子平均電荷密度範圍: [{avg_density_cat.min():.6e}, {avg_density_cat.max():.6e}] C/nm³")
print(f"  陰離子平均電荷密度範圍: [{avg_density_an.min():.6e}, {avg_density_an.max():.6e}] C/nm³")
print(f"  溶劑平均電荷密度範圍: [{avg_density_solv.min():.6e}, {avg_density_solv.max():.6e}] C/nm³")
print(f"  溶液平均電荷密度範圍: [{avg_density_solution.min():.6e}, {avg_density_solution.max():.6e}] C/nm³")
print()


### Write results to files
print("正在寫入結果檔案...")

with open("hist_q_cat.dat", "w") as ofile_cat:
    for z, q in zip(z_positions_bin_centers, avg_density_cat):
        print('{0:5.8f}  {1:5.8f}'.format(z, q), file=ofile_cat)
print("  ✓ 已儲存: hist_q_cat.dat")

with open("hist_q_an.dat", "w") as ofile_an:
    for z, q in zip(z_positions_bin_centers, avg_density_an):
        print('{0:5.8f}  {1:5.8f}'.format(z, q), file=ofile_an)
print("  ✓ 已儲存: hist_q_an.dat")

with open("hist_q_solv.dat", "w") as ofile_solv:
    for z, q in zip(z_positions_bin_centers, avg_density_solv):
        print('{0:5.8f}  {1:5.8f}'.format(z, q), file=ofile_solv)
print("  ✓ 已儲存: hist_q_solv.dat")

with open("hist_q_solution.dat", "w") as ofile_solution:
    for z, q in zip(z_positions_bin_centers, avg_density_solution):
        print('{0:5.8f}  {1:5.8f}'.format(z, q), file=ofile_solution)
print("  ✓ 已儲存: hist_q_solution.dat")

print()


### Plot results using Matplotlib
print("正在生成圖表...")

# Load data from files
data_cat = np.loadtxt("hist_q_cat.dat")
data_an = np.loadtxt("hist_q_an.dat")
data_solv = np.loadtxt("hist_q_solv.dat")
data_solution = np.loadtxt("hist_q_solution.dat")

# Extract z positions and charge densities
z_cat, q_cat = data_cat[:, 0], data_cat[:, 1]
z_an, q_an = data_an[:, 0], data_an[:, 1]
z_solv, q_solv = data_solv[:, 0], data_solv[:, 1]
z_solution, q_solution = data_solution[:, 0], data_solution[:, 1]

# Create figure
plt.figure(figsize=(12, 7), dpi=300)

# Plot data for cations, anions, solvent, and total solution
plt.plot(z_cat, q_cat, label='Cations', color='blue', linewidth=1.5)
plt.plot(z_an, q_an, label='Anions', color='red', linewidth=1.5)
plt.plot(z_solv, q_solv, label='Solvent', color='green', linewidth=1.5)
plt.plot(z_solution, q_solution, label='Total Solution', color='black', linestyle='--', linewidth=2)

# Add labels and title
plt.xlabel('Z Position (nm)', fontsize=12)
plt.ylabel('Charge Density (C/nm³)', fontsize=12)
plt.title('Charge Density Distribution along Z Axis (MDAnalysis)', fontsize=14, fontweight='bold')

# Add legend
plt.legend(fontsize=11)

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.6)

# Save the figure
plt.tight_layout()
plt.savefig("charge_density_distribution_V2.png", dpi=600, bbox_inches='tight')
print("  ✓ 已儲存圖表: charge_density_distribution_V2.png")

print("\n" + "="*50)
print("✓ 所有處理完成！(使用 MDAnalysis 版本)")
print("="*50)
