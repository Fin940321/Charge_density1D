### Import modules
import MDAnalysis as mda
from openmm.app import *
from openmm import *
from openmm.unit import nanometer, picosecond, picoseconds, kelvin
from sys import stdout
import numpy as np
from numpy import linalg as LA
from copy import deepcopy
import sys
from numba import jit, prange
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

sys.setrecursionlimit(2000)


### Definition of numba-accelerated functions (OPTIMIZED)
@jit(nopython=True)
def calculate_histogram_numba_fast(positions_z, charges, bins_array):
    """使用 numba + searchsorted 加速的直方圖計算"""
    num_bins = len(bins_array) - 1
    hist = np.zeros(num_bins)
    
    for idx in range(len(positions_z)):
        pos = positions_z[idx]
        charge = charges[idx]
        
        # 使用二分搜索找到對應的 bin (更快)
        bin_idx = np.searchsorted(bins_array, pos, side='right') - 1
        
        # 確保在有效範圍內
        if 0 <= bin_idx < num_bins:
            hist[bin_idx] += charge
    
    return hist


@jit(nopython=True, parallel=False)
def extract_and_histogram(frame_xyz_z, atom_indices, charges_array, C1_grpcA, bins_array):
    """
    優化版：直接提取 Z 座標並計算直方圖，減少中間陣列
    frame_xyz_z: 只傳入 Z 座標 (第三列)
    """
    num_bins = len(bins_array) - 1
    hist = np.zeros(num_bins)
    
    for i in range(len(atom_indices)):
        atom_idx = atom_indices[i]
        pos_z = frame_xyz_z[atom_idx] - C1_grpcA
        charge = charges_array[atom_idx]
        
        # 使用二分搜索找到對應的 bin
        bin_idx = np.searchsorted(bins_array, pos_z, side='right') - 1
        
        if 0 <= bin_idx < num_bins:
            hist[bin_idx] += charge
    
    return hist


@jit(nopython=True)
def process_frame_all_species(frame_xyz_z, cation_indices, anion_indices, 
                               solvent_indices, charges_array, C1_grpcA, bins_array):
    """
    超級優化版：一次性處理一個影格的所有物種
    避免多次函數調用和陣列分配
    """
    hist_cat = extract_and_histogram(frame_xyz_z, cation_indices, charges_array, C1_grpcA, bins_array)
    hist_an = extract_and_histogram(frame_xyz_z, anion_indices, charges_array, C1_grpcA, bins_array)
    hist_solv = extract_and_histogram(frame_xyz_z, solvent_indices, charges_array, C1_grpcA, bins_array)
    
    return hist_cat, hist_an, hist_solv


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

print("=" * 60)
print("    優化版電荷密度分析 (V3 - MDAnalysis + Numba)")
print("=" * 60)
print()
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


### Extract all atomic charges (OPTIMIZED - use numpy array directly)
print("正在提取所有原子的電荷資訊...")
num_particles = nbondedForce.getNumParticles()
all_charges = np.zeros(num_particles, dtype=np.float64)

for atom_idx in tqdm(range(num_particles), desc="提取電荷", ncols=80):
    (q, sig, eps) = nbondedForce.getParticleParameters(atom_idx)
    all_charges[atom_idx] = q._value

# Convert atom indices to numpy arrays for numba (OPTIMIZED - ensure C-contiguous)
cation_array = np.ascontiguousarray(np.array(cation, dtype=np.int32))
anion_array = np.ascontiguousarray(np.array(anion, dtype=np.int32))
solvent_array = np.ascontiguousarray(np.array(solvent, dtype=np.int32))
solution_array = np.ascontiguousarray(np.array(solution, dtype=np.int32))

# Ensure charges array is also C-contiguous
all_charges = np.ascontiguousarray(all_charges)

print(f"✓ 電荷提取完成 (使用連續記憶體陣列)")
print()


### Set up histogram bins (OPTIMIZED - ensure C-contiguous)
dz = 0.01  # nm
cell_dist = 14.0  # nm
num_bins = int(cell_dist / dz)
bins = np.ascontiguousarray(np.linspace(0, cell_dist, num_bins + 1))

allQcat_i = np.zeros(num_bins, dtype=np.float64)
allQan_i = np.zeros(num_bins, dtype=np.float64)
allQsolv_i = np.zeros(num_bins, dtype=np.float64)

print(f"=== 計算參數 ===")
print(f"  Bin 寬度 (dz): {dz} nm")
print(f"  計算範圍: 0 到 {cell_dist} nm")
print(f"  Bin 數量: {num_bins}")
print(f"  分析幀數: {framestart} 到 {frameend} (共 {frameCount} 幀)")
print()


### OPTIMIZATION: Create MDAnalysis AtomGroups for faster access
print("=== 建立 MDAnalysis AtomGroup (優化) ===")
# 注意：如果 MDAnalysis 的原子索引與 OpenMM 不同，需要映射
# 這裡假設它們是一致的
try:
    cation_ag = u.atoms[cation_array]
    anion_ag = u.atoms[anion_array]
    if len(solvent) > 0:
        solvent_ag = u.atoms[solvent_array]
    else:
        solvent_ag = None
    print(f"✓ AtomGroup 建立完成")
    print(f"  陽離子 AtomGroup: {len(cation_ag)} 原子")
    print(f"  陰離子 AtomGroup: {len(anion_ag)} 原子")
    use_atomgroup = True
except:
    print("⚠ AtomGroup 建立失敗，使用索引模式")
    use_atomgroup = False
print()


### Main calculation loop using MDAnalysis (SUPER OPTIMIZED)
start_time = time.time()

print("正在計算電荷密度 (超級優化版)...")
print("優化項目:")
print("  ✓ 使用 numba JIT 編譯")
print("  ✓ 使用二分搜索 (searchsorted)")
print("  ✓ 減少中間陣列分配")
print("  ✓ 使用連續記憶體陣列")
print("  ✓ 合併函數調用")
print()

# Warm-up: 預編譯 numba 函數
print("預熱 numba JIT 編譯器...")
u.trajectory[framestart]
dummy_xyz_z = np.ascontiguousarray(u.atoms.positions[:, 2] / 10.0)
dummy_C1 = dummy_xyz_z[1]
_ = process_frame_all_species(
    dummy_xyz_z, cation_array[:10], anion_array[:10], 
    solvent_array[:10] if len(solvent) > 0 else np.array([0], dtype=np.int32),
    all_charges, dummy_C1, bins
)
print("✓ JIT 編譯完成\n")

# Reset timer after warm-up
start_time = time.time()

for frame_idx in tqdm(range(framestart, frameend), desc="處理影格", unit="frame", ncols=80):
    # Move to the specific frame
    u.trajectory[frame_idx]
    
    # OPTIMIZATION: Only extract Z coordinates (reduce memory transfer)
    # Convert from Angstrom to nm and ensure C-contiguous
    frame_xyz_z = np.ascontiguousarray(u.atoms.positions[:, 2] / 10.0)
    
    # Get reference coordinate (atom index 1, z coordinate)
    C1_grpcA_current = frame_xyz_z[1]
    
    # OPTIMIZATION: Process all species in one go
    hist_cat, hist_an, hist_solv = process_frame_all_species(
        frame_xyz_z, 
        cation_array, 
        anion_array, 
        solvent_array if len(solvent) > 0 else np.array([0], dtype=np.int32),
        all_charges, 
        C1_grpcA_current, 
        bins
    )
    
    # Accumulate histograms
    allQcat_i += hist_cat
    allQan_i += hist_an
    allQsolv_i += hist_solv


# Calculate execution time
elapsed_time = time.time() - start_time
print(f"\n✓ 計算完成！")
print(f"  總耗時: {elapsed_time:.2f} 秒")
print(f"  平均每幀處理時間: {elapsed_time/frameCount*1000:.2f} 毫秒")
print(f"  處理速度: {frameCount/elapsed_time:.2f} 幀/秒")
print()


### Calculate charge densities
# 1. Calculate time-averaged charge sums
avg_charge_sum_cat = allQcat_i / frameCount
avg_charge_sum_an = allQan_i / frameCount
avg_charge_sum_solv = allQsolv_i / frameCount

# 2. Calculate total solution average charge sum (OPTIMIZATION: direct sum)
avg_charge_sum_solution = avg_charge_sum_cat + avg_charge_sum_an + avg_charge_sum_solv

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

with open("hist_q_cat_V3.dat", "w") as ofile_cat:
    for z, q in zip(z_positions_bin_centers, avg_density_cat):
        print('{0:5.8f}  {1:5.8f}'.format(z, q), file=ofile_cat)
print("  ✓ 已儲存: hist_q_cat_V3.dat")

with open("hist_q_an_V3.dat", "w") as ofile_an:
    for z, q in zip(z_positions_bin_centers, avg_density_an):
        print('{0:5.8f}  {1:5.8f}'.format(z, q), file=ofile_an)
print("  ✓ 已儲存: hist_q_an_V3.dat")

with open("hist_q_solv_V3.dat", "w") as ofile_solv:
    for z, q in zip(z_positions_bin_centers, avg_density_solv):
        print('{0:5.8f}  {1:5.8f}'.format(z, q), file=ofile_solv)
print("  ✓ 已儲存: hist_q_solv_V3.dat")

with open("hist_q_solution_V3.dat", "w") as ofile_solution:
    for z, q in zip(z_positions_bin_centers, avg_density_solution):
        print('{0:5.8f}  {1:5.8f}'.format(z, q), file=ofile_solution)
print("  ✓ 已儲存: hist_q_solution_V3.dat")

print()


### Plot results using Matplotlib
print("正在生成圖表...")

# Load data from files
data_cat = np.loadtxt("hist_q_cat_V3.dat")
data_an = np.loadtxt("hist_q_an_V3.dat")
data_solv = np.loadtxt("hist_q_solv_V3.dat")
data_solution = np.loadtxt("hist_q_solution_V3.dat")

# Extract z positions and charge densities
z_cat, q_cat = data_cat[:, 0], data_cat[:, 1]
z_an, q_an = data_an[:, 0], data_an[:, 1]
z_solv, q_solv = data_solv[:, 0], data_solv[:, 1]
z_solution, q_solution = data_solution[:, 0], data_solution[:, 1]

# Create figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), dpi=300)

# Upper panel: Individual species
ax1.plot(z_cat, q_cat, label='Cations', color='blue', linewidth=1.5, alpha=0.8)
ax1.plot(z_an, q_an, label='Anions', color='red', linewidth=1.5, alpha=0.8)
ax1.plot(z_solv, q_solv, label='Solvent', color='green', linewidth=1.5, alpha=0.8)
ax1.set_xlabel('Z Position (nm)', fontsize=11)
ax1.set_ylabel('Charge Density (C/nm³)', fontsize=11)
ax1.set_title('Individual Species Charge Density Distribution', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.5)

# Lower panel: Total solution
ax2.plot(z_solution, q_solution, label='Total Solution', color='black', linewidth=2)
ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
ax2.set_xlabel('Z Position (nm)', fontsize=11)
ax2.set_ylabel('Charge Density (C/nm³)', fontsize=11)
ax2.set_title('Total Solution Charge Density Distribution', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.5)

# Add overall title with optimization info
fig.suptitle('Charge Density Analysis (V3 - Optimized with MDAnalysis + Numba)', 
             fontsize=14, fontweight='bold', y=0.995)

# Save the figure
plt.tight_layout()
plt.savefig("charge_density_distribution_V3_optimized.png", dpi=600, bbox_inches='tight')
print("  ✓ 已儲存圖表: charge_density_distribution_V3_optimized.png")

print()
print("=" * 60)
print("✓ 所有處理完成！(V3 超級優化版)")
print(f"⚡ 性能提升: 使用多項優化技術")
print("=" * 60)
