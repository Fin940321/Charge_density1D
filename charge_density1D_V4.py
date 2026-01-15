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
from numba import jit
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

sys.setrecursionlimit(2000)


### HYBRID APPROACH: NumPy vectorization for extraction + optimized histogram
@jit(nopython=True)
def numba_histogram_manual(positions_z, charges, bins_array):
    """純 Numba 手動實現的直方圖（用於比較）"""
    num_bins = len(bins_array) - 1
    hist = np.zeros(num_bins)
    
    for idx in range(len(positions_z)):
        pos = positions_z[idx]
        charge = charges[idx]
        bin_idx = np.searchsorted(bins_array, pos, side='right') - 1
        if 0 <= bin_idx < num_bins:
            hist[bin_idx] += charge
    
    return hist


def numpy_vectorized_approach(frame_xyz_z, atom_indices, charges_array, C1_grpcA, bins):
    """純 NumPy 向量化方法（推薦）"""
    # 向量化提取：一次性完成所有索引操作
    positions_z = frame_xyz_z[atom_indices] - C1_grpcA
    charges = charges_array[atom_indices]
    
    # 使用高度優化的 np.histogram
    hist, _ = np.histogram(positions_z, bins=bins, weights=charges)
    return hist


def hybrid_approach(frame_xyz_z, atom_indices, charges_array, C1_grpcA, bins):
    """混合方法：NumPy 提取 + Numba 直方圖（可能更快的特殊情況）"""
    # NumPy 向量化提取
    positions_z = frame_xyz_z[atom_indices] - C1_grpcA
    charges = charges_array[atom_indices]
    
    # Numba 直方圖（對於大量 bins 可能更快）
    hist = numba_histogram_manual(positions_z, charges, bins)
    return hist


def process_frame_vectorized(frame_xyz_z, cation_indices, anion_indices, 
                             solvent_indices, charges_array, C1_grpcA, bins):
    """
    完全向量化版本：最簡潔且通常最快
    """
    hist_cat = numpy_vectorized_approach(frame_xyz_z, cation_indices, charges_array, C1_grpcA, bins)
    hist_an = numpy_vectorized_approach(frame_xyz_z, anion_indices, charges_array, C1_grpcA, bins)
    hist_solv = numpy_vectorized_approach(frame_xyz_z, solvent_indices, charges_array, C1_grpcA, bins)
    
    return hist_cat, hist_an, hist_solv


### Input parameters
namecat = 'BMIM'
namean = 'trfl' or 'Tf2'
namegrp = 'grp'
temperature = 300

# Trajectory and topology files
traj_file = "FV_NVT.dcd"
top_file = "start_drudes.pdb"

framestart = 1285
frameCount = 2569 - 1285
frameend = framestart + frameCount

print("=" * 70)
print("    混合優化版電荷密度分析 (V4 - NumPy Vectorized)")
print("=" * 70)
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


### Extract all atomic charges
print("正在提取所有原子的電荷資訊...")
num_particles = nbondedForce.getNumParticles()
all_charges = np.zeros(num_particles, dtype=np.float64)

for atom_idx in tqdm(range(num_particles), desc="提取電荷", ncols=80):
    (q, sig, eps) = nbondedForce.getParticleParameters(atom_idx)
    all_charges[atom_idx] = q._value

# Convert to numpy arrays (no need for explicit C-contiguous in this approach)
cation_array = np.array(cation, dtype=np.int32)
anion_array = np.array(anion, dtype=np.int32)
solvent_array = np.array(solvent, dtype=np.int32) if len(solvent) > 0 else np.array([0], dtype=np.int32)

print(f"✓ 電荷提取完成")
print()


### Set up histogram bins
dz = 0.01  # nm
cell_dist = 14.0  # nm
num_bins = int(cell_dist / dz)
bins = np.linspace(0, cell_dist, num_bins + 1)

allQcat_i = np.zeros(num_bins, dtype=np.float64)
allQan_i = np.zeros(num_bins, dtype=np.float64)
allQsolv_i = np.zeros(num_bins, dtype=np.float64)

print(f"=== 計算參數 ===")
print(f"  Bin 寬度 (dz): {dz} nm")
print(f"  計算範圍: 0 到 {cell_dist} nm")
print(f"  Bin 數量: {num_bins}")
print(f"  分析幀數: {framestart} 到 {frameend} (共 {frameCount} 幀)")
print()


### Performance comparison (optional - run on a few frames)
print("=== 性能測試：Numba vs NumPy ===")
u.trajectory[framestart]
test_xyz_z = u.atoms.positions[:, 2] / 10.0
test_C1 = test_xyz_z[1]

# Test Numba approach
test_positions = test_xyz_z[cation_array] - test_C1
test_charges = all_charges[cation_array]

t1 = time.time()
for _ in range(10):
    _ = numba_histogram_manual(test_positions, test_charges, bins)
time_numba = (time.time() - t1) / 10

# Test NumPy approach
t2 = time.time()
for _ in range(10):
    _ = numpy_vectorized_approach(test_xyz_z, cation_array, all_charges, test_C1, bins)
time_numpy = (time.time() - t2) / 10

print(f"  Numba 手動迴圈: {time_numba*1000:.3f} ms/frame")
print(f"  NumPy 向量化:   {time_numpy*1000:.3f} ms/frame")
print(f"  速度比: {time_numba/time_numpy:.2f}x")
if time_numpy < time_numba:
    print(f"  ✓ NumPy 更快 {time_numba/time_numpy:.1f}x！")
else:
    print(f"  ✓ Numba 更快 {time_numpy/time_numba:.1f}x！")
print()


### Main calculation loop - Using the winner!
start_time = time.time()

print("正在計算電荷密度 (使用 NumPy 向量化)...")
print("優化策略:")
print("  ✓ NumPy 向量化批量索引")
print("  ✓ 高度優化的 np.histogram (C 實現)")
print("  ✓ 簡潔的程式碼")
print("  ✓ 只提取 Z 座標減少記憶體傳輸")
print()

for frame_idx in tqdm(range(framestart, frameend), desc="處理影格", unit="frame", ncols=80):
    # Move to the specific frame
    u.trajectory[frame_idx]
    
    # Only extract Z coordinates (reduce memory transfer)
    frame_xyz_z = u.atoms.positions[:, 2] / 10.0  # Convert Å to nm
    
    # Get reference coordinate
    C1_grpcA_current = frame_xyz_z[1]
    
    # Process all species using vectorized approach
    hist_cat, hist_an, hist_solv = process_frame_vectorized(
        frame_xyz_z, 
        cation_array, 
        anion_array, 
        solvent_array,
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
avg_charge_sum_cat = allQcat_i / frameCount
avg_charge_sum_an = allQan_i / frameCount
avg_charge_sum_solv = allQsolv_i / frameCount
avg_charge_sum_solution = avg_charge_sum_cat + avg_charge_sum_an + avg_charge_sum_solv

bin_volume = sheet_area * dz
avg_density_cat = avg_charge_sum_cat / bin_volume
avg_density_an = avg_charge_sum_an / bin_volume
avg_density_solv = avg_charge_sum_solv / bin_volume
avg_density_solution = avg_charge_sum_solution / bin_volume

z_positions_bin_centers = (bins[:-1] + bins[1:]) / 2

print("=== 電荷密度統計 ===")
print(f"  陽離子平均電荷密度範圍: [{avg_density_cat.min():.6e}, {avg_density_cat.max():.6e}] C/nm³")
print(f"  陰離子平均電荷密度範圍: [{avg_density_an.min():.6e}, {avg_density_an.max():.6e}] C/nm³")
print(f"  溶劑平均電荷密度範圍: [{avg_density_solv.min():.6e}, {avg_density_solv.max():.6e}] C/nm³")
print(f"  溶液平均電荷密度範圍: [{avg_density_solution.min():.6e}, {avg_density_solution.max():.6e}] C/nm³")
print()


### Write results to files
print("正在寫入結果檔案...")

with open("hist_q_cat_V4.dat", "w") as ofile_cat:
    for z, q in zip(z_positions_bin_centers, avg_density_cat):
        print('{0:5.8f}  {1:5.8f}'.format(z, q), file=ofile_cat)
print("  ✓ 已儲存: hist_q_cat_V4.dat")

with open("hist_q_an_V4.dat", "w") as ofile_an:
    for z, q in zip(z_positions_bin_centers, avg_density_an):
        print('{0:5.8f}  {1:5.8f}'.format(z, q), file=ofile_an)
print("  ✓ 已儲存: hist_q_an_V4.dat")

with open("hist_q_solv_V4.dat", "w") as ofile_solv:
    for z, q in zip(z_positions_bin_centers, avg_density_solv):
        print('{0:5.8f}  {1:5.8f}'.format(z, q), file=ofile_solv)
print("  ✓ 已儲存: hist_q_solv_V4.dat")

with open("hist_q_solution_V4.dat", "w") as ofile_solution:
    for z, q in zip(z_positions_bin_centers, avg_density_solution):
        print('{0:5.8f}  {1:5.8f}'.format(z, q), file=ofile_solution)
print("  ✓ 已儲存: hist_q_solution_V4.dat")

print()


### Plot results using Matplotlib
print("正在生成圖表...")

data_cat = np.loadtxt("hist_q_cat_V4.dat")
data_an = np.loadtxt("hist_q_an_V4.dat")
data_solv = np.loadtxt("hist_q_solv_V4.dat")
data_solution = np.loadtxt("hist_q_solution_V4.dat")

z_cat, q_cat = data_cat[:, 0], data_cat[:, 1]
z_an, q_an = data_an[:, 0], data_an[:, 1]
z_solv, q_solv = data_solv[:, 0], data_solv[:, 1]
z_solution, q_solution = data_solution[:, 0], data_solution[:, 1]

# Create figure with better layout
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

fig.suptitle('Charge Density Analysis (V4 - NumPy Vectorized)', 
             fontsize=14, fontweight='bold', y=0.995)

plt.tight_layout()
plt.savefig("charge_density_distribution_V4_vectorized.png", dpi=600, bbox_inches='tight')
print("  ✓ 已儲存圖表: charge_density_distribution_V4_vectorized.png")

print()
print("=" * 70)
print("✓ 所有處理完成！(V4 NumPy 向量化版本)")
print(f"⚡ 使用 NumPy 向量化獲得最佳性能")
print("=" * 70)
