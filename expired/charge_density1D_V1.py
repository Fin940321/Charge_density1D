### Import modules
import mdtraj as mdtraj
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
sys.setrecursionlimit(2000)


### Definition of classes and functions
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

class get_charge:
    def __init__(self, atomlist):
        self.atomlist = atomlist
        # self.position_z = []
        # self.charges = []
        #C1_grpcA =  frame.xyz[i, 1 , 2]
# 接受當前影格座標 frame_xyz 和參考座標 C1_grpcA 作為參數
    def pointcharge(self, frame_xyz, C1_grpcA):
        position_z = []
        charges = []
        for atom_i in range(len(self.atomlist)):
            atom_idx = self.atomlist[atom_i]
            # 使用傳入的 frame_xyz，注意索引變為 0，因為只傳入了一幀的數據
            pos_z = frame_xyz[atom_idx, 2] - float(C1_grpcA)
            (q, sig, eps) = nbondedForce.getParticleParameters(atom_idx)
            position_z.append(pos_z)
            charges.append(q._value)
        return position_z, charges


### input ions & trajectory & topology
namecat = 'BMIM'
namean = 'trfl'
#namesolv= 'trfl'
namegrp = 'grp'

temperature=300
# here, we load pdb with drude oscillators
traj = "FV_NVT.dcd"
top = "start_drudes.pdb"

framestart = 1000
frameCount = 1010

frameend = framestart + frameCount
#framestart = 49000
frame = mdtraj.load(traj, top=top)
#frameend = frame.n_frames


### import the file we want
# use OpenMM library to get charges
# here we load pdb without drude oscillators
ffdir = './ffdir/'
electrode_ffdir = './electrode_ffdir/'

pdb = PDBFile('start_nodrudes.pdb')
pdb.topology.loadBondDefinitions(ffdir + 'graph_residue_c.xml')
pdb.topology.loadBondDefinitions(ffdir + 'graph_residue_n.xml')
#pdb.topology.loadBondDefinitions(ffdir + 'graph_residue_s.xml')
pdb.topology.loadBondDefinitions(ffdir + 'sapt_residues.xml')
pdb.topology.loadBondDefinitions(electrode_ffdir + 'nanotube9x9_residue_c.xml')
pdb.topology.loadBondDefinitions(electrode_ffdir + 'nanotube9x9_residue_n.xml')
pdb.topology.createStandardBonds();
modeller = Modeller(pdb.topology, pdb.positions)
#forcefield = ForceField(ffdir + 'graph_residue_c.xml',ffdir + 'graph_c_freeze.xml',ffdir + 'sapt_noDB.xml',electrode_ffdir + 'nanotube9x9_c_freeze.xml',electrode_ffdir + 'nanotube9x9_n_freeze.xml')
forcefield = ForceField(ffdir + 'graph_c_freeze.xml',ffdir + 'graph_n_freeze.xml',ffdir + 'sapt_noDB_2sheets.xml',electrode_ffdir + 'nanotube9x9_c_freeze.xml',electrode_ffdir + 'nanotube9x9_n_freeze.xml')
#forcefield = ForceField(ffdir + 'graph_c_freeze.xml',ffdir + 'graph_n_freeze.xml',ffdir + 'graph_s_freeze.xml',ffdir + 'sapt_noDB_2sheets.xml',electrode_ffdir + 'nanotube9x9_c_freeze.xml',electrode_ffdir + 'nanotube9x9_n_freeze.xml')
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
sheet_area1 = np.dot(crossBox, crossBox)
# print the outcome to see it's true or false :
print(sheet_area1)
print(sheet_area)


### input the values we want to use
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
    # if res.name == namesolv:
    #     for atom in res._atoms:
    #         solvent.append(atom.index)

solution =  deepcopy(cation)
solution.extend(deepcopy(anion))
solution.extend(deepcopy(solvent))
print(f"Total solution atoms: {len(solution)}")

# 預先提取所有原子的電荷（用於 numba 加速）
print("正在提取原子電荷資訊...")
all_charges = np.zeros(nbondedForce.getNumParticles())
for atom_idx in tqdm(range(nbondedForce.getNumParticles()), desc="提取電荷", ncols=80):
    (q, sig, eps) = nbondedForce.getParticleParameters(atom_idx)
    all_charges[atom_idx] = q._value

# 將原子索引轉換為 numpy 陣列（用於 numba）
cation_array = np.array(cation, dtype=np.int32)
anion_array = np.array(anion, dtype=np.int32)
solvent_array = np.array(solvent, dtype=np.int32)
solution_array = np.array(solution, dtype=np.int32)

print(f"陽離子原子數: {len(cation)}")
print(f"陰離子原子數: {len(anion)}")
print(f"溶劑原子數: {len(solvent)}")
# print the outcome to see it's true or false :
#print(cation)
#print(anion)
#print(len(solvent))
#print(charges, len(charges))


### use a loop to go through each frame
dz = 0.01
cell_dist = 14.
num_bins = int(cell_dist / dz)
# bins 是一個 NumPy 陣列，儲存了所有區間的邊界值，例如 [0.0, 0.01, 0.02, ...]
bins = np.linspace(0, cell_dist, num_bins + 1)
#zbins = [0.0 for i in range(1, int(cell_dist/dz))]

allQcat_i = np.zeros(num_bins)
allQan_i = np.zeros(num_bins)
allQsolv_i = np.zeros(num_bins) # 如果有溶劑
allQsolution_i = np.zeros(num_bins) # 雖然可以後算，但若要累加就需初始化
# count_i = np.array(bins)
print(f"\n=== 計算參數 ===")
print(f"Bin 寬度 (dz): {dz} nm")
print(f"計算範圍: 0 到 {cell_dist} nm")
print(f"Bin 數量: {num_bins}")
print(f"分析幀數: {framestart} 到 {frameend} (共 {frameCount} 幀)")
print()

# 開始計時
start_time = time.time()

print("正在計算電荷密度...")
for i in tqdm(range(framestart, frameend), desc="處理影格", unit="frame", ncols=80):
    current_frame_xyz = frame.xyz[i] # 獲取當前影格的所有原子座標
    C1_grpcA_current = frame.xyz[i, 1, 2] # 獲取當前影格的參考座標

    # 使用 numba 優化的函數提取座標和電荷
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

    # 使用 numba 優化的直方圖計算（如果原子數夠多的話比 np.histogram 快）
    # 但對於小系統，np.histogram 可能更快，這裡保留原來的方式
    hist_cat, bin_edges = np.histogram(cat_pos_z, bins=bins, weights=cat_charges)
    allQcat_i += hist_cat

    hist_an, bin_edges = np.histogram(an_pos_z, bins=bins, weights=an_charges)
    allQan_i += hist_an

    hist_solv, bin_edges = np.histogram(solv_pos_z, bins=bins, weights=solv_charges)
    allQsolv_i += hist_solv

    hist_solution, bin_edges = np.histogram(solution_pos_z, bins=bins, weights=solution_charges)
    allQsolution_i += hist_solution

    # 迴圈結束後，allQcat_i 等陣列儲存的是所有影格在每個區間的「電荷總和」的總和

# 計算執行時間
elapsed_time = time.time() - start_time
print(f"\n計算完成！耗時: {elapsed_time:.2f} 秒")
print(f"平均每幀處理時間: {elapsed_time/frameCount*1000:.2f} 毫秒")
print()

# 1. 計算「時間平均」的電荷總和
avg_charge_sum_cat = allQcat_i / frameCount
avg_charge_sum_an = allQan_i / frameCount
avg_charge_sum_solv = allQsolv_i / frameCount

# 2. 計算總溶液的平均電荷總和
avg_charge_sum_solution = avg_charge_sum_cat + avg_charge_sum_an

# 3. 計算最終的「電荷密度」 (除以區間體積)
bin_volume = sheet_area * dz
avg_density_cat = avg_charge_sum_cat / bin_volume
avg_density_an = avg_charge_sum_an / bin_volume
avg_density_solv = avg_charge_sum_solv / bin_volume
avg_density_solution = avg_charge_sum_solution / bin_volume

z_positions_bin_centers = (bins[:-1] + bins[1:]) / 2

print("=== 電荷密度統計 ===")
print(f"陽離子平均電荷密度範圍: [{avg_density_cat.min():.6e}, {avg_density_cat.max():.6e}] C/nm³")
print(f"陰離子平均電荷密度範圍: [{avg_density_an.min():.6e}, {avg_density_an.max():.6e}] C/nm³")
print(f"溶液平均電荷密度範圍: [{avg_density_solution.min():.6e}, {avg_density_solution.max():.6e}] C/nm³")
print()


### calculate
print("正在寫入結果檔案...")
with open("hist_q_cat.dat", "w") as ofile_cat:
    for z, q in zip(z_positions_bin_centers, avg_density_cat):
        print('{0:5.8f}  {1:5.8f}'.format(z, q), file=ofile_cat)

print("✓ 已儲存: hist_q_cat.dat")

with open("hist_q_an.dat", "w") as ofile_an:
    for z, q in zip(z_positions_bin_centers, avg_density_an):
        print('{0:5.8f}  {1:5.8f}'.format(z, q), file=ofile_an)

print("✓ 已儲存: hist_q_an.dat")

with open("hist_q_solv.dat", "w") as ofile_solv:
    for z, q in zip(z_positions_bin_centers, avg_density_solv):
        print('{0:5.8f}  {1:5.8f}'.format(z, q), file=ofile_solv)

print("✓ 已儲存: hist_q_solv.dat")
        
with open("hist_q_solution.dat", "w") as ofile_solution:
    for z, q in zip(z_positions_bin_centers, avg_density_solution):
        print('{0:5.8f}  {1:5.8f}'.format(z, q), file=ofile_solution)
        
print("✓ 已儲存: hist_q_solution.dat")
print("\n所有資料檔案已生成完畢！")


### Matplotlib to plot the results
import matplotlib.pyplot as plt

print("\n正在生成圖表...")

# Load data from files (example filenames, adjust as needed)
data_cat = np.loadtxt("hist_q_cat.dat")
data_an = np.loadtxt("hist_q_an.dat")
data_solv = np.loadtxt("hist_q_solv.dat")
data_solution = np.loadtxt("hist_q_solution.dat")

# Extract z positions and charge densities
z_cat, q_cat = data_cat[:, 0], data_cat[:, 1]
z_an, q_an = data_an[:, 0], data_an[:, 1]
z_solv, q_solv = data_solv[:, 0], data_solv[:, 1]
z_solution, q_solution = data_solution[:, 0], data_solution[:, 1]

# Create a plot
plt.figure(figsize=(12, 7), dpi=300)

# Plot data for cations, anions, solvent, and total solution
plt.plot(z_cat, q_cat, label='Cations', color='blue', linewidth=1.5)
plt.plot(z_an, q_an, label='Anions', color='red', linewidth=1.5)
plt.plot(z_solv, q_solv, label='Solvent', color='green', linewidth=1.5)
plt.plot(z_solution, q_solution, label='Total Solution', color='black', linestyle='--', linewidth=2)

# Add labels and title
plt.xlabel('Z Position (nm)', fontsize=12)
plt.ylabel('Charge Density (C/nm³)', fontsize=12)
plt.title('Charge Density Distribution along Z Axis', fontsize=14, fontweight='bold')

# Add legend
plt.legend(fontsize=11)

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.6)

# Save the figure
plt.tight_layout()
plt.savefig("charge_density_distribution.png", dpi=600, bbox_inches='tight')
print("✓ 已儲存圖表: charge_density_distribution.png")

print("\n✓ 所有處理完成！")