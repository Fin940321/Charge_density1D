#!/usr/bin/env python3
"""
Overlay plot for 2V and 4V total charge density comparison.
Reads hist_q_total*.dat from 2V and 4V folders and creates overlay plot.
Auto-detects electrode positions from PDB structure using MDAnalysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda

# =============================================================================
# Configuration
# =============================================================================
DATA_1V = "1V_charge_density1D/hist_q_total_1V.dat"
DATA_2V = "2V_charge_density1D/hist_q_total_2V.dat"
DATA_3V = "3V_charge_density1D/hist_q_total_3V.dat"
DATA_4V = "4V_charge_density1D/hist_q_total_4V.dat"
ELECTRODE_PDB = "2V_charge_density1D/2V_start_nodrudes.pdb"  # For electrode detection
OUTPUT_FILE = "charge_density_overlay_total.png"

# =============================================================================
# Detect Electrode Boundaries
# =============================================================================
print("=== Detecting Electrode Boundaries ===")
print(f"  Loading electrode structure: {ELECTRODE_PDB}")

u_electrode = mda.Universe(ELECTRODE_PDB)
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

print(f"  ✓ Electrode atoms (grpc): {len(electrode)}")
print(f"  ✓ Positive electrode position: {z_min_angstrom:.2f} Å")
print(f"  ✓ Negative electrode position:   {z_max_angstrom:.2f} Å")
print()

# =============================================================================
# Load Data
# =============================================================================
print("=== Loading Data ===")
data_1V = np.loadtxt(DATA_1V)
data_2V = np.loadtxt(DATA_2V)
data_3V = np.loadtxt(DATA_3V)
data_4V = np.loadtxt(DATA_4V)

z_1V = data_1V[:, 0]
q_1V = data_1V[:, 1]

z_2V = data_2V[:, 0]  # Z position (Å)
q_2V = data_2V[:, 1]  # Charge density (C/nm³)

z_3V = data_3V[:, 0]
q_3V = data_3V[:, 1]

z_4V = data_4V[:, 0]
q_4V = data_4V[:, 1]

print(f"1V data: {len(z_1V)} points, Z range: {z_1V.min():.2f} - {z_1V.max():.2f} Å")
print(f"2V data: {len(z_2V)} points, Z range: {z_2V.min():.2f} - {z_2V.max():.2f} Å")
print(f"3V data: {len(z_3V)} points, Z range: {z_3V.min():.2f} - {z_3V.max():.2f} Å")
print(f"4V data: {len(z_4V)} points, Z range: {z_4V.min():.2f} - {z_4V.max():.2f} Å")

# =============================================================================
# Plot
# =============================================================================
print("Creating overlay plot...")

fig, ax = plt.subplots(figsize=(12, 6), dpi=600)

### Plot total charge density for all voltages
ax.plot(z_1V, q_1V, label='1V', color='green', linewidth=1.5, alpha=0.8)
ax.plot(z_2V, q_2V, label='2V', color='blue', linewidth=1.5, alpha=0.8)
ax.plot(z_3V, q_3V, label='3V', color='orange', linewidth=1.5, alpha=0.8)
ax.plot(z_4V, q_4V, label='4V', color='red', linewidth=1.5, alpha=0.8)

# Electrode position markers
ax.axvline(x=z_min_angstrom, color='black', linestyle='--', linewidth=2, alpha=0.7, 
           label=f'Positive Electrode ({z_min_angstrom:.1f} Å)')
ax.axvline(x=z_max_angstrom, color='gray', linestyle='--', linewidth=2, alpha=0.7, 
           label=f'Negative Electrode ({z_max_angstrom:.1f} Å)')

# Axis labels and formatting
ax.set_xlabel('Z Position (Å)', fontsize=11)
ax.set_ylabel('Charge Density (C/nm³)', fontsize=11)
ax.set_title('Charge Density of [BMIM][TFSI] at 2CNT Electrodes', fontsize=12, fontweight='bold')

# Set y-axis limits (adjusted for 4V larger amplitude)
ax.set_ylim(-20, 30)

# Grid and legend
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(fontsize=10, loc='best')

# Tight layout
plt.tight_layout()

# Save figure
plt.savefig(OUTPUT_FILE, dpi=600, bbox_inches='tight')
print(f"Saved: {OUTPUT_FILE}")

### Plot total charge density at CNT/G surface for all voltages

print("Creating surface zoom plot...")
fig2, ax2 = plt.subplots(figsize=(10, 6), dpi=600)

# Shift Z coordinates: set positive electrode to 0
ax2.plot(z_1V - z_min_angstrom, q_1V, label='1V', color='green', linewidth=1.5, alpha=0.8)
ax2.plot(z_2V - z_min_angstrom, q_2V, label='2V', color='blue', linewidth=1.5, alpha=0.8)
ax2.plot(z_3V - z_min_angstrom, q_3V, label='3V', color='orange', linewidth=1.5, alpha=0.8)
ax2.plot(z_4V - z_min_angstrom, q_4V, label='4V', color='red', linewidth=1.5, alpha=0.8)

# Electrode surface marker
ax2.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Electrode Surface')

# Set X-axis limit to 0-14 Å
ax2.set_xlim(0, 14)
ax2.set_ylim(-20, 30)

# Axis labels and formatting
ax2.set_xlabel('Distance from Positive Electrode (Å)', fontsize=11)
ax2.set_ylabel('Charge Density (C/nm³)', fontsize=11)
ax2.set_title('Charge Density Near Positive Electrode (0-14 Å)', fontsize=12, fontweight='bold')

ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(fontsize=10, loc='best')

plt.tight_layout()

# Save the new figure
OUTPUT_FILE2 = "charge_density_at_surface.png"
plt.savefig(OUTPUT_FILE2, dpi=600, bbox_inches='tight')
print(f"Saved: {OUTPUT_FILE2}")