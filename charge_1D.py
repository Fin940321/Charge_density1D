import argparse
import time
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from MDAnalysis import Universe
import numpy as np
import seaborn as sns
from numba import jit
from simtk.openmm import NonbondedForce
from simtk.openmm.app import ForceField, Modeller, PDBFile
from simtk.unit import nanometer

DEFAULT_ELECTROLYTE = ("BMIM", "Tf2N", "HOH")
BOND_DEFINITION_FILES = (
    "sapt_residues.xml",
    "graph_residue_c.xml",
    "graph_residue_n.xml",
)
FORCEFIELD_FILES = (
    "sapt_noDB_2sheets.xml",
    "graph_c_freeze.xml",
    "graph_n_freeze.xml",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute 1D charge density profile along the z-axis."
    )
    parser.add_argument("--pdb", default="for_openmm.pdb", help="Input PDB file")
    parser.add_argument("--trajectory", default="FV_NVT.dcd", help="Input DCD file")
    parser.add_argument(
        "--ffdir",
        default="../ffdir/",
        help="Directory containing force-field XML definitions",
    )
    parser.add_argument(
        "--electrolyte",
        nargs="+",
        default=list(DEFAULT_ELECTROLYTE),
        help="Residue names treated as electrolyte",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=3000,
        help="Frame index to start processing",
    )
    parser.add_argument(
        "--avgfreq",
        type=int,
        default=1,
        help="Sampling frequency in frames",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=500,
        help="Number of histogram bins along z",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on the number of frames processed",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip rendering figures (files still saved)",
    )
    parser.add_argument(
        "--output-prefix",
        default="charge_density",
        help="Prefix for generated output files",
    )
    return parser.parse_args()


@jit(nopython=True)
def calculate_charge_bins(
    positions_z: np.ndarray, charges_array: np.ndarray, z_min: float, dz: float, bins: int
) -> np.ndarray:
    """Numba-accelerated accumulation of charges into spatial bins."""
    charge_density = np.zeros(bins)
    for idx in range(positions_z.shape[0]):
        z = positions_z[idx]
        q = charges_array[idx]
        bin_index = int((z - z_min) / dz)
        if bin_index < 0:
            bin_index = 0
        elif bin_index >= bins:
            bin_index = bins - 1
        charge_density[bin_index] += q
    return charge_density


def load_forcefield_system(pdb_path: str, ffdir: str) -> Tuple[Modeller, NonbondedForce]:
    pdb = PDBFile(pdb_path)
    for bond_file in BOND_DEFINITION_FILES:
        pdb.topology.loadBondDefinitions(str(Path(ffdir) / bond_file))
    pdb.topology.createStandardBonds()
    modeller = Modeller(pdb.topology, pdb.positions)

    forcefield = ForceField(*(str(Path(ffdir) / ff) for ff in FORCEFIELD_FILES))
    modeller.addExtraParticles(forcefield)

    system = forcefield.createSystem(
        modeller.topology,
        nonbondedCutoff=1.4 * nanometer,
        constraints=None,
        rigidWater=True,
    )

    nonbonded_force = None
    for idx in range(system.getNumForces()):
        force = system.getForce(idx)
        if isinstance(force, NonbondedForce):
            nonbonded_force = force
            break
    if nonbonded_force is None:
        raise RuntimeError("Failed to locate NonbondedForce in the system")

    return modeller, nonbonded_force


def build_charge_array(
    modeller: Modeller, nonbonded_force: NonbondedForce, selection_atoms
) -> np.ndarray:
    charge_lookup = {}
    for atom in modeller.topology.atoms():
        q, _, _ = nonbonded_force.getParticleParameters(atom.index)
        charge_lookup[atom.index] = q._value

    charges = np.empty(len(selection_atoms), dtype=np.float64)
    for idx, atom in enumerate(selection_atoms):
        try:
            charges[idx] = charge_lookup[atom.index]
        except KeyError as exc:
            raise KeyError(
                f"Atom index {atom.index} from MDAnalysis selection not found in OpenMM topology"
            ) from exc
    return charges


def compute_frame_indices(
    universe: Universe, start: int, step: int, max_frames: Optional[int]
) -> List[int]:
    total_frames = universe.trajectory.n_frames
    indices = list(range(start, total_frames, step))
    if max_frames is not None:
        indices = indices[:max_frames]
    return indices


def main() -> None:
    args = parse_args()

    pdb_path = Path(args.pdb)
    trajectory_path = Path(args.trajectory)
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")
    if not trajectory_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {trajectory_path}")

    print("=== Charge Density 1D Analysis ===")
    print(f"PDB: {pdb_path}")
    print(f"Trajectory: {trajectory_path}")
    print(f"Force-field directory: {args.ffdir}")
    print(f"Electrolyte residues: {', '.join(args.electrolyte)}")

    universe = Universe(str(pdb_path), str(trajectory_path))

    area = (
        universe.trajectory[0].triclinic_dimensions[0][0]
        * universe.trajectory[0].triclinic_dimensions[1][1]
    )

    frame_indices = compute_frame_indices(
        universe, args.start_frame, args.avgfreq, args.max_frames
    )
    if not frame_indices:
        raise ValueError("No frames selected for processing; adjust start frame or sampling.")

    print(f"Total frames in trajectory: {universe.trajectory.n_frames}")
    print(f"Start frame: {args.start_frame}")
    print(f"Sampling every {args.avgfreq} frames")
    if args.max_frames is not None:
        print(f"Limiting to first {args.max_frames} sampled frames")
    print(f"Frames scheduled for processing: {len(frame_indices)}")

    universe.trajectory[frame_indices[0]]
    all_atoms = universe.select_atoms("all")
    z_coords = all_atoms.positions[:, 2]
    z_min = float(np.min(z_coords))
    z_max = float(np.max(z_coords))
    bins = args.bins
    dz = (z_max - z_min) / bins
    bin_edges = np.linspace(z_min, z_max, bins + 1)
    z_positions = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    print(f"z-range: [{z_min:.3f}, {z_max:.3f}] (Å)")
    print(f"Bin width dz: {dz:.5f} Å")

    selection_string = "resname " + " ".join(args.electrolyte)
    electrolyte_group = universe.select_atoms(selection_string)
    if electrolyte_group.n_atoms == 0:
        raise ValueError(
            f"Electrolyte selection '{selection_string}' returned zero atoms; check residue names."
        )

    print(f"Electrolyte atoms selected: {electrolyte_group.n_atoms}")

    modeller, nonbonded_force = load_forcefield_system(str(pdb_path), args.ffdir)
    charges_array = build_charge_array(modeller, nonbonded_force, electrolyte_group.atoms)
    if charges_array.shape[0] != electrolyte_group.n_atoms:
        raise RuntimeError(
            "Mismatch between computed charges and selected electrolyte atoms"
        )

    charge_density = np.zeros(bins, dtype=np.float64)

    print("\n=== Processing Trajectory Sequentially ===")
    start_time = time.time()
    for processed_frames, frame_index in enumerate(frame_indices, start=1):
        universe.trajectory[frame_index]
        positions_z = electrolyte_group.positions[:, 2]
        frame_charge = calculate_charge_bins(
            positions_z, charges_array, z_min, dz, bins
        )
        charge_density += frame_charge
        if processed_frames % max(1, len(frame_indices) // 10) == 0 or processed_frames == len(frame_indices):
            progress = processed_frames / len(frame_indices) * 100.0
            print(f"  Processed {processed_frames}/{len(frame_indices)} frames ({progress:.1f}%)")

    end_time = time.time()
    processing_time = end_time - start_time

    total_frames_processed = len(frame_indices)
    performance = (
        total_frames_processed / processing_time if processing_time > 0 else float("inf")
    )

    print("\n=== Processing Complete ===")
    print(f"Elapsed time: {processing_time:.2f} s")
    print(f"Frames processed: {total_frames_processed}")
    print(f"Throughput: {performance:.2f} frames/s")

    charge_density = charge_density / float(total_frames_processed) / area / dz

    output_prefix = Path(args.output_prefix)
    output_dir = output_prefix.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = output_prefix.name

    print("\n=== Charge Density Results ===")
    for idx, rho in enumerate(charge_density):
        print(f"{idx:03d}  z={z_positions[idx]:8.3f}  rho={rho:12.6e}")

    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.4)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    colors = sns.color_palette("husl", 3)
    ax.plot(
        z_positions,
        charge_density,
        linewidth=2.5,
        color=colors[0],
        label="Charge Density",
        alpha=0.85,
    )
    ax.fill_between(z_positions, charge_density, alpha=0.3, color=colors[0])
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    ax.set_xlabel("z Position (Å)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Charge Density (e/Å³)", fontsize=14, fontweight="bold")
    ax.set_title("1D Charge Density Profile", fontsize=16, fontweight="bold", pad=18)
    ax.legend(frameon=True, fancybox=True)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    figure_path = output_dir / f"{base_name}_1D_seaborn.png"
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    print(f"Charge density figure saved to {figure_path}")
    if args.no_plot:
        plt.close(fig)

    cumulative_charge = np.cumsum(charge_density) * dz * area
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), dpi=150)
    ax1.plot(z_positions, charge_density, linewidth=2.5, color=colors[0])
    ax1.fill_between(z_positions, charge_density, alpha=0.3, color=colors[0])
    ax1.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    ax1.set_ylabel("Charge Density (e/Å³)", fontsize=12, fontweight="bold")
    ax1.set_title("Charge Density Profile", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    ax2.plot(z_positions, cumulative_charge, linewidth=2.5, color=colors[1])
    ax2.fill_between(z_positions, cumulative_charge, alpha=0.3, color=colors[1])
    ax2.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    ax2.set_xlabel("z Position (Å)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Cumulative Charge (e)", fontsize=12, fontweight="bold")
    ax2.set_title("Cumulative Charge Profile", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    figure_path_detailed = output_dir / f"{base_name}_1D_detailed.png"
    plt.savefig(figure_path_detailed, dpi=300, bbox_inches="tight")
    print(f"Detailed figure saved to {figure_path_detailed}")
    if args.no_plot:
        plt.close(fig2)

    output_txt = output_dir / f"{base_name}_parallel.txt"
    np.savetxt(
        output_txt,
        np.column_stack([z_positions, charge_density]),
        header="z_position charge_density",
        fmt="%.6f %.12e",
    )
    print(f"Charge density data saved to {output_txt}")

    if not args.no_plot:
        plt.show()

    print("\n=== Summary ===")
    print(f"Average charge density: {np.mean(charge_density):.6e} e/Å³")
    print(f"Max charge density: {np.max(charge_density):.6e} e/Å³")
    print(f"Min charge density: {np.min(charge_density):.6e} e/Å³")
    print(f"Total integrated charge: {np.sum(charge_density) * dz * area:.6f} e")


if __name__ == "__main__":
    main()


