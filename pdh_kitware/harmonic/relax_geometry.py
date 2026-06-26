from pathlib import Path

from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter
from ase.io import read, write
from ase.optimize import FIRE


def optimize(cif_file, calc, output_path=None):
    """Relax atomic positions and cell of a CIF structure.

    Returns the path of the written output file.
    """
    cif_file = Path(cif_file)
    if output_path is None:
        output_path = cif_file.with_stem(f"{cif_file.stem}_relaxed")
    else:
        output_path = Path(output_path)

    atoms = read(str(cif_file))
    atoms.calc = calc
    atoms.set_constraint(FixSymmetry(atoms))

    log_path = output_path.with_suffix(".log")
    relaxer = FIRE(FrechetCellFilter(atoms, scalar_pressure=0), logfile=str(log_path))

    converged = relaxer.run(fmax=1e-5, steps=10000)
    if not converged:
        print(f"Warning: {cif_file.name} did not fully converge within 10000 steps.")

    write(str(output_path), atoms)
    return output_path
