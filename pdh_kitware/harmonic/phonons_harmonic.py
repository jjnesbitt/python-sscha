from pathlib import Path

import cellconstructor
import cellconstructor.Phonons
import cellconstructor.Structure
from ase.io import read
from ase.io.espresso import kspacing_to_grid


def phonons(cif_file, calc, supercell=None, output_prefix=None):
    """Compute harmonic phonons via finite displacements.

    supercell: tuple of 3 ints; auto-derived from kspacing=0.125 if None.
    output_prefix: file path prefix for QE dyn files; defaults to
        <cif_dir>/{k}x{k}x{k}_harmonic_dyn.

    Returns the prefix path used to save the files.
    """
    cif_file = Path(cif_file)

    if supercell is None:
        atoms = read(str(cif_file))
        supercell = tuple(int(k) for k in kspacing_to_grid(atoms, 0.125))

    if output_prefix is None:
        s = supercell
        output_prefix = cif_file.parent / f"{s[0]}x{s[1]}x{s[2]}_harmonic_dyn"
    else:
        output_prefix = Path(output_prefix)

    structure = cellconstructor.Structure.Structure()
    structure.read_generic_file(str(cif_file))

    harmonic_dyn = cellconstructor.Phonons.compute_phonons_finite_displacements(
        structure, calc, supercell=supercell, epsilon=0.01
    )

    harmonic_dyn.Symmetrize()
    harmonic_dyn.save_qe(str(output_prefix))
    return output_prefix
