import os
from pathlib import Path

import cellconstructor
import cellconstructor.Phonons
import numpy as np
import sscha
import sscha.Ensemble
import sscha.Relax
import sscha.SchaMinimizer
import sscha.Utilities


def _detect_nqirr(prefix):
    prefix = Path(prefix)
    return len(sorted(prefix.parent.glob(f"{prefix.name}[0-9]*")))


def run_sscha(
    dyn_prefix, calc, output_dir=None, nqirr=None, n_configs=1000, max_pop=50
):
    """Run SSCHA free-energy minimization starting from harmonic dyn files.

    dyn_prefix: absolute or relative path prefix for harmonic dyn files
        (e.g. '/data/4x4x4_harmonic_dyn').
    output_dir: directory for final_dyn* and data/ ensemble output;
        defaults to the current working directory.

    SSCHA saves relative to its working directory, so this function
    temporarily changes CWD to output_dir for the duration of the run.
    """
    dyn_prefix = Path(dyn_prefix).resolve()

    if nqirr is None:
        nqirr = _detect_nqirr(dyn_prefix)
        if nqirr == 0:
            raise FileNotFoundError(f"No dyn files found matching {dyn_prefix}[0-9]*")
        print(f"Auto-detected nqirr={nqirr}")

    if output_dir is None:
        output_dir = Path.cwd()
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    orig_dir = Path.cwd()
    os.chdir(output_dir)
    try:
        np.random.seed(seed=0)

        dyn = cellconstructor.Phonons.Phonons(str(dyn_prefix), nqirr)
        dyn.Symmetrize()
        dyn.ForcePositiveDefinite()

        ensemble = sscha.Ensemble.Ensemble(dyn, 0)
        minim = sscha.SchaMinimizer.SSCHA_Minimizer(
            ensemble,
            kong_liu_ratio=0.5,
            meaningful_factor=1e-1,
            root_representation="sqrt",
        )
        minim.set_minimization_step(0.1)
        minim.print_info()

        relax = sscha.Relax.SSCHA(
            minimizer=minim,
            ase_calculator=calc,
            N_configs=n_configs,
            max_pop=max_pop,
            save_ensemble=True,
            save_dyn=True,
        )

        ioinfo = sscha.Utilities.IOInfo()
        ioinfo.SetupSaving("minim_info")
        relax.setup_custom_functions(custom_function_post=ioinfo.CFP_SaveAll)

        relax.relax(get_stress=True)

        # SSCHA saves intermediate dyn matrices as dyn_pop{N}_* but does not
        # write a "final_dyn" file.  Save it explicitly so the hessian step
        # can locate the converged dynamical matrix by a predictable name.
        relax.minim.dyn.save_qe("final_dyn")
    finally:
        os.chdir(orig_dir)

    return output_dir
