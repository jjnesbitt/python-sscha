import re
from pathlib import Path

from cellconstructor.Phonons import Phonons
from sscha.Ensemble import Ensemble


def _detect_nqirr(prefix):
    prefix = Path(prefix)
    return len(sorted(prefix.parent.glob(f"{prefix.name}[0-9]*")))


def _detect_last_population(data_dir):
    """Return the highest population ID found in data_dir."""
    data_dir = Path(data_dir)
    pop_ids = set()
    for f in data_dir.iterdir():
        m = re.match(r"dyn_gen_pop(\d+)_", f.name)
        if m:
            pop_ids.add(int(m.group(1)))
    if not pop_ids:
        raise FileNotFoundError(f"No dyn_gen_pop* files found in {data_dir}")
    return max(pop_ids)


def free_energy_hessian(
    data_dir, final_dyn_prefix, output_prefix, pop_id=None, nqirr=None
):
    """Compute the free-energy Hessian from a completed SSCHA run.

    data_dir: directory containing the saved ensemble (dyn_gen_pop* files).
    final_dyn_prefix: prefix for the final dynamical matrix files from SSCHA.
    output_prefix: prefix for the output hessian dyn files.
    pop_id: population iteration to load; auto-detects the last one if None.
    nqirr: number of irreducible q-points; auto-detected from files if None.
    """
    data_dir = Path(data_dir)
    final_dyn_prefix = Path(final_dyn_prefix)

    if pop_id is None:
        pop_id = _detect_last_population(data_dir)
        print(f"Auto-detected last population: {pop_id}")

    pop_prefix = data_dir / f"dyn_gen_pop{pop_id}_"

    if nqirr is None:
        nqirr = _detect_nqirr(pop_prefix)
        if nqirr == 0:
            raise FileNotFoundError(f"No files found matching {pop_prefix}[0-9]*")
        print(f"Auto-detected nqirr={nqirr}")

    dyn_initial = Phonons(str(pop_prefix), nqirr)
    dyn_final = Phonons(str(final_dyn_prefix), nqirr)

    ensemble = Ensemble(dyn_initial, 0)
    ensemble.load_bin(str(data_dir), population_id=pop_id)
    ensemble.update_weights(dyn_final, 0)

    dyn_hessian = ensemble.get_free_energy_hessian(
        include_v4=False, get_full_hessian=True, verbose=True
    )
    dyn_hessian.save_qe(str(output_prefix))
    return Path(output_prefix)
