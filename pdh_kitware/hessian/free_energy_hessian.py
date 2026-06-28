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


def _detect_final_dyn_prefix(sscha_dir):
    """Locate the converged dyn prefix in a run-sscha output directory.

    Prefers an explicit 'final_dyn' saved by our CLI wrapper.  Falls back to
    the last 'dyn_pop{N}_' written by SSCHA itself, which is what older runs
    (or runs not started through this CLI) will have.
    """
    sscha_dir = Path(sscha_dir)
    if list(sscha_dir.glob("final_dyn[0-9]*")):
        return sscha_dir / "final_dyn"

    pop_ids = set()
    for f in sscha_dir.glob("dyn_pop*_[0-9]*"):
        m = re.match(r"dyn_pop(\d+)_", f.name)
        if m:
            pop_ids.add(int(m.group(1)))
    if not pop_ids:
        raise FileNotFoundError(
            f"No final_dyn* or dyn_pop*_* files found in {sscha_dir}"
        )
    last = max(pop_ids)
    print(f"No final_dyn found; using last SSCHA population dyn_pop{last}_")
    return sscha_dir / f"dyn_pop{last}_"


def free_energy_hessian(sscha_dir, output_prefix, pop_id=None, nqirr=None, final_dyn_prefix=None):
    """Compute the free-energy Hessian from a completed SSCHA run.

    sscha_dir: output directory produced by run-sscha.  Expected to contain
        dyn_pop*_* (or final_dyn*) files and a data/ subdirectory with the
        saved ensemble.
    output_prefix: prefix for the output hessian dyn files.
    pop_id: population to load from the ensemble; auto-detects the last one.
    nqirr: number of irreducible q-points; auto-detected from files if None.
    final_dyn_prefix: override the auto-detected final dyn prefix.
    """
    sscha_dir = Path(sscha_dir)
    data_dir = sscha_dir / "data"

    if final_dyn_prefix is None:
        final_dyn_prefix = _detect_final_dyn_prefix(sscha_dir)
    else:
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
