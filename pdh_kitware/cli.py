from pathlib import Path

import click


@click.group()
def main():
    """PDH Kitware SSCHA pipeline.

    Run individual pipeline stages or the full end-to-end workflow.
    The NEQUIP_MODEL_PATH environment variable sets the default model location.

    Example full run:

        pdh-sscha fullrun /workspace/data/PdH.cif /workspace/data/hessian_dyn
    """


# ---------------------------------------------------------------------------
# build-nequip
# ---------------------------------------------------------------------------


@main.command("build-nequip")
@click.option(
    "--model",
    "model_path",
    default=None,
    envvar="NEQUIP_MODEL_PATH",
    help="Destination path for the compiled model.",
    show_envvar=True,
)
def cmd_build_nequip(model_path):
    """Compile the NequIP model from the registry (no-op if already present)."""
    from pdh_kitware.nequip_utils import ensure_model

    path = ensure_model(model_path)
    click.echo(f"Model ready at {path}")


# ---------------------------------------------------------------------------
# relax-geometry
# ---------------------------------------------------------------------------


@main.command("relax-geometry")
@click.argument("input_cif", type=click.Path(exists=True, path_type=Path))
@click.argument(
    "output_cif", required=False, default=None, type=click.Path(path_type=Path)
)
@click.option(
    "--model",
    "model_path",
    default=None,
    envvar="NEQUIP_MODEL_PATH",
    help="Path to compiled NequIP model (built automatically if absent).",
    show_envvar=True,
)
def cmd_relax_geometry(input_cif, output_cif, model_path):
    """Relax INPUT_CIF and write the result to OUTPUT_CIF.

    OUTPUT_CIF defaults to <input_stem>_relaxed.cif next to the input file.
    """
    from pdh_kitware.harmonic.relax_geometry import optimize
    from pdh_kitware.nequip_utils import load_calculator

    calc = load_calculator(model_path)
    result = optimize(input_cif, calc, output_cif)
    click.echo(f"Relaxed structure written to {result}")


# ---------------------------------------------------------------------------
# phonons-harmonic
# ---------------------------------------------------------------------------


@main.command("phonons-harmonic")
@click.argument("input_cif", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output-prefix",
    default=None,
    type=click.Path(path_type=Path),
    help=(
        "Prefix for QE dyn output files (default: <cif_dir>/{k}x{k}x{k}_harmonic_dyn)."
    ),
)
@click.option(
    "--supercell",
    default=None,
    nargs=3,
    type=int,
    metavar="NX NY NZ",
    help="Supercell dimensions (default: auto from kspacing=0.125).",
)
@click.option(
    "--model",
    "model_path",
    default=None,
    envvar="NEQUIP_MODEL_PATH",
    help="Path to compiled NequIP model (built automatically if absent).",
    show_envvar=True,
)
def cmd_phonons_harmonic(input_cif, output_prefix, supercell, model_path):
    """Compute harmonic phonons from a relaxed INPUT_CIF.

    Writes numbered QE dyn files using OUTPUT_PREFIX as the base name.
    """
    from pdh_kitware.harmonic.phonons_harmonic import phonons
    from pdh_kitware.nequip_utils import load_calculator

    calc = load_calculator(model_path)
    sc = tuple(supercell) if supercell else None
    result = phonons(input_cif, calc, supercell=sc, output_prefix=output_prefix)
    click.echo(f"Harmonic dyn files written with prefix {result}")


# ---------------------------------------------------------------------------
# run-sscha
# ---------------------------------------------------------------------------


@main.command("run-sscha")
@click.argument("dyn_prefix", type=click.Path(path_type=Path))
@click.option(
    "--output-dir",
    default=None,
    type=click.Path(path_type=Path),
    help="Directory for final_dyn* and ensemble data (default: current dir).",
)
@click.option(
    "--nqirr",
    default=None,
    type=int,
    help="Number of irreducible q-points (auto-detected if omitted).",
)
@click.option(
    "--n-configs",
    default=1000,
    show_default=True,
    help="Configurations per population.",
)
@click.option(
    "--max-pop", default=50, show_default=True, help="Maximum number of populations."
)
@click.option(
    "--model",
    "model_path",
    default=None,
    envvar="NEQUIP_MODEL_PATH",
    help="Path to compiled NequIP model (built automatically if absent).",
    show_envvar=True,
)
def cmd_run_sscha(dyn_prefix, output_dir, nqirr, n_configs, max_pop, model_path):
    """Run SSCHA minimization from DYN_PREFIX harmonic dyn files.

    DYN_PREFIX is the path prefix for the harmonic dyn files
    (e.g. /workspace/data/4x4x4_harmonic_dyn).

    Writes final_dyn* and data/dyn_gen_pop*  files into OUTPUT_DIR.
    """
    from pdh_kitware.nequip_utils import load_calculator
    from pdh_kitware.sscha.run_sscha import run_sscha

    calc = load_calculator(model_path)
    result = run_sscha(
        dyn_prefix,
        calc,
        output_dir=output_dir,
        nqirr=nqirr,
        n_configs=n_configs,
        max_pop=max_pop,
    )
    click.echo(f"SSCHA output written to {result}/")


# ---------------------------------------------------------------------------
# free-energy-hessian
# ---------------------------------------------------------------------------


@main.command("free-energy-hessian")
@click.argument("data_dir", type=click.Path(exists=True, path_type=Path))
@click.argument("final_dyn_prefix", type=click.Path(path_type=Path))
@click.argument("output_prefix", type=click.Path(path_type=Path))
@click.option(
    "--pop-id",
    default=None,
    type=int,
    help="Population ID to load (auto-detects the last one if omitted).",
)
@click.option(
    "--nqirr",
    default=None,
    type=int,
    help="Number of irreducible q-points (auto-detected if omitted).",
)
def cmd_free_energy_hessian(data_dir, final_dyn_prefix, output_prefix, pop_id, nqirr):
    """Compute the free-energy Hessian from a completed SSCHA run.

    DATA_DIR       directory holding the saved ensemble (dyn_gen_pop* files).
    FINAL_DYN_PREFIX  prefix of the final_dyn* files produced by run-sscha.
    OUTPUT_PREFIX  prefix for the output hessian dyn files.

    Does not require a NequIP model.
    """
    from pdh_kitware.hessian.free_energy_hessian import free_energy_hessian

    result = free_energy_hessian(
        data_dir,
        final_dyn_prefix,
        output_prefix,
        pop_id=pop_id,
        nqirr=nqirr,
    )
    click.echo(f"Hessian written with prefix {result}")


# ---------------------------------------------------------------------------
# fullrun  (end-to-end pipeline)
# ---------------------------------------------------------------------------


@main.command("fullrun")
@click.argument("input_cif", type=click.Path(exists=True, path_type=Path))
@click.argument("output_prefix", type=click.Path(path_type=Path))
@click.option(
    "--work-dir",
    default="pipeline_output",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Directory for all intermediate files.",
)
@click.option(
    "--n-configs",
    default=1000,
    show_default=True,
    help="Configurations per population.",
)
@click.option(
    "--max-pop", default=50, show_default=True, help="Maximum number of populations."
)
@click.option(
    "--model",
    "model_path",
    default=None,
    envvar="NEQUIP_MODEL_PATH",
    help="Path to compiled NequIP model (built automatically if absent).",
    show_envvar=True,
)
def cmd_fullrun(input_cif, output_prefix, work_dir, n_configs, max_pop, model_path):
    """Run the full pipeline from a CIF file to the free-energy Hessian.

    Stages: relax-geometry → phonons-harmonic → run-sscha → free-energy-hessian

    INPUT_CIF      starting crystal structure.
    OUTPUT_PREFIX  prefix for the final hessian dyn files.

    Intermediate files are placed under WORK_DIR.
    """
    from pdh_kitware.harmonic.phonons_harmonic import phonons
    from pdh_kitware.harmonic.relax_geometry import optimize
    from pdh_kitware.hessian.free_energy_hessian import free_energy_hessian
    from pdh_kitware.nequip_utils import load_calculator
    from pdh_kitware.sscha.run_sscha import run_sscha

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    calc = load_calculator(model_path)

    click.echo("[1/4] Relaxing geometry...")
    relaxed_cif = optimize(input_cif, calc, work_dir / "relaxed.cif")
    click.echo(f"      -> {relaxed_cif}")

    click.echo("[2/4] Computing harmonic phonons...")
    harmonic_prefix = phonons(
        relaxed_cif, calc, output_prefix=work_dir / "harmonic_dyn"
    )
    click.echo(f"      -> {harmonic_prefix}*")

    sscha_dir = work_dir / "sscha"
    click.echo("[3/4] Running SSCHA minimization...")
    run_sscha(
        harmonic_prefix,
        calc,
        output_dir=sscha_dir,
        n_configs=n_configs,
        max_pop=max_pop,
    )
    click.echo(f"      -> {sscha_dir}/")

    click.echo("[4/4] Computing free-energy Hessian...")
    result = free_energy_hessian(
        data_dir=sscha_dir / "data",
        final_dyn_prefix=sscha_dir / "final_dyn",
        output_prefix=output_prefix,
    )
    click.echo(f"      -> {result}*")
    click.echo("Pipeline complete.")


if __name__ == "__main__":
    main()
