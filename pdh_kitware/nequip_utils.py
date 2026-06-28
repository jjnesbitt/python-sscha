import os
import subprocess
from pathlib import Path

DEFAULT_MODEL_PATH = os.getenv(
    "NEQUIP_MODEL_PATH", "/opt/nequip/NequIP-OAM-XL.nequip.pt2"
)
_NEQUIP_SOURCE = "nequip.net:mir-group/NequIP-OAM-XL:0.1"


def ensure_model(model_path=None):
    """Return the model path, compiling it from the registry if it doesn't exist."""
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    model_path = Path(model_path)
    if not model_path.exists():
        print(
            f"NequIP model not found at {model_path}, compiling (this may take a while)..."
        )
        model_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "nequip-compile",
                _NEQUIP_SOURCE,
                str(model_path),
                "--device",
                "cuda",
                "--mode",
                "aotinductor",
                "--target",
                "ase",
            ],
            check=True,
        )
        print(f"Model compiled to {model_path}")
    return model_path


def load_calculator(model_path=None):
    """Load a NequIP ASE calculator, building the model first if needed."""
    import openequivariance  # noqa: F401
    from nequip.integrations.ase import NequIPCalculator

    model_path = ensure_model(model_path)
    return NequIPCalculator.from_compiled_model(
        compile_path=str(model_path),
        device="cuda",
        chemical_species_to_atom_type_map=True,
    )
