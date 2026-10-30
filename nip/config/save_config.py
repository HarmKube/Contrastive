from pathlib import Path
import dataclasses, yaml

def save_config(cfg, file: Path) -> None:
    """
    Schreibt den Inhalt der Dataclass-Konfiguration als YAML.

    Parameters
    ----------
    cfg  : Beliebige Dataclass (hier: Config)
    file : Zielpfad (z. B. run_dir / "config.yaml")
    """
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("w") as fh:
        yaml.safe_dump(
            dataclasses.asdict(cfg),
            fh,
            sort_keys=False,
            default_flow_style=False,
        )