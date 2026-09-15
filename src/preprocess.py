"""Chargement et préparation identiques des images pour l'entraînement et le test."""

from __future__ import annotations

from pathlib import Path

from torchvision import datasets, transforms


IMAGE_SIZE = 64
CLASS_NAMES = ["vide", "personne"]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}


def build_transform() -> transforms.Compose:
    """Redimensionne les images, les convertit en tenseurs et les normalise.

    ``ToTensor`` convertit une image RGB en trois matrices de nombres entre 0 et
    1. La normalisation centre ensuite ces valeurs autour de 0, ce qui aide le
    réseau à apprendre plus régulièrement.
    """

    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )


class PresenceImageFolder(datasets.ImageFolder):
    """ImageFolder avec l'ordre explicitement fixé : 0=vide, 1=personne.

    ``ImageFolder`` classe normalement les noms de dossiers par ordre
    alphabétique, ce qui donnerait ``personne=0``. Cette classe impose le
    convention du projet pour éviter toute confusion entre les labels.
    """

    def find_classes(self, directory: str | Path) -> tuple[list[str], dict[str, int]]:
        directory_path = Path(directory)
        missing = [name for name in CLASS_NAMES if not (directory_path / name).is_dir()]
        if missing:
            folders = ", ".join(missing)
            raise FileNotFoundError(
                f"Dossiers de classe absents dans {directory_path}: {folders}."
            )
        return CLASS_NAMES.copy(), CLASS_TO_INDEX.copy()


def load_dataset(directory: str | Path) -> PresenceImageFolder:
    """Charge un dossier ``train`` ou ``validation`` avec la préparation 64x64."""

    return PresenceImageFolder(root=str(directory), transform=build_transform())
