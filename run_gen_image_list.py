import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate image_list.txt for each immediate subdirectory under a root dir. "
            "If a subdirectory contains a 'color' folder, list all .jpg/.png files in "
            "that folder using paths relative to the root dir."
        )
    )
    parser.add_argument(
        "dir",
        type=Path,
        help="Root directory whose immediate subdirectories will be scanned.",
    )
    return parser.parse_args()


def is_target_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in {".jpg", ".png"}


def main():
    args = parse_args()
    root_dir = args.dir.resolve()

    if not root_dir.exists() or not root_dir.is_dir():
        raise ValueError(f"Invalid dir: {root_dir}")

    generated = 0
    scanned = 0

    for subdir in sorted(p for p in root_dir.iterdir() if p.is_dir()):
        scanned += 1
        color_dir = subdir / "color"
        if not color_dir.is_dir():
            continue

        image_paths = sorted(p for p in color_dir.iterdir() if is_target_image(p))
        relative_paths = [p.relative_to(root_dir).as_posix() for p in image_paths]

        out_path = subdir / "image_list.txt"
        out_path.write_text(
            ("\n".join(relative_paths) + "\n") if relative_paths else "",
            encoding="utf-8",
        )
        generated += 1
        print(f"wrote {out_path} ({len(relative_paths)} images)")

    print(f"done: scanned_subdirs={scanned}, generated_lists={generated}")


if __name__ == "__main__":
    main()
