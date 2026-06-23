from pathlib import Path

from PIL import Image


def pngs_to_pdf(input_dir: Path, output_pdf: Path) -> None:
    png_files = sorted(input_dir.glob("spectra_fooof*.png"), key=lambda p: p.name.lower())

    if not png_files:
        raise FileNotFoundError(f"No .png files found in: {input_dir}")

    images = []
    for png_path in png_files:
        with Image.open(png_path) as img:
            images.append(img.convert("RGB"))

    first_image, remaining_images = images[0], images[1:]
    # Pillow's PDF backend expects the JPEG saver to be registered for RGB pages.
    Image.init()
    if "JPEG" not in Image.SAVE:
        raise RuntimeError(
            "Pillow JPEG encoder is unavailable in this environment. "
            "Install JPEG support (e.g., libjpeg/libjpeg-turbo) and reinstall Pillow."
        )

    first_image.save(output_pdf, format="PDF", save_all=True, append_images=remaining_images)

    for image in images:
        image.close()

    print(f"Saved PDF: {output_pdf}")
    print(f"Included {len(png_files)} PNG files from: {input_dir}")


if __name__ == "__main__":
    input_dir = Path("out_figures_concat_single_channel")
    output_pdf = input_dir / "out_figures_concat_single_channel_all_plots.pdf"

    pngs_to_pdf(input_dir=input_dir, output_pdf=output_pdf)
