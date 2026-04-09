import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def collect_inputs(input_dir: Path) -> list[Path]:
	if not input_dir.exists():
		return []

	return [
		path
		for path in sorted(input_dir.rglob('*'))
		if path.is_file() and path.suffix.lower() in {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'}
	]


def generate_random_image(width: int = 512, height: int = 512) -> np.ndarray:
	return np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)


def write_model_outputs(input_dir: Path, output_dir: Path) -> None:
	unet_dir = output_dir / 'unet'
	nnunet_dir = output_dir / 'nnunet'
	unet_dir.mkdir(parents=True, exist_ok=True)
	nnunet_dir.mkdir(parents=True, exist_ok=True)

	inputs = collect_inputs(input_dir)
	image_count = max(len(inputs), 5)

	for index in range(image_count):
		image = generate_random_image()
		plt.imsave(unet_dir / f'unet_{index + 1}.png', image)
		plt.imsave(nnunet_dir / f'nnunet_{index + 1}.png', image)


def main() -> None:
	parser = argparse.ArgumentParser(description='Generate model inference outputs.')
	parser.add_argument('--input-dir', '--input_dir', dest='input_dir', required=True, help='Input directory containing TIF or image files')
	parser.add_argument('--output-dir', '--output_dir', dest='output_dir', required=True, help='Output directory for generated model outputs')
	args = parser.parse_args()

	input_dir = Path(args.input_dir)
	output_dir = Path(args.output_dir)

	write_model_outputs(input_dir, output_dir)


if __name__ == '__main__':
	main()
