"""Resize calibration images without changing their aspect ratio."""
import argparse
from pathlib import Path

import cv2

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def main():
    parser = argparse.ArgumentParser(
        description='Resize same-aspect-ratio calibration images to a common resolution.'
    )
    parser.add_argument('--input', required=True, help='folder containing source images')
    parser.add_argument('--output', required=True, help='folder for resized copies')
    parser.add_argument('--width', type=int, required=True, help='target image width in pixels')
    parser.add_argument('--height', type=int, required=True, help='target image height in pixels')
    args = parser.parse_args()

    input_folder = Path(args.input)
    output_folder = Path(args.output)
    if args.width <= 0 or args.height <= 0:
        parser.error('--width and --height must be positive')
    if not input_folder.is_dir():
        parser.error(f'Input folder does not exist: {input_folder}')

    paths = sorted(path for path in input_folder.iterdir()
                   if path.suffix.lower() in EXTENSIONS)
    if not paths:
        parser.error(f'No supported images found in {input_folder}')

    target_ratio = args.width / args.height
    images = []
    incompatible = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            incompatible.append(f'{path.name}: unreadable image')
            continue
        height, width = image.shape[:2]
        ratio = width / height
        relative_difference = abs(ratio - target_ratio) / target_ratio
        if relative_difference > 0.001:
            incompatible.append(
                f'{path.name}: {width}x{height} aspect ratio {ratio:.4f} '
                f'does not match target ratio {target_ratio:.4f}'
            )
        images.append((path, image))

    if incompatible:
        print('Cannot safely normalize this set without distorting calibration geometry:')
        for message in incompatible:
            print(' -', message)
        print('Use original images with the same camera aspect ratio and orientation,')
        print('or recapture/export the photos consistently. Originals were not changed.')
        raise SystemExit(1)

    output_folder.mkdir(parents=True, exist_ok=True)
    for path, image in images:
        resized = cv2.resize(image, (args.width, args.height), interpolation=cv2.INTER_AREA)
        output_path = output_folder / path.name
        if not cv2.imwrite(str(output_path), resized):
            raise OSError(f'Could not write resized image: {output_path}')
        print(f'[OK] {path.name} -> {args.width}x{args.height}')

    print(f'Resized {len(images)} images to: {output_folder.resolve()}')
    print('Use this output only when all source images came from the same camera setup')
    print('and have the same aspect ratio; keep the original files unchanged.')


if __name__ == '__main__':
    main()