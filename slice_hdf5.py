"""
Slice HDF5 file to first N timesteps.
Useful for visualizing large files that crash VS Code.

Example usage:
  python slice_hdf5.py  # Default: slice first 100 timesteps from episode_0.hdf5 with (24,32) placeholder

  # Custom slice:
  from slice_hdf5 import slice_hdf5
  slice_hdf5('data/tomato_picking/episode_1.hdf5', 'data/episode_1_sliced.hdf5', n_slice=100)
  # With custom placeholder size:
  slice_hdf5('data/tomato_picking/episode_2.hdf5', 'data/episode_2_sliced.hdf5',
             n_slice=50, image_placeholder_size=(48, 64))
  # Full copy with image placeholders:
  slice_hdf5('data/tomato_picking/episode_3.hdf5', 'data/episode_3_copy.hdf5',
             full=True, image_placeholder_size=(24, 32))
"""

import h5py
import numpy as np
import os


def slice_hdf5(
    input_path, output_path, n_slice=50, image_placeholder_size=(24, 32), full=False
):
    """
    Copy HDF5 file with optional slicing and image placeholder reduction.

    Args:
        input_path: Path to input HDF5 file
        output_path: Path to output HDF5 file
        n_slice: Number of timesteps to slice (default: 50, ignored if full=True)
        image_placeholder_size: (height, width) for image placeholders (default: 24, 32)
        full: If True, copy full dataset without slicing (ignores n_slice)
    """
    if full:
        print(f"Copying full dataset from {input_path}")
    else:
        print(f"Slicing first {n_slice} timesteps from {input_path}")
    print(f"Output: {output_path}")
    print(f"Image placeholder size: {image_placeholder_size}")

    placeholder_h, placeholder_w = image_placeholder_size

    with h5py.File(input_path, "r") as f_in:
        # Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with h5py.File(output_path, "w") as f_out:
            # Copy all global attributes
            for key, value in f_in.attrs.items():
                f_out.attrs[key] = value

            # Recursively copy groups and datasets
            def copy_sliced(name, obj):
                """Copy object with slicing for datasets."""
                # Skip root group
                if name == "/":
                    return

                # Create parent groups if they don't exist
                parent_name = os.path.dirname(name)
                if parent_name != "/" and parent_name not in f_out:
                    # Create parent groups recursively
                    current_path = ""
                    for part in parent_name.split("/")[1:]:
                        current_path = (
                            f"{current_path}/{part}" if current_path else f"/{part}"
                        )
                        if current_path not in f_out:
                            f_out.create_group(current_path)

                if isinstance(obj, h5py.Dataset):
                    # Check if this is an image dataset
                    is_image = name.startswith("observations/images/")

                    shape = obj.shape
                    dtype = obj.dtype

                    if len(shape) > 0:
                        # Determine slice size
                        if full:
                            slice_size = shape[0]
                            slice_desc = "full"
                        else:
                            slice_size = min(n_slice, shape[0])
                            slice_desc = "sliced"

                        if is_image:
                            # Create placeholder for images
                            if len(shape) == 4:  # (T, H, W, C)
                                new_shape = (
                                    slice_size,
                                    placeholder_h,
                                    placeholder_w,
                                    3,
                                )
                                placeholder = np.zeros(new_shape, dtype=dtype)
                                data = placeholder
                            else:
                                # Unexpected image shape, fallback to slice
                                new_shape = (slice_size,) + shape[1:]
                                data = obj[:slice_size]

                            print(
                                f"  Image dataset: {name} {shape} -> {new_shape} ({slice_desc}, placeholder)"
                            )
                        else:
                            # Normal slice for non-image datasets
                            new_shape = (slice_size,) + shape[1:]
                            print(
                                f"  Dataset: {name} {shape} -> {new_shape} ({slice_desc})"
                            )
                            data = obj[:slice_size]

                        # Create new dataset
                        dset = f_out.create_dataset(name, data=data, dtype=dtype)

                        # Copy dataset attributes
                        for key, value in obj.attrs.items():
                            dset.attrs[key] = value
                    else:
                        # Scalar or empty dataset
                        print(f"  Dataset: {name} (scalar/empty)")
                        dset = f_out.create_dataset(name, data=obj[()], dtype=dtype)
                        for key, value in obj.attrs.items():
                            dset.attrs[key] = value

                elif isinstance(obj, h5py.Group):
                    # Create group if it doesn't exist
                    if name not in f_out:
                        f_out.create_group(name)

                    # Copy group attributes
                    for key, value in obj.attrs.items():
                        f_out[name].attrs[key] = value

            # Start recursive copy from root
            f_in.visititems(copy_sliced)

    print(f"Done. Output file created: {output_path}")


if __name__ == "__main__":
    # Configuration
    input_file = "data/tomato_picking/episode_180.hdf5"
    output_file = "data/episode_180_sliced.hdf5"  # New name to avoid overwriting
    slice_size = 100
    placeholder_size = (24, 32)  # 1/20 of original 480x640
    full_copy = True  # Set to True to copy full dataset without slicing

    if full_copy:
        # Copy full dataset with image placeholders
        slice_hdf5(
            input_file, output_file, full=True, image_placeholder_size=placeholder_size
        )
    else:
        # Slice first N timesteps
        slice_hdf5(input_file, output_file, slice_size, placeholder_size)
