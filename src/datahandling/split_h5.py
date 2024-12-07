from os.path import join

from sae_cooccurrence.streamlit import split_large_h5_files
from sae_cooccurrence.utils.set_paths import get_git_root

path = join(
    get_git_root(),
    "results/gpt2-small/res-jb-feature-splitting/blocks_8_hook_resid_pre_24576/n_batches_500/blocks_8_hook_resid_pre_24576_pca_for_streamlit_test",
)

split_large_h5_files(path)


# def split_h5(input_file: str, output1: str, output2: str):
#     with h5py.File(input_file, "r") as f:
#         # Split each dataset
#         with h5py.File(output1, "w") as f1, h5py.File(output2, "w") as f2:
#             for key in f.keys():
#                 data = load_dataset(f[key])
#                 mid = len(data) // 2
#                 f1.create_dataset(key, data=data[:mid])
#                 f2.create_dataset(key, data=data[mid:])

# def split_large_h5_files(directory: str):
#     for filename in os.listdir(directory):
#         if filename.endswith('.h5'):
#             file_path = os.path.join(directory, filename)
#             if os.path.getsize(file_path) > 100 * 1024 * 1024:  # Check if file is over 100 MB
#                 output1 = os.path.join(directory, f"{filename}_part1.h5")
#                 output2 = os.path.join(directory, f"{filename}_part2.h5")
#                 split_h5(file_path, output1, output2)
#                 print(f"Split {filename} into {output1} and {output2}")


# def join_h5(file1: str, file2: str, output_file: str):
#     with h5py.File(file1, "r") as f1, h5py.File(file2, "r") as f2:
#         with h5py.File(output_file, "w") as out:
#             for key in f1.keys():
#                 combined = np.concatenate([f1[key][:], f2[key][:]])
#                 out.create_dataset(key, data=combined)
