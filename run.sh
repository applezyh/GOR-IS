# Replace {dataset_path}, e.g., .../gor-is-synthetic
python launcher.py --root_dir {dataset_path} \
  --data_list scene_1_colmap scene_2_colmap \
  --recon \
  --remove_object \
  --inpainting2D \
  --inpainting3D \
  --render_inpainting3D \
  --device_list 0

# --data_list: Specify the list of scenes to process; e.g., scene_1_colmap scene_2_colmap...
# --recon: Reconstruct the scene
# --remove_object: Remove the object from the scene; results are saved in model_path/train/.../removal
# --inpainting2D: Perform 2D inpainting; results are saved in model_path/train/.../inpainting
# --inpainting3D: Perform 3D inpainting
# --render_inpainting3D: Save rendered results in model_path/test
# --device_list: Supports multiple GPUs; specify GPU IDs (use -1 to use all GPUs)