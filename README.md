# HySpe_CTEL_AI
Snapshot Spectral Imaging Face Anti-spoofing Challenge First Place

# Prepare data
The "data_root" and "img_prefix" in the configuration file(work_dirs/Hy/*) should be specified

# Train
python train_swad.py work_dirs/Hy/cls_ress14_acer_50e_test_augv7_swad.py 
python train_swad.py work_dirs/Hy/cls_ress14_acer_50e_test_augv7_dropc_swad.py

# Infer
python infer.py work_dirs/Hy/cls_ress14_acer_50e_test_augv7_dropc_swad.py --load_from work_dirs/Hy/cls_ress14_acer_50e_test_augv7_dropc_swad/latest.pth
python infer.py work_dirs/Hy/cls_ress14_acer_50e_test_augv7_swad.py --load_from work_dirs/Hy/cls_ress14_acer_50e_test_augv7_swad/latest.pth

# Generate Result File
python tools/generate.py

final use 'tools/generate/result.txt' as submission
