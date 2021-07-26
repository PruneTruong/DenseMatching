conda activate dense_matching_env
mkdir -p data
cd data/
mkdir ETH3D
cd ETH3D/
mkdir multiview_testing/
mkdir multiview_training/
wget https://www.eth3d.net/data/multi_view_training_rig.7z -P multiview_training/
wget https://www.eth3d.net/data/multi_view_test_rig_undistorted.7z -P multiview_testing/
cd multiview_testing/
7z x multi_view_test_rig_undistorted.7z
rm multi_view_test_rig_undistorted.7z
cd ../multiview_training/
7z x multi_view_training_rig.7z
rm multi_view_training_rig.7z
cd ..
gdown https://drive.google.com/uc?id=1Okqs5QYetgVu_HERS88DuvsABGak08iN
unzip info_ETH3D_files.zip
rm info_ETH3D_files.zip



