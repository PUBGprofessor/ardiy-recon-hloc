
python run_gen_image_list.py f:/dev2/prjs1/data1/office3 
python run_build_map.py --config F:\dev2\prjs1\data1\office3\build_map.txt
python run_detect_markers.py   --image_dir f:/dev2/prjs1/data1/office3 --model_dir f:/dev2/prjs1/data1/office3/hloc_map/sfm_reference  --aruco_dict DICT_5X5_50 --min_views 2 --max_reproj_error 4.0
run_reloc_server.py --root_dir F:\dev2\prjs1\data1\office3 --map_dir hloc_map --default_camera_model PINHOLE --train_list scan1/image_list.txt --device cuda:0 --retvis

python run_reloc_server.py --root_dir F:\dev2\prjs1\data1\4031 --map_dir hloc_map --ip 101.76.210.138
