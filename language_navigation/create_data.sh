version="v0.12"

# generate xml files
python language_navigation/generate_language_xml_route.py \
    leaderboard/data/bench2drive_split \
    --output leaderboard/data/language_benchmark/instruction_following_${version}

# copy selected routes
python language_navigation/copy_selected_routes.py \
    --select-file language_navigation/route.txt \
    --source-dir leaderboard/data/language_benchmark/instruction_following_${version} \
    --output-dir leaderboard/data/language_benchmark/instruction_following_${version}_selected
# subset the data
python language_navigation/copy_selected_routes.py \
    --select-file language_navigation/route_subset.txt \
    --source-dir leaderboard/data/language_benchmark/instruction_following_${version} \
    --output-dir leaderboard/data/language_benchmark/instruction_following_${version}_subset

# visualize for debug
mkdir -p debug/${version}
python language_navigation/route_xml_bev.py \
    --input-dir leaderboard/data/language_benchmark/instruction_following_${version}_selected \
    --output debug/${version}