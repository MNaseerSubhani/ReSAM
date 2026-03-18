
cfg_file="configs.config_nwpu"
prompt="point"
load_type="soft"
num_points_list=(1)
output_dirs=("work_dir/nwpu/resam")

script_name="${1}.py"

# Check if the file actually exists before running
if [ ! -f "$script_name" ]; then
    echo "Error: File $script_name not found!"
    exit 1
fi


for output_dir in "${output_dirs[@]}"; do
    for num_points in "${num_points_list[@]}"; do   
        out_dir="${output_dir}/point_${num_points}"
        CUDA_VISIBLE_DEVICES=0 python "$script_name" --cfg "$cfg_file" --prompt "$prompt" --num_points "$num_points" --out_dir "$out_dir"  --load_type "$load_type"
    done
done

