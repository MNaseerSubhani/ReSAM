
cfg_file="configs.config_hrsid"
prompt="point"
load_type="soft"

script_name="train_${1}.py"
num_points=$2

output_dirs=("work_dir/nwpu/${1}")

# Check if the file actually exists before running
if [ ! -f "$script_name" ]; then
    echo "Error: File $script_name not found!"
    exit 1
fi

# Check if points were actually provided
if [ -z "$num_points" ]; then
    echo "Error: Please provide the number of points. Example: bash scripts/train_resam_{dataset}.sh resam 2"
    exit 1
fi

for output_dir in "${output_dirs[@]}"; do
    out_dir="${output_dir}/point_${num_points}"
    CUDA_VISIBLE_DEVICES=0 python "$script_name" --cfg "$cfg_file" --prompt "$prompt" --num_points "$num_points" --out_dir "$out_dir"  --load_type "$load_type"
done
