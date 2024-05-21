#!/bin/bash

# Define the path to your Python script
PYTHON_SCRIPT_PATH="experiments.py"

# Define arrays for each argument with possible values
depths=(3) # GNN layers
models=("GIN" "GCN")
activations=("relu" "sigmoid" "silu") # Used by GNN
num_clean=(5)  # Single value, array for consistency in loop generation
num_dirty=(5)  # Single value, array for consistency
targets=("exponent" "sign" "mantissa") #"all"
combo_length=(1)  # Single value, array for consistency

bit_flip_percentage=(0.95 0.01 0.05 0.1 0.25 0.375 0.5 0.625 0.75 0.875)
datasets=("MUTAG" "AIDS" "PTC_FM" "PTC_MR" "NCI1" "PROTEINS" "ENZYMES" "MSRC_9" "MSRC_21C" "IMDB-BINARY")
# Calculate the total number of combinations
total_combinations=$((${#depths[@]} * ${#models[@]} * ${#activations[@]} * ${#num_clean[@]} * ${#num_dirty[@]} * ${#targets[@]} * ${#combo_length[@]} * ${#bit_flip_percentage[@]} * ${#datasets[@]}))
current_combination=0
new_combination=0

# Start time
start_time=$(date +%s)

convert_seconds() {
    local total_seconds=$1
    local weeks=$((total_seconds / 604800))
    local days=$(( (total_seconds % 604800) / 86400))
    local hours=$(( (total_seconds % 86400) / 3600))
    local minutes=$(( (total_seconds % 3600) / 60))
    local seconds=$((total_seconds % 60))
    echo "$weeks weeks, $days days, $hours hours, $minutes minutes, $seconds seconds"
}

# Generate and execute commands for all combinations
for d in "${depths[@]}"; do
    for m in "${models[@]}"; do
        for a in "${activations[@]}"; do
            for nc in "${num_clean[@]}"; do
                for nd in "${num_dirty[@]}"; do
                    for t in "${targets[@]}"; do
                        for cl in "${combo_length[@]}"; do
                            for bfpc in "${bit_flip_percentage[@]}"; do
                                for ds in "${datasets[@]}"; do
                                    let current_combination++
                                    bfpc_int=$(echo "scale=4; $bfpc*100" | bc)
                                    fname='model_'$m'_depth_'$d'_activation_'$a'_numClean_'$nc'_numDirty_'$nd'_target_'$t'_comboLength_'$cl'_bitFlipPerc_'${bfpc_int%.*}'_dataset_'$ds'.json'
                                    path="./results"  # Replace this with your desired path
                                    full_path="${path}/${fname}"
                                    if [ -e "$full_path" ]; then
                                      echo "File exists at path: $full_path"
                                    else
                                      let new_combination++
                                      current_time=$(date +%s)
                                      elapsed=$((current_time - start_time))
                                      progress=$(echo "$new_combination / $total_combinations" | bc -l)
                                      estimated_total=$(echo "$elapsed / $progress" | bc)
                                      estimated_remaining=$(echo "$estimated_total - $elapsed" | bc)
                                      elapsed_readable=$(convert_seconds $elapsed)
                                      estimated_remaining_readable=$(convert_seconds $estimated_remaining)
                                      echo "Running combination $current_combination / $total_combinations"
                                      echo "Configuration: Depth=$d, Model=$m, Activation=$a, NumClean=$nc, NumDirty=$nd, Target=$t, ComboLength=$cl, BitFlipPerc=$bfpc, Dataset=$ds"
                                      echo "Elapsed time: $elapsed_readable seconds, Estimated remaining time: $estimated_remaining_readable" #$elpased $estimated_remaining seconds"
                                      # Uncomment the following line to actually run Python script with the arguments
                                      python $PYTHON_SCRIPT_PATH -d $d -m $m -a $a -nc $nc -nd $nd -t $t -cl $cl -bfpc $bfpc -ds $ds
                                    fi
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

echo "All combinations have been executed."

