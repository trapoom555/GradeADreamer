#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 [-opt|--option <value>] [--gpu <gpu_id>]"
    exit 1
}

# Check if no arguments were provided
if [ $# -eq 0 ]; then
    usage
fi

# Initialize the variable
OPTION=""

# Parse the provided option
while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in -opt|--option)
        OPTION="$2"
        if [ -z "$OPTION" ]; then
            echo "Error: No value provided for the option."
            usage
        fi
        shift # Move past the value
        ;;
        --gpu)
            GPU_ID="$2"
            if [ -z "$GPU_ID" ]; then
                echo "Error: No GPU ID provided."
                usage
            fi
            shift # Move past the value
            ;;
        *)
        echo "Error: Invalid option '$1'"
        usage
        ;;
    esac
    shift # Move to the next argument
done

# Echo the stored value
echo "The value of the option is: $OPTION"

FOLDER=logs/$OPTION

# Check if the folder exists
if [ -d "$FOLDER" ]; then
    # Delete the folder if it exists
    rm -rf "$FOLDER"
    echo "Deleted existing folder: $FOLDER"
fi

# Recreate the folder
mkdir -p "$FOLDER"
echo "Recreated folder: $FOLDER"

TIME_REPORT="logs/${OPTION}/time_report.txt"

# Clear the output file if it exists
> $TIME_REPORT

START_TIME=$(date +%s) # Get the start time in seconds

echo "[INFO] Running Stage 1... : Create Prior Point Clouds [MVDream + SDS]"
python main_prior.py --config configs/$OPTION/prior.yaml --gpu $GPU_ID
echo "[INFO] Running Stage 2... : Gaussian Splatting Optimization [Stable Diffusion + SDS]"
python main_gs.py --config configs/$OPTION/gs.yaml --gpu $GPU_ID
echo "[INFO] Running Stage 3... : Texture Optimization [Stable Diffusion + SDS]"
python main_appearance.py --config configs/$OPTION/appearance.json --gpu $GPU_ID

END_TIME=$(date +%s) # Get the end time in seconds
ELAPSED_TIME=$((END_TIME - START_TIME)) # Calculate elapsed time in seconds
ELAPSED_HUMAN=$(printf '%02d:%02d:%02d' $((ELAPSED_TIME/3600)) $((ELAPSED_TIME%3600/60)) $((ELAPSED_TIME%60))) # Convert to HH:MM:SS format
echo "Run with argument '$OPTION': ${ELAPSED_HUMAN} (HH:MM:SS)" >> $TIME_REPORT

echo "[INFO] Saving VDO..."
kire logs/$OPTION/appearance/dmtet_mesh/mesh.obj --save_video logs/$OPTION/${OPTION}_output_vdo.mp4 --wogui
echo "Finished : the saved VDO is located at logs/$OPTION/${OPTION}_output_vdo.mp4"

echo "Time report saved to $TIME_REPORT"