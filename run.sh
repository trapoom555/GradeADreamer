#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 [--prompt <value>] [--gpu <gpu_id>]"
    exit 1
}

# Check if no arguments were provided
if [ $# -eq 0 ]; then
    usage
fi

# Initialize the variable
PROMPT=""

# Parse the provided prompt
while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in --prompt)
        PROMPT="$2"
        if [ -z "$PROMPT" ]; then
            echo "Error: No value provided for the prompt."
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

if [ -z "$PROMPT" ]; then
    echo "Error: No prompt provided."
    usage
fi

if [ -z "$GPU_ID" ]; then
    echo "Error: No GPU ID provided."
    usage
fi

# for folder, replace prompt spaces with _
PROMPT_F="${PROMPT// /_}"

FOLDER=logs/$PROMPT_F

# Check if the folder exists
# if [ -d "$FOLDER" ]; then
#     # stop because prompt already generated
#     echo "WARNING: The prompt '$PROMPT' has already been generated."
#     exit 0
# fi

# Recreate the folder
mkdir -p "$FOLDER"
echo "Recreated folder: $FOLDER"

TIME_REPORT="logs/${PROMPT_F}/time_report.txt"

# Clear the output file if it exists
> $TIME_REPORT

START_TIME=$(date +%s) # Get the start time in seconds

echo "[INFO] Running Stage 1... : Create Prior Point Clouds [MVDream + SDS]"
python main_prior.py --prompt "$PROMPT" --gpu $GPU_ID
echo "[INFO] Running Stage 2... : Gaussian Splatting Optimization [Stable Diffusion + SDS]"
python main_gs.py --prompt "$PROMPT" --gpu $GPU_ID
echo "[INFO] Running Stage 3... : Texture Optimization [Stable Diffusion + SDS]"
python main_appearance.py --prompt "$PROMPT" --gpu $GPU_ID

END_TIME=$(date +%s) # Get the end time in seconds
ELAPSED_TIME=$((END_TIME - START_TIME)) # Calculate elapsed time in seconds
ELAPSED_HUMAN=$(printf '%02d:%02d:%02d' $((ELAPSED_TIME/3600)) $((ELAPSED_TIME%3600/60)) $((ELAPSED_TIME%60))) # Convert to HH:MM:SS format
echo "Run with argument '$PROMPT': ${ELAPSED_HUMAN} (HH:MM:SS)" >> $TIME_REPORT

# echo "[INFO] Saving VDO..."
# kire logs/$PROMPT_F/appearance/dmtet_mesh/mesh.obj --save_video logs/$PROMPT_F/${PROMPT_F}_output_vdo.mp4 --wogui
# echo "Finished : the saved VDO is located at logs/$PROMPT_F/${PROMPT_F}_output_vdo.mp4"

echo "Time report saved to $TIME_REPORT"