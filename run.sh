#!/bin/bash

# Function to display usage information
usage() {
    echo "option: $0 [-opt|--option <value>]"
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

    case $key in
        -opt|--option)
        OPTION="$2"
        if [ -z "$OPTION" ]; then
            echo "Error: No value provided for the option."
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

echo "[INFO] Running Stage 1... : Create Prior Point Clouds [MVDream + SDS]"
python main_prior.py --config configs/$OPTION/prior.yaml
echo "[INFO] Running Stage 2... : Gaussian Splatting Optimization [Stable Diffusion + SDS]"
python main_gs.py --config configs/$OPTION/gs.yaml
echo "[INFO] Saving VDO..."
kire logs/$OPTION/${OPTION}_mesh.obj --save_video logs/$OPTION/${OPTION}_output_vdo.mp4 --wogui
echo "Finished : the saved VDO is located at logs/$OPTION/${OPTION}_output_vdo.mp4"