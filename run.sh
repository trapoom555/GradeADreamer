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

echo "Running Stage 1... : Gaussian Splatting with VSD"
python main.py --config configs/$OPTION.yaml
echo "Running Stage 2... : Gaussian Splatting with VSD"
python main2.py --config configs/$OPTION.yaml
echo "Saving VDO : Gaussian Splatting with VSD"
kire logs/$OPTION/${OPTION}_refined_mesh.obj --save_video logs/$OPTION/${OPTION}_output_vdo.mp4 --wogui
echo "Finished : the saved VDO is located at logs/$OPTION/${OPTION}_output_vdo.mp4"