from eval_utils import StableDiffusion, seed_everything
from cleanfid import fid
import cleanfid
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def extract_frames(input_gif, output_prefix, frames_to_extract):
    with Image.open(input_gif) as img:
        # Ensure the image is a GIF
        if img.format != 'GIF':
            raise ValueError('The provided file is not a GIF.')

        # Calculate the total duration of the GIF
        total_duration = sum(img.info['duration'] for _ in range(img.n_frames))
        frame_duration = total_duration / frames_to_extract

        # Extract frames
        for i in range(frames_to_extract):
            # Set the GIF to the appropriate frame
            img.seek(int((i * frame_duration) // img.info['duration']))

            # Save the frame as a separate image
            os.makedirs(f'{output_prefix}_{i:02d}', exist_ok=True)
            img.save(f'{output_prefix}_{i:02d}/image.bmp', 'BMP')

def eval_all():
    # take all gif names in eval/ folder
    sd = None
    dirs = os.listdir("eval/images/")
    fid_scores = {}
    for dir_ in dirs:
        # if file is dir
        dir_full = f"eval/images/{dir_}"
        fid_scores[dir_] = {}
        if os.path.isdir(dir_full):
            # verify if file exists
            if not os.path.exists(f"eval/images/{dir_}/sd/{dir_}_sd.bmp"):
                if sd is None:
                    seed_everything(0)
                    sd = StableDiffusion(device="cuda",
                        fp16=False,
                        vram_O=False,
                        sd_version="2.1",
                        hf_key=None
                    )
                os.makedirs(f"eval/images/{dir_}/sd", exist_ok=True)
                # get prompts and negative prompts from config json
                json_file = f"configs/{dir_}/appearance.json"
                with open(json_file) as f:
                    data = json.load(f)
                    prompt = data["text"]
                    negative_prompt = data["negative_text"]
                    [img] = sd.prompt_to_img(prompt, negative_prompt)
                    # save image in bmp
                    plt.imsave(f"eval/images/{dir_}/sd/{dir_}_sd.bmp", img)
            # get all gifs in dir
            files = os.listdir(dir_full)
            for file in files:
                if file.endswith(".gif"):
                    # create folder with 'file'_stablediffusion generated image and 'file'_split where gifs are split into 10
                    filename = file.split(".")[0]
                    
                    # split gif into 10
                    os.makedirs(f"eval/images/{dir_}/{filename}", exist_ok=True)
                    gif = f"eval/images/{dir_}/{file}"
                    extract_frames(gif, f"eval/images/{dir_}/{filename}/{filename}_split", 10)
                    # calculate fid averaged with 10 pictures
                    scores = []
                    for i in range(10):
                        score = fid.compute_fid(f"eval/images/{dir_}/sd/", f"eval/images/{dir_}/{filename}/{filename}_split_{i:02d}", mode="clean", model_name="clip_vit_b_32")
                        scores.append(score)
                    avg_score = np.mean(scores)
                    fid_scores[dir_][filename] = avg_score
    
    # save fid scores in json
    # but make the json formatted nicely
    with open("eval/fid_scores.json", "w") as f:
        json.dump(fid_scores, f, indent=4)

if __name__ == "__main__":
    eval_all()