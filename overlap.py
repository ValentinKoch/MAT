import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from pathlib import Path
from scipy.ndimage import label
import pandas as pd
import h5py
import tqdm

def get_high_attention_patch(attention_map,slide_folder,image_name,save_path):
    return 


def overlay_colormap_on_rgb(rgb_image, heatmap_image, image_name,save_path,high_att_coord, cmap='viridis', alpha=0.4):

    alpha_channel = np.ones((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.uint8) * 255
    rgba_rgb_image = np.dstack((rgb_image, alpha_channel))
    cmap = plt.get_cmap(cmap)
    heatmap_colored = (cmap(heatmap_image) * 255).astype(np.uint8)
    rotated_heatmap_coloured = np.fliplr(np.rot90(heatmap_colored, k=3))
    resized_binary_mask = np.rot90(heatmap_image, k=3)
    resized_heatmap = Image.fromarray(rotated_heatmap_coloured).resize((rgb_image.shape[1],rgb_image.shape[0]), Image.BILINEAR)
    resized_binary_mask=Image.fromarray(resized_binary_mask).resize((rgb_image.shape[1],rgb_image.shape[0]), Image.BILINEAR)
    pil_rgb_image = Image.fromarray(rgba_rgb_image)
    overlaid_image = Image.blend(pil_rgb_image, resized_heatmap, alpha)
    resized_binary_mask_expanded=np.expand_dims(np.flip(resized_binary_mask,1),-1)
    overlaid_image=np.array(overlaid_image)
    zero_array = np.ones_like(overlaid_image)*255
    overlaid_image = np.where(resized_binary_mask_expanded, overlaid_image, zero_array)
    overlaid_image[:,:,3]=255
    overlaid_image=Image.fromarray(overlaid_image,mode='RGBA')
    if save_path is not None:
        overlaid_image.save(Path(save_path)/(image_name+".png"))
        resized_heatmap.save(Path(save_path)/(image_name+"_heat.png"))
        #overlaid_image.save(Path(save_path)/image_name.replace(".npy",".png"))
    return pil_rgb_image

def map_risk_score(image_id,attention_values,feature_dir="/mnt/ceph_vol/UCEC/features",feature_size=256,orig_downscale=4):
    file= sorted(Path(feature_dir).glob(f"*{image_id}*.h5"))[0]
    all_ims=[]
    with h5py.File(file, 'r') as h5_file:
        # Assuming the data is stored in the root of the HDF5 file
        # Change 'dataset_name' to the actual name of your dataset # Adjust this if necessary
        coords,slide_sizes = h5_file["coords"],h5_file["slide_sizes"]
        # Return the first entry in the dataset
        all_coord_len=0
        scenes=list(np.unique([int(c[0]) for c in coords]))
        for i in scenes:
            filtered_coords=[c for c in list(coords) if c[0]==i]
            c_l=len(filtered_coords)
            im=np.zeros(np.uint64(slide_sizes[i]/orig_downscale))
            for coord,attention in zip(filtered_coords,attention_values[all_coord_len:all_coord_len+c_l]):
                y,x=int(coord[1]),int(coord[2])
                im[x:x+feature_size,y:y+feature_size]=attention
            all_coord_len+=c_l
            im=scale_negatives(im)
            all_ims.append(im)
            #all_ims.append(im/np.nanmax(np.abs(im)))
    return all_ims


def map_attentions(image_id,attention_values,feature_dir="/mnt/ceph_vol/UCEC/features",feature_size=256,orig_downscale=4):
    file= sorted(Path(feature_dir).glob(f"*{image_id}*.h5"))[0]
    all_ims=[]
    with h5py.File(file, 'r') as h5_file:
        # Assuming the data is stored in the root of the HDF5 file
        # Change 'dataset_name' to the actual name of your dataset # Adjust this if necessary
        coords,slide_sizes = h5_file["coords"],h5_file["slide_sizes"]
        # Return the first entry in the dataset
        all_coord_len=0
        scenes=sorted(np.unique([int(c[0]) for c in coords]))
        for i in scenes:
            filtered_coords=[c for c in list(coords) if c[0]==i]
            c_l=len(filtered_coords)
            im=np.zeros(np.uint64(slide_sizes[i]/orig_downscale))
            if attention_values.shape[0]==4:
                attention_values=np.average(attention_values,axis=0)
            for coord,attention in zip(filtered_coords,attention_values[all_coord_len:all_coord_len+c_l]):
                y,x=int(coord[1]),int(coord[2])
                im[x:x+feature_size,y:y+feature_size]=attention
            all_coord_len+=c_l
            all_ims.append(im/np.nanmax(np.abs(im)))
    return all_ims


if __name__=="__main__":
    dataset_list=["ucec","brca","blca","gbmlgg","luad"]
    save_folder=Path("/mnt/ceph_vol/surv_attention")
    preview_dir="/mnt/ceph_vol/"

    for dataset in dataset_list:
        done=[]
        attention_folder="results/5foldcv/MTF_nll_surv_a0.0_lr2e-05_5foldcv_gc16_concat/tcga_"+dataset+"_MTF_nll_surv_a0.0_lr2e-05_5foldcv_gc16_concat_s1/bottom_attention"
        dataset_save_folder=save_folder/dataset
        dataset_save_folder.mkdir(parents=True, exist_ok=True)

        dataset_preview_folder=preview_dir+dataset.upper()+"/preview"
        attention_files=sorted(Path(attention_folder).glob("*.npy"))

        for attention_file in tqdm.tqdm(attention_files):
            image_id=attention_file.name.replace(".npy","")
            attention_values=np.load(attention_file)
            attentions_mapped=map_attentions(image_id,attention_values)
            previews=sorted(Path(dataset_preview_folder).glob(f"*{image_id}*.png"))
            for attention_mapped,preview in zip(attentions_mapped,previews):
                prev_im=np.array(Image.open(preview))
                overlay_colormap_on_rgb(prev_im,attention_mapped,image_id,save_folder,None)
            print(attention_file.name)


