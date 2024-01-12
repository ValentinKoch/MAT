import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from pathlib import Path
from scipy.ndimage import label
import pandas as pd
import h5py
import tqdm
import slideio
import os
import re

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

def scale_negatives(arr):
    # Make a copy to avoid modifying the original array
    result = arr.copy()

    # Identify negative values
    negative_values = arr < 0

    # Scale negative values to the range 0-1
    min_val = arr[negative_values].min()
    max_val = arr[negative_values].max()
    
    # Avoid division by zero if all negative values are the same
    if min_val != max_val:
        result[negative_values] = 1-(arr[negative_values] - min_val) / (max_val - min_val)
    
    return result

def scale_positives(arr):
    # Make a copy to avoid modifying the original array
    result = arr.copy()

    # Identify negative values
    positive_values = arr > 0

    # Scale negative values to the range 0-1
    min_val = arr[positive_values].min()
    max_val = arr[positive_values].max()
    
    # Avoid division by zero if all negative values are the same
    if min_val != max_val:
        result[positive_values] = (arr[positive_values] - min_val) / (max_val - min_val)
    
    return result

def map_risk_score(image_id,attention_values,previews,orig_downscales,feature_dir="/mnt/ceph_vol/MAT/feats",orig_tile_size=256,orig_downscale=4):
    file= sorted(Path(feature_dir).glob(f"*{image_id}*.h5"))[0]
    all_ims=[]
    with h5py.File(file, 'r') as h5_file:
        # Assuming the data is stored in the root of the HDF5 file
        # Change 'dataset_name' to the actual name of your dataset # Adjust this if necessary
        coords,slide_sizes = h5_file["coords"],h5_file["slide_sizes"]
        # Return the first entry in the dataset
        all_coord_len=0
        scenes=[sorted(np.unique([int(c[0]) for c in coords]))[0]]
        for i in scenes:
            prev_im=np.array(Image.open(previews[i]))
            scale=get_scaling(prev_im,slide_sizes[i])
            filtered_coords=[c for c in list(coords) if c[0]==i]
            c_l=len(filtered_coords)
            width=np.uint64(prev_im.shape[1])
            length=np.uint64(prev_im.shape[0])
            im=np.zeros((width,length))
            scale_x,scale_y=get_scaling(prev_im,slide_sizes[i])
            tile_size=(orig_downscales[i]*orig_tile_size/scale_x,orig_downscales[i]*orig_tile_size/scale_y)
            for coord,attention in zip(filtered_coords,attention_values[all_coord_len:all_coord_len+c_l]):

                y,x=int(orig_downscales[i]*coord[1]/scale_x),int(orig_downscales[i]*coord[2]/scale_y)
                im[x:int(np.ceil(x+tile_size[0])),y:int(np.ceil(y+tile_size[1]))]=attention

            all_coord_len+=c_l
            im=scale_negatives(im)
            all_ims.append(im)
            #all_ims.append(im/np.nanmax(np.abs(im)))
    return all_ims

def get_scaling(prev_im,slide_size):
    factor_a=slide_size[1]/prev_im.shape[0]
    factor_b=slide_size[0]/prev_im.shape[1]
    return factor_a,factor_b 

def get_orig_downscale(slide_files,resolution_in_mpp=0.5,driver="SVS"):
    all_scales=[]
    for slide_file in slide_files:
        slide = slideio.Slide(str(slide_file), driver)
        for scn in range(slide.num_scenes):
            scene = slide.get_scene(scn)
            all_scales.append(resolution_in_mpp / (scene.resolution[0] * 1e06))
    return all_scales


def map_attentions(image_id,attention_values,previews,orig_downscales,feature_dir="/mnt/ceph_vol/MAT/feats",orig_tile_size=256):
    file= sorted(Path(feature_dir).glob(f"*{image_id}*.h5"))[0]
    all_ims=[]
    with h5py.File(file, 'r') as h5_file:

        coords,slide_sizes = h5_file["coords"],h5_file["slide_sizes"]
        # Return the first entry in the dataset
        #print(coords.shape)
        all_coord_len=0
        scenes=[sorted(np.unique([int(c[0]) for c in coords]))[0]]
        for i in scenes:
            if i==0:
                prev_im=np.array(Image.open(previews[i]))
                scale_x,scale_y=get_scaling(prev_im,slide_sizes[i])
                filtered_coords=[c for c in list(coords) if c[0]==i]
                c_l=len(filtered_coords)
                width=np.uint64(prev_im.shape[1])
                length=np.uint64(prev_im.shape[0])
                im=np.zeros((width,length))
                tile_size=(orig_downscales[i]*orig_tile_size/scale_x,orig_downscales[i]*orig_tile_size/scale_y)
                #im=np.zeros(np.uint64(slide_sizes[i]/orig_downscale))
                if attention_values.shape[0]==4:
                    attention_values=np.average(attention_values,axis=0)
                for j,(coord,attention) in enumerate(zip(filtered_coords,attention_values[all_coord_len:all_coord_len+c_l])):
                    y,x=int(orig_downscales[i]*coord[1]/scale_x),int(orig_downscales[i]*coord[2]/scale_y)
                    im[x:int(np.ceil(x+tile_size[0])),y:int(np.ceil(y+tile_size[1]))]=attention
                im=scale_positives(im)
                all_coord_len+=c_l
                all_ims.append(im/np.nanmax(np.abs(im)))
    return all_ims




def find_files(directory, pattern1, pattern2="DX",pattern3=".svs"):
    matches = []
    compiled_pattern1 = re.compile(pattern1)
    compiled_pattern2 = re.compile(pattern2)
    compiled_pattern3 = re.compile(pattern3)
    
    for root, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if compiled_pattern1.search(filename) and compiled_pattern2.search(filename) and compiled_pattern3.search(filename):
                matches.append(os.path.join(root, filename))
    return matches

# Usage

if __name__=="__main__":



    slide_dir= '/mnt/ceph_vol/MAT'
    dataset_list=["ucec","brca","blca","gbmlgg","luad"]
    save_folder=Path("/mnt/ceph_vol/surv_attention")
    #preview_dir="/mnt/ceph_vol/"
    preview_dir="/mnt/ceph_vol/MAT/prev"
    #orig_downscale=2
    for dataset in dataset_list:
        done=[]
        #attention_folder="/mnt/ceph_vol/MAT/results/5foldcv/MTF_nll_surv_a0.0_lr2e-05_5foldcv_gc16_concat/tcga_"+dataset+"_MTF_nll_surv_a0.0_lr2e-05_5foldcv_gc16_concat_s1/bottom_attention"
        #risk_folder="/mnt/ceph_vol/MAT/results/5foldcv/MTF_nll_surv_a0.0_lr2e-05_5foldcv_gc16_concat/tcga_"+dataset+"_MTF_nll_surv_a0.0_lr2e-05_5foldcv_gc16_concat_s1/patch_risk"
        attention_folder="/mnt/ceph_vol/MAT/att"
        risk_folder="/mnt/ceph_vol/MAT/risk"
        dataset_save_folder=save_folder/dataset
        dataset_save_folder.mkdir(parents=True, exist_ok=True)

        dataset_preview_folder="/mnt/ceph_vol/MAT/prev"#preview_dir+dataset.upper()+"/preview"
        attention_files=sorted(Path(attention_folder).glob("*.npy"))
        
        for attention_file in tqdm.tqdm(attention_files):
            attention_values=np.load(attention_file)
            if np.max(attention_values.shape)>=10000:
                break
            search_term1 = attention_file.name.replace(".npy","")
            slide_files = sorted(find_files(slide_dir, search_term1))

            orig_downscales=get_orig_downscale(slide_files)
            image_id=attention_file.name.replace(".npy","")
            previews=sorted(Path(dataset_preview_folder).glob(f"*{image_id}*.png"))
            risk_values=risk_folder+"/"+attention_file.name
            

            risk_values=np.load(risk_values)
            attentions_mapped=map_attentions(image_id,attention_values,previews,orig_downscales=orig_downscales)
            risks_mapped=map_risk_score(image_id,risk_values,previews,orig_downscales=orig_downscales)
            # for attention_mapped,preview in zip(attentions_mapped,previews):
            #     prev_im=np.array(Image.open(preview))
            #     #joint=np.multiply(risk_mapped,attention_mapped)
            #     #joint=joint/np.max(joint)
            #     attention_mapped=attention_mapped/np.max(attention_mapped)
                
            #     overlay_colormap_on_rgb(prev_im,attention_mapped,image_id+'sc',dataset_save_folder,None)
            #     #overlay_colormap_on_rgb(prev_im,risk_mapped,image_id+"_risk",dataset_save_folder,None)
            for risk_mapped,attention_mapped,preview in zip(risks_mapped,attentions_mapped,previews):
                prev_im=np.array(Image.open(preview))
                joint=np.multiply(risk_mapped,attention_mapped)
                joint=joint/np.max(joint)
                attention_mapped=attention_mapped/np.max(attention_mapped)
                
                overlay_colormap_on_rgb(prev_im,attention_mapped,image_id+'sc',dataset_save_folder,None)
                overlay_colormap_on_rgb(prev_im,risk_mapped,image_id+"_risk",dataset_save_folder,None)
                overlay_colormap_on_rgb(prev_im,joint,image_id+"_joint",dataset_save_folder,None)

            print(attention_file.name)
    
