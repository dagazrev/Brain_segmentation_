import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import nibabel as nib
from skimage import exposure
from skimage.morphology import disk
from skimage.morphology import ball
from skimage.filters import rank
from skimage.restoration import denoise_nl_means, estimate_sigma
from scipy.ndimage import gaussian_filter
import SimpleITK as sitk
import csv
import data_handling as dh
import initial_utils as iu
from sklearn.metrics import mutual_info_score


def local_weighting_segmentation(data_handler, similarity_metric='MI'):
    """
    Perform local weighting segmentation on validation images using registered atlas images and calculate Dice Score Coefficients.
    """
    validation_data = data_handler.retrieve_data('validation', 0)

    for v_image_path, v_label_path, _ in validation_data:
        v_folder_name = os.path.basename(os.path.dirname(v_image_path))

        # Paths for registered atlas images and transformed labels
        
        atlas_image_paths = []
        transformed_label_paths = []

        atlases_paths = [os.path.join(os.path.dirname(v_label_path), file) for file in os.listdir(os.path.dirname(v_label_path)) if '_PREPR_BF_NL_EQ.nii.gz' in file]
        labels_paths = [os.path.join(os.path.dirname(v_label_path), file) for file in os.listdir(os.path.dirname(v_label_path)) if '_seg_transformed.nii.gz' in file]

        
        target_image = nib.load(v_image_path).get_fdata()
        final_segmentation = np.squeeze(np.zeros_like(target_image))

        weight_sum = 0  # To keep track of the total weight
        for atlas_path, label_path in zip(atlases_paths, labels_paths):
            atlas_image = nib.load(atlas_path).get_fdata()
            label_image = nib.load(label_path).get_fdata()

            # Select the similarity metric
            if similarity_metric == 'MI':
                similarity_weight = calculate_mutual_information(target_image, atlas_image)
            elif similarity_metric == 'NCC':
                similarity_weight = calculate_normalized_cross_correlation(target_image, atlas_image)
            else:
                raise ValueError("Unknown similarity metric specified")

            similarity_weight = calculate_mutual_information(target_image, atlas_image)
            weighted_label_image = similarity_weight * label_image
            final_segmentation += weighted_label_image
            weight_sum += similarity_weight

        # Avoid division by zero
        weight_sum += np.finfo(float).eps

        # Normalize and finalize the segmentation
        final_segmentation /= weight_sum
        final_segmentation = np.round(final_segmentation).astype(int)

        # Replace any NaN or inf values
        final_segmentation = np.nan_to_num(final_segmentation, nan=0, posinf=0, neginf=0)

        # Debug prints
        print(f"Weight Sum for {v_folder_name}: {weight_sum}")
        print(f"Max of final_segmentation after normalization: {np.max(final_segmentation)}")

        # Save the final segmentation
        final_seg_nifti = nib.Nifti1Image(final_segmentation, nib.load(v_image_path).affine)
        nib.save(final_seg_nifti, os.path.join(os.path.dirname(v_label_path), f"{v_folder_name}_local_weighting_segmentation_{similarity_metric}.nii.gz"))

        # Calculate and save the Dice Scores
        original_seg = nib.load(v_label_path).get_fdata()
        dice_scores = calculate_dice_scores(original_seg, final_segmentation)
        save_dice_scores(dice_scores, os.path.dirname(v_label_path), v_folder_name, similarity_metric)

def calculate_dice_scores(original_seg, final_segmentation):
    dice_scores = {}
    for tissue_label in [1, 2, 3]:  # 1: CSF, 2: GM, 3: WM
        dice_scores[tissue_label] = iu.dice_score_per_tissue(original_seg, final_segmentation, tissue_label)
    return dice_scores

def save_dice_scores(dice_scores, output_folder, v_folder_name, similarity_metric):
    csv_file_path = os.path.join(output_folder, f"{v_folder_name}_{similarity_metric}_local_weighting_dice_scores.csv")
    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(['Tissue Type', 'Dice Score'])
        for tissue_label in [1, 2, 3]:
            csv_writer.writerow([f"Tissue {tissue_label}", dice_scores[tissue_label]])

    print(f"Dice scores for {v_folder_name} with {similarity_metric} saved to {csv_file_path}")

def calculate_mutual_information(target, atlas):
    """
    Calculate the mutual information between two images.
    
    :param target: Target image array.
    :param atlas: Atlas image array.
    :return: Mutual information score.
    """
    # Flatten the image arrays
    target_flat = target.ravel()
    atlas_flat = atlas.ravel()

    # Calculate mutual information
    mi = mutual_info_score(target_flat, atlas_flat)
    return mi

def postprocessing_registered_images(data_handler):
    validation_data = data_handler.retrieve_data('validation', 0)
    for v_image_path, v_label_path, _ in validation_data:
        v_folder_name = os.path.basename(os.path.dirname(v_image_path))
        original_img_path = os.path.join(os.path.dirname(v_label_path), f"{v_folder_name}.nii.gz")
        original_seg = nib.load(original_img_path)
        # Paths for registered atlas images and transformed labels
        
        atlas_image_paths = []

        atlases_paths = [os.path.join(os.path.dirname(v_label_path), file) for file in os.listdir(os.path.dirname(v_label_path)) if '_PREPR_BF_NL_EQ.nii.gz' in file]
        
        atlases_read = [nib.load(path).get_fdata() for path in atlases_paths]
        output_folder = os.path.dirname(v_image_path)
        for file in atlases_read:
            adjusted = data_handler.rescale_to_255(file)
            output_image_path = os.path.join(output_folder, f"image_registered_{os.path.dirname(v_folder_name)}_PREPR_BF_NL_EQ.nii.gz")
            nib.save(nib.Nifti1Image(adjusted, original_seg.affine), output_image_path)


def calculate_normalized_cross_correlation(target, atlas):
    """
    Calculate the normalized cross-correlation between two images.
    
    :param target: Target image array.
    :param atlas: Atlas image array.
    :return: Normalized cross-correlation score.
    """
    target_flat = target.ravel() - np.mean(target)
    atlas_flat = atlas.ravel() - np.mean(atlas)
    correlation = np.sum(target_flat * atlas_flat)
    normalization = np.sqrt(np.sum(target_flat**2) * np.sum(atlas_flat**2))
    return correlation / normalization if normalization != 0 else 0