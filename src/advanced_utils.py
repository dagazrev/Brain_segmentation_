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


def local_weighting_segmentation(data_handler):
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

        for i in range(1, 21):  # Adjust the range as needed
            atlas_path = os.path.join(os.path.dirname(v_image_path), f"image_registered_IBSR_{i:02}_PREPR_BF_NL_EQ.nii.gz")
            label_path = os.path.join(os.path.dirname(v_label_path), f"IBSR_{i:02}_seg_transformed.nii.gz")

        # Check if both the atlas image and label image exist
        if os.path.exists(atlas_path) and os.path.exists(label_path):
            atlas_image_paths.append(atlas_path)
            transformed_label_paths.append(label_path)

        
        target_image = nib.load(v_image_path).get_fdata()
        final_segmentation = np.squeeze(np.zeros_like(target_image))

        for atlas_path, label_path in zip(atlases_paths, labels_paths):
            atlas_image = np.squeeze(nib.load(atlas_path).get_fdata())
            label_image = np.squeeze(nib.load(label_path).get_fdata())

            # Calculate similarity weight between atlas and target image
            similarity_weight = calculate_mutual_information(target_image, atlas_image)
            print(similarity_weight.shape)
            # Check if similarity_weight is a scalar
            if np.isscalar(similarity_weight):
                # Scalar multiplication should work fine
                weighted_label_image = similarity_weight * label_image
            else:
                # If similarity_weight is an array, ensure it has the correct shape
                similarity_weight = np.squeeze(similarity_weight)
                if similarity_weight.shape != label_image.shape:
                    raise ValueError("Shape of similarity_weight does not match label_image")
                weighted_label_image = similarity_weight * label_image
            print(final_segmentation.shape)
            print(weighted_label_image.shape)
            # Add the weighted label image to final segmentation
            final_segmentation += weighted_label_image

        # After the loop, also ensure final_segmentation does not have singleton dimensions
        final_segmentation = np.squeeze(final_segmentation)

        # Normalize and finalize the segmentation
        final_segmentation = np.round(final_segmentation / len(atlas_image_paths)).astype(int)

        # Save the final segmentation
        final_seg_nifti = nib.Nifti1Image(final_segmentation, nib.load(v_image_path).affine)
        nib.save(final_seg_nifti, os.path.join(os.path.dirname(v_label_path), f"{v_folder_name}_local_weighting_segmentation.nii.gz"))

        # Calculate and save the Dice Scores
        original_seg = nib.load(v_label_path).get_fdata()
        dice_scores = calculate_dice_scores(original_seg, final_segmentation)
        save_dice_scores(dice_scores, os.path.dirname(v_label_path), v_folder_name)

def calculate_dice_scores(original_seg, final_segmentation):
    dice_scores = {}
    for tissue_label in [1, 2, 3]:  # 1: CSF, 2: GM, 3: WM
        dice_scores[tissue_label] = iu.dice_score_per_tissue(original_seg, final_segmentation, tissue_label)
    return dice_scores

def save_dice_scores(dice_scores, output_folder, v_folder_name):
    csv_file_path = os.path.join(output_folder, f"{v_folder_name}_local_weighting_dice_scores.csv")
    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(['Tissue Type', 'Dice Score'])
        for tissue_label in [1, 2, 3]:
            csv_writer.writerow([f"Tissue {tissue_label}", dice_scores[tissue_label]])

    print(f"Dice scores for {v_folder_name} saved to {csv_file_path}")

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

