import SimpleITK as sitk
import itk
import os
import nibabel as nib
import numpy as np
import csv

def register_image(fixed_image_path, moving_image_path, fixed_mask_path, output_folder, parameter_files):
    fixed_image = itk.imread(fixed_image_path, itk.F)
    fixed_mask = itk.imread(fixed_mask_path, itk.UC)
    moving_image = itk.imread(moving_image_path, itk.F)

    parameter_object = itk.ParameterObject.New()
    for param_file in parameter_files:
        parameter_object.AddParameterFile(param_file)

    elastix_object = itk.ElastixRegistrationMethod.New(fixed_image, moving_image)
    elastix_object.SetFixedMask(fixed_mask)
    elastix_object.SetParameterObject(parameter_object)
    elastix_object.SetLogToConsole(False)
    elastix_object.SetOutputDirectory(output_folder)

    # Perform the registration
    elastix_object.UpdateLargestPossibleRegion()

    # Save the result of registration
    result_image = elastix_object.GetOutput()
    output_image_path = os.path.join(output_folder, f"image_registered_{os.path.basename(moving_image_path)}")
    itk.imwrite(result_image, output_image_path)


def apply_transformation(moving_label_path, output_folder, transform_param_files):
    moving_label = sitk.ReadImage(moving_label_path)

    transformix_transform = sitk.TransformixImageFilter()
    transformix_transform.SetMovingImage(moving_label)
    transformix_transform.SetOutputDirectory(output_folder)
    transformix_transform.SetTransformParameterMap(sitk.ReadParameterFile(transform_param_files[0]))
    transformix_transform.SetTransformParameter("FinalBSplineInterpolationOrder",'0')

    for param_file in transform_param_files[1:]:
        transformix_transform.AddTransformParameterMap(sitk.ReadParameterFile(param_file))
        transformix_transform.SetTransformParameter("FinalBSplineInterpolationOrder", '0')

    transformix_transform.Execute()

    result_label_transformix = transformix_transform.GetResultImage()
    output_label_path = os.path.join(output_folder, f"{os.path.basename(moving_label_path).replace('.nii.gz', '')}_transformed.nii.gz")
    sitk.WriteImage(result_label_transformix, output_label_path)

def find_transform_parameter_files(output_folder):
    #because we cannot iterate on result_parameter object so this is necessary to pass the different
    #transformparameter
    parameter_files = []
    i = 0
    while True:
        param_file = os.path.join(output_folder, f"TransformParameters.{i}.txt")
        if os.path.exists(param_file):
            parameter_files.append(param_file)
            i += 1
        else:
            break
    return parameter_files

def register_and_propagate_labels(data_handler, parameter_files):
    training_data = data_handler.retrieve_data('training', 0)
    validation_data = data_handler.retrieve_data('validation', 0)

    for t_image_path, t_label_path, _ in training_data:
        t_folder_name = os.path.basename(os.path.dirname(t_image_path))
        moving_image_path = os.path.join(os.path.dirname(t_image_path), f"{t_folder_name}_PREPR_BF_NL_EQ.nii.gz")
        
        # Find corresponding validation image and mask
        for v_image_path, _, v_mask_path in validation_data:
            v_folder_name = os.path.basename(os.path.dirname(v_image_path))
            fixed_image_path = os.path.join(os.path.dirname(v_image_path), f"{v_folder_name}_PREPR_BF_NL_EQ.nii.gz")
            fixed_mask_path = v_mask_path

            # Perform registration
            output_folder = os.path.dirname(v_image_path)
            register_image(fixed_image_path, moving_image_path, fixed_mask_path, output_folder, parameter_files)

            # Apply label propagation
            moving_label_path = t_label_path
            transform_param_files = find_transform_parameter_files(output_folder)
            apply_transformation(moving_label_path, output_folder, transform_param_files)

def dice_score_per_tissue(true_labels, pred_labels, tissue_value):
    true_labels = np.squeeze(true_labels)  # Remove singleton dimensions if any
    pred_labels = np.squeeze(pred_labels)  # Remove singleton dimensions if any

    true_labels_tissue = (true_labels == tissue_value).astype(int)
    pred_labels_tissue = (pred_labels == tissue_value).astype(int)
    intersection = np.sum(true_labels_tissue * pred_labels_tissue)

    total = np.sum(true_labels_tissue) + np.sum(pred_labels_tissue)
    return 2. * intersection / total if total > 0 else 0


def calculate_segmentation_scores(data_handler):
    """
    Calculate Dice Score Coefficients for transformed segmentation images against original validation segmentations,
    and save the results in a CSV file in each validation subfolder.
    """
    validation_data = data_handler.retrieve_data('validation', 0)

    for v_image_path, v_label_path, _ in validation_data:
        v_folder_name = os.path.basename(os.path.dirname(v_image_path))
        original_seg_path = os.path.join(os.path.dirname(v_label_path), f"{v_folder_name}_seg.nii.gz")
        original_seg = nib.load(original_seg_path).get_fdata()

        transformed_segs = [file for file in os.listdir(os.path.dirname(v_label_path)) if '_seg_transformed.nii.gz' in file]

        # Prepare CSV file
        csv_file_path = os.path.join(os.path.dirname(v_label_path), f"{v_folder_name}_dice_scores.csv")
        with open(csv_file_path, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            # Write header
            csv_writer.writerow(['Transformed Segmentation', 'CSF Dice Score', 'GM Dice Score', 'WM Dice Score'])

            for transformed_seg_file in transformed_segs:
                transformed_seg_path = os.path.join(os.path.dirname(v_label_path), transformed_seg_file)
                transformed_seg = nib.load(transformed_seg_path).get_fdata()

                dice_scores = {}
                for tissue_label in [1, 2, 3]:  # 1: CSF, 2: GM, 3: WM
                    dice_scores[tissue_label] = dice_score_per_tissue(original_seg, transformed_seg, tissue_label)

                # Write data to CSV
                csv_writer.writerow([
                    transformed_seg_file, 
                    dice_scores[1],  # CSF
                    dice_scores[2],  # GM
                    dice_scores[3]   # WM
                ])

        print(f"Dice scores for {v_folder_name} saved to {csv_file_path}")
