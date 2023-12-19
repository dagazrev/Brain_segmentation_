import SimpleITK as sitk
import itk
import os
from src import initial_utils as ut
import time
import glob
import pandas as pd

# def perform_registration_and_analysis(data_loader, output_folder, parameter_files):
#     similar_images = data_loader.compare_histograms()  # Assume this returns a list of tuples (validation_image, training_image)

#     for val_image, train_image in similar_images:
#         fixed_image_path = val_image[0]  # Validation image
#         moving_image_path = train_image[0]  # Training image
#         fixed_mask_path = val_image[2]  # Validation mask
#         ground_truth_path = val_image[1]  # Validation ground truth (segmentation)

#         # Perform the registration
#         register_image(fixed_image_path, moving_image_path, fixed_mask_path, output_folder, parameter_files)

#         # Find the transform parameter files
#         transform_param_files = find_transform_parameter_files(output_folder)

#         # Propagate labels
#         apply_transformation(ground_truth_path, output_folder, transform_param_files)

#         # Calculate DSC for each tissue type
#         calculate_dice_scores(ground_truth_path, transformed_label_path)

def calculate_dice_scores(ground_truth_path, transformed_label_path):
    # Load ground truth and transformed label images
    ground_truth = sitk.ReadImage(ground_truth_path)
    transformed_label = sitk.ReadImage(transformed_label_path)

    # Calculate DSC for each tissue type
    for tissue_type in [1, 2, 3]:  # CSF, Gray Matter, White Matter
        dice_score = calculate_dice_for_tissue(ground_truth, transformed_label, tissue_type)
        print(f"Dice Score for tissue type {tissue_type}: {dice_score}")

def calculate_dice_for_tissue(ground_truth, transformed_label, tissue_type):
    # Extract specific tissue type from both images and calculate DSC
    # Implementation of DSC calculation goes here
    pass

def register_image(fixed_image_path, moving_image_path, output_folder, parameter_files, labels):
    name = os.path.basename(moving_image_path).replace('.nii.gz', '_reg.nii.gz')
    fixed_image = itk.imread(fixed_image_path, itk.F)
    moving_image = itk.imread(moving_image_path,itk.F)

    parameter_object = itk.ParameterObject.New()
    for param_file in parameter_files:
        parameter_object.AddParameterFile(param_file)

    elastix_object = itk.ElastixRegistrationMethod.New(fixed_image, moving_image)
    elastix_object.SetParameterObject(parameter_object)
    elastix_object.SetLogToConsole(False)
    elastix_object.SetOutputDirectory(output_folder)

    # Perform the registration
    elastix_object.UpdateLargestPossibleRegion()
    print("\t\t Registro completado.", '.'*100)
    # Save the result of registration
    result_image = elastix_object.GetOutput()
    itk.imwrite(result_image, os.path.join(output_folder,name))

    # Save the transformation parameter files
    result_transform_parameters = elastix_object.GetTransformParameterObject()

    transform_param_files = find_transform_parameter_files(output_folder)
    print(transform_param_files)
    print("\t\t Transformando labels.", '.'*100)
    apply_transformation(labels, output_folder, transform_param_files)


def apply_transformation(moving_label_path, output_folder, transform_param_files):
    # Read the moving label image and doing everything with SITK from this point 'cause I don't fucking understand how to load sucessive transfromparameters with ITK
    moving_label = sitk.ReadImage(moving_label_path)

    transformix_transform = sitk.TransformixImageFilter()
    transformix_transform.SetMovingImage(moving_label)
    transformix_transform.SetOutputDirectory(output_folder)
    #in https://github.com/SuperElastix/SimpleElastix/issues/341 says you need to Set the first parameter file and add the rest
    param = sitk.ReadParameterFile(transform_param_files[0])
    param["FinalBSplineInterpolationOrder"] = 0
    transformix_transform.SetTransformParameterMap()

    # Load and add the subsequent parameter maps
    for param_file in transform_param_files[1:]:
        param = sitk.ReadParameterFile(param_file)
        param["FinalBSplineInterpolationOrder"] = 0
        transformix_transform.AddTransformParameterMap(param)

    transformix_transform.Execute()
    print("\t\t Transformacion de labels completado.", '.'*100)
    result_label_transformix = transformix_transform.GetResultImage()
    label_name = os.path.basename(moving_label_path).replace('.nii.gz', '_reg.nii.gz')
    sitk.WriteImage(result_label_transformix, os.path.join(output_folder, label_name))

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

def full_reg(dataloader,parameter_files):
    for i in dataloader.retrieve_data('validation', 0):
        print("Registrando train a: ", os.path.basename(i[0], '.'*100))
        times_list = []
        for j in dataloader.retrieve_data('train', 0):
            start_time = time.time()
            print("\t Moviendo: ", os.path.basename(j[0]),'.'*100)       
            outdir= os.path.join(os.path.dirname(i[0]),os.path.basename(j[0]).replace('.nii.gz', '_reg'))
            os.makedirs(outdir)
            register_image(i[0], j[0], outdir,parameter_files, j[1])
            end_time = time.time()
            # Calcular el tiempo transcurrido y agregarlo a la lista
            elapsed_time = end_time - start_time
            times_list.append(elapsed_time)
        mov_labels = glob.glob(os.path.dirname(i[0] + "/*/*_seg_reg.nii.gz"))
        for label in mov_labels:
            scores = calculate_dice_scores(i[1], label)
        df = pd.DataFrame({'time': times_list,
                           'Dice': scores})
        df.to_csv(os.path.join(os.path.dirname(i[0]), 'eval.csv'))
