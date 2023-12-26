import SimpleITK as sitk
import itk
import os

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
    transformix_transform.SetTransformParameter("FinalBSplineInterpolationOrder", 0)

    for param_file in transform_param_files[1:]:
        transformix_transform.AddTransformParameterMap(sitk.ReadParameterFile(param_file))
        transformix_transform.SetTransformParameter("FinalBSplineInterpolationOrder", 0)

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

        fixed_image_path = None
        fixed_mask_path = None
        
        # Find corresponding validation image and mask
        for v_image_path, _, v_mask_path in validation_data:
            fixed_image_path = v_image_path
            fixed_mask_path = v_mask_path

            # Perform registration
            output_folder = os.path.dirname(v_image_path)
            register_image(fixed_image_path, moving_image_path, fixed_mask_path, output_folder, parameter_files)

            # Apply label propagation
            moving_label_path = t_label_path
            transform_param_files = find_transform_parameter_files(output_folder)
            apply_transformation(moving_label_path, output_folder, transform_param_files)
