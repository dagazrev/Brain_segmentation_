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


class data_handling:#remaing from dataloader since seeing the pytorch dataloaderxD
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.types = ['test', 'training', 'validation']

    def find_files(self):
        data = {}
        for t in self.types:
            type_folder = os.path.join(self.data_folder, f"images-{t}")
            data[t] = []
            for folder_name in sorted(os.listdir(type_folder)):
                if not folder_name.startswith('IBSR'):
                    continue  # skip non-IBSR folders
                full_folder_path = os.path.join(type_folder, folder_name)
                image_file = os.path.join(full_folder_path, f"{folder_name}.nii.gz")
                label_file = os.path.join(full_folder_path, f"{folder_name}_seg.nii.gz")
                mask_file = os.path.join(full_folder_path, f"{folder_name}_mask.nii.gz")
                data[t].append((image_file, label_file, mask_file))
        return data

    def format_number(self, number):#I think it was easier this way than renaming the whole folders
        if number < 10:
            return f"0{number}"
        return str(number)

    def retrieve_data(self, folder_type, numbers):
        if numbers == 0:  # Retrieve all
            return self.find_files()[folder_type]

        result = []
        for number in numbers:
            formatted_number = self.format_number(number)
            target_folder = os.path.join(self.data_folder, f"images-{folder_type}", f"IBSR_{formatted_number}")
            if not os.path.exists(target_folder):
                continue  # Or handle this case as you prefer
            image_filename = os.path.join(target_folder, f"IBSR_{formatted_number}.nii.gz")
            label_filename = os.path.join(target_folder, f"IBSR_{formatted_number}_seg.nii.gz")
            mask_filename = os.path.join(target_folder, f"IBSR_{formatted_number}_mask.nii.gz")
            result.append((image_filename, label_filename, mask_filename))
        return result
    
    def compare_histograms_top_three(self):
        training_data = self.retrieve_data('training', 0)
        validation_data = self.retrieve_data('validation', 0)

        for v_image, _, _ in validation_data:
            print(f"Comparing for {v_image}:")
            v_img_nifti = nib.load(v_image)
            v_img_data = v_img_nifti.get_fdata()
            v_hist, _ = np.histogram(v_img_data.flatten(), bins=256, range=[0, 256])

            comparisons = []
            for t_image, _, _ in training_data:
                t_img_nifti = nib.load(t_image)
                t_img_data = t_img_nifti.get_fdata()
                t_hist, _ = np.histogram(t_img_data.flatten(), bins=256, range=[0, 256])

                # Calculate the correlation coefficient
                coef = np.corrcoef(v_hist, t_hist)[0, 1]
                comparisons.append((t_image, coef))

            # Sort based on the coefficient
            comparisons.sort(key=lambda x: x[1], reverse=True)

            # Print top 3 matches
            for match in comparisons[:3]:
                print(f"Training Image: {match[0]}, Coefficient: {match[1]}")

    def compare_histograms(self):
        training_data = self.retrieve_data('training', 0)
        validation_data = self.retrieve_data('validation', 0)

        top_pairs = []

        for v_image, _, _ in validation_data:
            top_similarity = 0
            top_training_image = None

            v_img_nifti = nib.load(v_image)
            v_img_data = v_img_nifti.get_fdata()
            v_hist, _ = np.histogram(v_img_data.flatten(), bins=256, range=[0, 256])

            for t_image, _, _ in training_data:
                t_img_nifti = nib.load(t_image)
                t_img_data = t_img_nifti.get_fdata()
                t_hist, _ = np.histogram(t_img_data.flatten(), bins=256, range=[0, 256])

                coef = np.corrcoef(v_hist, t_hist)[0, 1]
                if coef > top_similarity:
                    top_similarity = coef
                    top_training_image = t_image

            print(f"Validation Image: {v_image}, Top Training Image: {top_training_image}, Similarity Coefficient: {top_similarity}")
            top_pairs.append((v_image, top_training_image))

        return top_pairs
    
    def apply_histogram_normalization_and_plot(self):
        training_data = self.retrieve_data('training', 0)

        for image_path, _, _ in training_data:
            original_img_nifti = nib.load(image_path)
            original_img_data = original_img_nifti.get_fdata()

            # Apply histogram normalization
            normalized_img_data = self.histogram_normalization(original_img_data)

            # Plot slices
            self.plot_slices(original_img_data, normalized_img_data, image_path)

    def histogram_normalization(self, img_data):
        # Flatten the image data to 1D for equalization
        img_data_flat = img_data.flatten()
        # Apply histogram equalization
        img_data_eq = exposure.equalize_hist(img_data_flat)

        # Reshape back to original shape
        normalized_img_data = img_data_eq.reshape(img_data.shape)
        return normalized_img_data
    
    def plot_slices(self, original_img_data, normalized_img_data, image_path):
        # Choose a slice index for each view
        axial_slice = original_img_data.shape[2] // 2
        coronal_slice = original_img_data.shape[1] // 2
        sagittal_slice = original_img_data.shape[0] // 2

        axial_slicee = normalized_img_data.shape[2] // 2
        coronal_slicee = normalized_img_data.shape[1] // 2
        sagittal_slicee = normalized_img_data.shape[0] // 2

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Slices of {image_path}')

        # Original Image Slices
        axes[0, 0].imshow(original_img_data[:, :, axial_slice], cmap='gray')
        axes[0, 0].set_title('Original Axial')
        axes[0, 1].imshow(original_img_data[:, coronal_slice, :], cmap='gray')
        axes[0, 1].set_title('Original Coronal')
        axes[0, 2].imshow(original_img_data[sagittal_slice, :, :], cmap='gray')
        axes[0, 2].set_title('Original Sagittal')

        # Normalized Image Slices
        axes[1, 0].imshow(normalized_img_data[:, :, axial_slicee], cmap='gray')
        axes[1, 0].set_title('Normalized Axial')
        axes[1, 1].imshow(normalized_img_data[:, coronal_slicee, :], cmap='gray')
        axes[1, 1].set_title('Normalized Coronal')
        axes[1, 2].imshow(normalized_img_data[sagittal_slicee, :, :], cmap='gray')
        axes[1, 2].set_title('Normalized Sagittal')

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        plot_filename = f"{base_name}_og_vs_pr.png"
        plt.savefig(os.path.join(self.data_folder, plot_filename))
        plt.close()

        #plt.show()

    def apply_denoising_and_histogram_normalization_and_plot(self):
        training_data = self.retrieve_data('training', 0)

        for image_path, _, _ in training_data:
            original_img_nifti = nib.load(image_path)
            original_img_data = original_img_nifti.get_fdata()

            # Apply non-local means denoising
            denoised_img_data = self.non_local_means_denoising(original_img_data)

            # Apply histogram normalization
            normalized_img_data = self.histogram_normalization(denoised_img_data)

            # Plot slices
            self.plot_slices(original_img_data, normalized_img_data, image_path)

    def non_local_means_denoising(self, img_data):
        # Estimate the noise standard deviation from the image
        sigma_est = np.mean(estimate_sigma(img_data, channel_axis=-1))

        patch_kw = dict(patch_size=5,      # 5x5 patches
                patch_distance=6,  # 13x13 search area
                channel_axis=-1)

        # Apply non-local means denoising
        denoised_img = denoise_nl_means(img_data, h=1.15 * sigma_est, fast_mode=True, **patch_kw)
        return denoised_img
    
    def apply_bias_field_correction(self, img_data):
        img_data_float = img_data.astype(np.float64)


        # Estimate the bias field using a Gaussian filter
        bias_field = gaussian_filter(img_data_float, sigma=5)

        # Correct the image
        corrected_img_data = np.divide(img_data_float, bias_field, out=np.zeros_like(img_data_float), where=bias_field!=0)
        # Rescale the corrected data back to 0-255 range if necessary
        corrected_img_data_rescaled = np.clip(corrected_img_data, 0, 255)
        # Cast back to uint8
        corrected_img_data_uint8 = corrected_img_data_rescaled.astype(np.uint8)


        return corrected_img_data
    
    def process_images(self, operation_sequence):
        training_data = self.retrieve_data('test', 0)

        for image_path, _, _ in training_data:
            original_img_nifti = nib.load(image_path)
            img_data = original_img_nifti.get_fdata()

            # Create a mask from the original image
            mask = np.squeeze(img_data > 0)

            # Rescale to 0-255
            img_data = self.rescale_to_255(img_data)

            for operation in operation_sequence:
                if operation == 'BF':
                    img_data = self.apply_bias_field_correction_(img_data, image_path, original_img_nifti.affine)
                elif operation == 'EQ':
                    img_data = self.apply_clahe(img_data)
                elif operation == 'NL':
                    img_data = self.non_local_means_denoising(img_data)

            # Apply the mask to retain the original background
            img_data = img_data * mask
            # Construct new filename
            base_name = os.path.basename(image_path)
            name, ext = os.path.splitext(base_name)
            if ext == '.gz':
                name, _ = os.path.splitext(name)
                ext = '.nii.gz'
            new_name = f"{name}_PREPR_{'_'.join(operation_sequence)}{ext}"

            # Save the processed image
            processed_nifti = nib.Nifti1Image(img_data, original_img_nifti.affine)
            nib.save(processed_nifti, os.path.join(self.data_folder, new_name))

            # self.plot_slices(original_img_nifti.get_fdata(), img_data, image_path)






    def rescale_to_255(self, img_data):
        img_min, img_max = img_data.min(), img_data.max()
        img_data_rescaled = (img_data - img_min) / (img_max - img_min) * 255
        return img_data_rescaled.astype(np.uint8)


    def contrast_stretching(self, img_data):
        # Rescale the pixel values to span the full range
        p2, p98 = np.percentile(img_data, (2, 98))
        img_rescaled = exposure.rescale_intensity(img_data, in_range=(p2, p98))
        return img_rescaled
    
    def apply_clahe(self, img_data):
        # Rescale data to 0-1
        img_data_rescaled = img_data / 255.0

        # Apply CLAHE to each slice
        clahe = exposure.equalize_adapthist
        img_clahe = np.zeros_like(img_data_rescaled)
        for i in range(img_data_rescaled.shape[2]): 
            img_clahe[:, :, i] = clahe(img_data_rescaled[:, :, i])

        # Rescale back to 0-255
        img_clahe_rescaled = (img_clahe * 255).astype(np.uint8)

        return img_clahe_rescaled
    
    def apply_bias_field_correction_(self, img_data, image_path, affin):
        img_data_float = img_data.astype(np.float64)

        # Estimate the bias field using a Gaussian filter
        bias_field = gaussian_filter(img_data_float, sigma=26)
        
        # Normalize the bias field
        bias_field_normalized = (bias_field - bias_field.min()) / (bias_field.max() - bias_field.min()) * (img_data_float.max() - img_data_float.min())

        # Correct the image
        corrected_img_data = np.divide(img_data_float, bias_field_normalized, out=np.zeros_like(img_data_float), where=bias_field_normalized!=0)
        corrected_img_data_rescaled = np.clip(corrected_img_data, 0, img_data.max())
        corrected_img_data_cast = corrected_img_data_rescaled.astype(img_data.dtype)

        # Save the bias field as a NIfTI file
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        if '.gz' in base_name:
            base_name = os.path.splitext(base_name)[0]
        bias_field_filename = f"{base_name}_BIAS_FIELD.nii.gz"
        bias_field_nifti = nib.Nifti1Image(bias_field_normalized, affin)
        nib.save(bias_field_nifti, os.path.join(self.data_folder, bias_field_filename))

        return corrected_img_data_rescaled


    def plot_bias_field(self, original_img, corrected_img, bias_field, image_path):
        slice_idx = original_img.shape[2] // 2  # Example for axial slice

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(original_img[:, :, slice_idx], cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(corrected_img[:, :, slice_idx], cmap='gray')
        plt.title('Corrected Image')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(bias_field[:, :, slice_idx], cmap='gray')
        plt.title('Bias Field')
        plt.axis('off')

        plt.suptitle(f'Bias Field Correction: {os.path.basename(image_path)}')

       
        #plt.show()