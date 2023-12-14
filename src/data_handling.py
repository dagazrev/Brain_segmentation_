import os
import numpy as np
import nibabel as nib

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
