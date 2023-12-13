import os

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