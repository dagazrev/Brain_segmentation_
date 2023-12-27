import data_handling as dh
import initial_utils as iu
obj = dh.data_handling("images")

#obj.process_images(['BF', 'NL', 'EQ'])

parameter_files = ['src\Par0033similarity.txt', 'src\Par0033bspline.txt']

#iu.register_and_propagate_labels(obj, parameter_files)

#iu.calculate_segmentation_scores(obj)

iu.majority_voting(obj)
