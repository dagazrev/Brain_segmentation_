import data_handling as dh
import initial_utils as iu
import advanced_utils as au


obj = dh.data_handling("images")

#################################
########Preprocessing############
#################################

#The order is done by the order in the list, can be one, 2 or all operations
#obj.process_images(['BF', 'NL', 'EQ'])

parameter_files = ['src\Par0033similarity.txt', 'src\Par0033bspline.txt']

#iu.register_and_propagate_labels(obj, parameter_files)

au.postprocessing_registered_images(obj)

#iu.calculate_segmentation_scores(obj)

#iu.majority_voting(obj)
#iu.staple_fusion_and_dice(obj)

#####Advanced Methods
#au.local_weighting_segmentation(obj)