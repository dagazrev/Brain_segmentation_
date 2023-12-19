import data_handling as dh
import initial_utils as ut
obj = dh.data_handling("images")
print(obj.retrieve_data('training', [9]))

paths = obj.compare_histograms()

parameter_files = ['Par0033similarity.txt', 'Par0033bspline.txt']

ut.full_reg(obj,parameter_files)