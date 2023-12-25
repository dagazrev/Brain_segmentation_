import data_handling as dh

obj = dh.data_handling("images")
#print(obj.retrieve_data('training', [9]))

#paths = obj.compare_histograms()
obj.process_images(['BF', 'NL', 'EQ'])

parameter_files = ['Par0033similarity.txt', 'Par0033bspline.txt']