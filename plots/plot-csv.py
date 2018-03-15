import os, sys
import matplotlib.style as style
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook

from os.path import basename

# example: python plot-csv.py
# source: https://matplotlib.org/examples/pylab_examples/plotfile_demo.html

# styles
#style.use('fivethirtyeight')
style.use('ggplot')
#style.use('seaborn-darkgrid')

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 9

for file in sys.argv[1:]:	
	absolute_path = os.path.abspath(file)
	path, filename = os.path.split(absolute_path)	
	name, file_extension = os.path.splitext(file)	
	output = path + "/" + name + "-output.png"

	csvfile = cbook.get_sample_data(absolute_path, asfileobj=False)

	# plt.plotfile(csvfile, cols=(0,1,2,5), subplots=False)
	# plt.plotfile(csvfile, cols=(0,5), newfig=False)
	plt.plotfile(csvfile, (0,1,2,5), delimiter=',')

	# fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)

	# plt.show()
	plt.savefig(output, bbox_inches='tight', dpi=300)