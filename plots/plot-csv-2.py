import os, sys
import matplotlib.style as style
import matplotlib.pyplot as plt
import pandas as pd

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
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9

for file in sys.argv[1:]:	
	absolute_path = os.path.abspath(file)
	path, filename = os.path.split(absolute_path)	
	name, file_extension = os.path.splitext(file)	
	output = path + "/" + name + "-output.png"
	
	csvfile = pd.read_csv(absolute_path)

	ep = csvfile.Epoch
	acc = csvfile['Accuracy']
	wei = csvfile['Weight loss']
	xent = csvfile['Cross entropy']

	fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)
	fig.tight_layout()
	
	ax1.plot(ep, acc, '#0044cc')	
	ax1.set_ylabel('Accuracy')

	ax2.plot(ep, wei, '#0044cc')
	ax2.set_ylabel('Weight loss')

	ax3.plot(ep, xent, '#0044cc')
	ax3.set_ylabel('Cross entropy')
	ax3.set_xlabel('Epoch')

	plt.subplots_adjust(hspace = .05)	   


	#plt.show()
	plt.savefig(output, bbox_inches='tight', dpi=300)