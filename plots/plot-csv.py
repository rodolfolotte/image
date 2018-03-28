import os, sys
import matplotlib.style as style
import matplotlib.pyplot as plt
import pandas as pd

from os.path import basename

# example: python plot-csv.py
# source: https://matplotlib.org/examples/pylab_examples/plotfile_demo.html

# styles
#style.use('fivethirtyeight')
#style.use('ggplot')
#style.use('seaborn-darkgrid')
style.use('seaborn-ticks')
#style.use('seaborn-notebook')
#style.use('Solarize_Light2')
#style.use('seaborn-poster')
#style.use('seaborn-deep')

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9
plt.rcParams["figure.figsize"] = (4,3)

for file in sys.argv[1:]:	
	absolute_path = os.path.abspath(file)
	path, filename = os.path.split(absolute_path)
	name, file_extension = os.path.splitext(filename)
	output = path + "/" + name + "-output.png"

	# print(absolute_path)
	# print(path)
	# print(filename)
	# print(name)
	# print(output)
	
	csvfile = pd.read_csv(absolute_path)

	ep = csvfile.Epoch
	acc = csvfile['Accuracy']
	wei = csvfile['Weight loss']
	xent = csvfile['Cross entropy']
	
	fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)
	fig.tight_layout()
	
	ax1.plot(ep, acc, '#00aacc')	
	ax1.set_ylabel('Accuracy')
	ax1.axvline(x=10000, color='#ff0000', linewidth=0.5)
	ax1.axvline(x=30000, color='#ff0000', linewidth=0.5)
	ax1.axvline(x=49000, color='#ff0000', linewidth=0.5)
	ax1.annotate(acc[10], xy=(2., -1), xycoords='data', xytext=(-100, 60), textcoords='offset points', size=20)
	# ax1.spines['bottom'].set_color('black')
	# ax1.spines['bottom'].set_linewidth(2)
	
	ax2.plot(ep, wei, '#00aacc')
	ax2.set_ylabel('Weight loss')
	ax2.axvline(x=10000, color='#ff0000', linewidth=0.5)
	ax2.axvline(x=30000, color='#ff0000', linewidth=0.5)
	ax2.axvline(x=49000, color='#ff0000', linewidth=0.5)
	# ax2.spines['bottom'].set_color('black')
	# ax2.spines['bottom'].set_linewidth(2)
	
	ax3.plot(ep, xent, '#00aacc')
	ax3.set_ylabel('Cross entropy')
	ax3.set_xlabel('Epoch')
	ax3.axvline(x=10000, color='#ff0000', linewidth=0.5)
	ax3.axvline(x=30000, color='#ff0000', linewidth=0.5)
	ax3.axvline(x=49000, color='#ff0000', linewidth=0.5)

	plt.subplots_adjust(hspace=0.)	   
	
	#plt.show()
	plt.savefig(output, bbox_inches='tight', dpi=300)