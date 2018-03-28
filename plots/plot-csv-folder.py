import os, sys
import matplotlib.style as style
import matplotlib.pyplot as plt
import pandas as pd

from scipy.interpolate import spline
from os.path import basename

# example: python plot-csv.py
# source: https://matplotlib.org/examples/pylab_examples/plotfile_demo.html

# styles
#style.use('fivethirtyeight')
#style.use('ggplot')
#style.use('seaborn-darkgrid')
#style.use('seaborn-ticks')
#style.use('seaborn-notebook')
#style.use('Solarize_Light2')
#style.use('seaborn-poster')
#style.use('seaborn-deep')
#style.use('seaborn-bright')

# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = 'Ubuntu'
# plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 6
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 6
plt.rcParams['ytick.labelsize'] = 6
#plt.rcParams['legend.fontsize'] = 9
# plt.rcParams['legend.fontsize'] = 9
plt.rcParams["figure.figsize"] = (4,2.5)

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)
fig.tight_layout()

ax1.set_ylabel('Accuracy')
ax1.axvline(x=10000, color='#a9a9a9', linewidth=0.5)
ax1.axvline(x=30000, color='#a9a9a9', linewidth=0.5)
ax1.axvline(x=49000, color='#a9a9a9', linewidth=0.5)

ax2.set_ylabel('Weight loss')
ax2.axvline(x=10000, color='#a9a9a9', linewidth=0.5)
ax2.axvline(x=30000, color='#a9a9a9', linewidth=0.5)
ax2.axvline(x=49000, color='#a9a9a9', linewidth=0.5)

ax3.set_ylabel('Cross entropy')
ax3.set_xlabel('Epoch')
ax3.axvline(x=10000, color='#a9a9a9', linewidth=0.5)
ax3.axvline(x=30000, color='#a9a9a9', linewidth=0.5)
ax3.axvline(x=49000, color='#a9a9a9', linewidth=0.5)

plt.subplots_adjust(hspace=0.0)

colors = ['#5eeccc', '#5ebfec', '#ec5e6a', '#b0ec5e', '#5f5eec', '#e05eec']
color_index = 0

files = os.listdir(sys.argv[1])

for file in files:	
	if file.endswith(".csv"):
		absolute_path = sys.argv[1] + "/" + file
		path, filename = os.path.split(absolute_path)
		name, file_extension = os.path.splitext(filename)
		dataset = name.replace("evaluation-", "")	
		output =  sys.argv[1] + "/" + name + "-output.svg"
		output =  sys.argv[1] + "/" + name + "-output.pdf"

		# print(file)
		# print(absolute_path)
		# print(path)
		# print(filename)
		# print(name)
		# print(dataset)
		# print(output)
	
		csvfile = pd.read_csv(absolute_path)

		ep = csvfile['Epoch']
		acc = csvfile['Accuracy']
		wei = csvfile['Weight loss']
		xent = csvfile['Cross entropy']
		
		ax1.plot(ep, acc, '#00aacc', label=dataset, color=colors[color_index], markevery=100)		
		# ax1.annotate(acc[10], xy=(2., -1), xycoords='data', xytext=(-100, 60), textcoords='offset points', size=20)
		# ax1.spines['bottom'].set_color('black')
		# ax1.spines['bottom'].set_linewidth(2)
		
		ax2.plot(ep, wei, '#00aacc', label=dataset, color=colors[color_index], markevery=100)	
		# ax2.spines['bottom'].set_color('black')
		# ax2.spines['bottom'].set_linewidth(2)
		
		ax3.plot(ep, xent, '#00aacc', label=dataset, color=colors[color_index], markevery=100)
		
		color_index += 1
		
#leg = plt.gca().get_legend()

#plt.legend(bbox_to_anchor=(0.7, 40000), loc=7, ncol=12, mode="expand", borderaxespad=0.)
# plt.legend(loc=7, shadow=True, fancybox=True)
plt.legend()
#plt.show()
plt.savefig(output, bbox_inches='tight', dpi=300)
plt.savefig(output2, bbox_inches='tight', dpi=300)