import os
import math
import numpy as np

def renderTable(targetCaps, outputCaps, ncoln):
	assert len(targetCaps) == len(outputCaps), 'Target and output Caps dimension mismatch.'
	htmlStr = '<table border="1">\n'
	for row in range(math.ceil(len(targetCaps)/ncoln)):
		htmlStr +=  '<tr>'
		for coln in range(ncoln):
			htmlStr += '<td>'
			ID = row*ncoln + coln
			if ID < len(targetCaps):
				htmlStr += '<br>'
				htmlStr += targetCaps[ID]
				htmlStr += '<br>'
				htmlStr += outputCaps[ID]
				htmlStr += '<br>'
			htmlStr += '</td>'
		htmlStr += '</tr>\n'
	htmlStr += '</table>\n'
	return htmlStr


def writeHTML(targetCaps, outputCaps, epoch, split, opt):
	print('==> Writing visualize html ...')
	htmlFile = 'index.html'
	rootDir = os.path.join(opt.www, split + '_' + str(epoch))
	if not os.path.exists(rootDir):
		os.makedirs(rootDir)
	f = open(os.path.join(rootDir, htmlFile), 'w')
	f.write('<html>\n<body>\n')
	table = renderTable(targetCaps, outputCaps, opt.visWidth)
	f.write(table)
	f.write('</body>\n</html>\n')
	f.close()



