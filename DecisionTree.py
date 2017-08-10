import sys, csv, math
from PIL import Image, ImageDraw



# The results column in the data.
ATTRITION_COLUMN_INDEX = 1
EMPLOYEE_ID_COLUMN_INDEX = 8


## A node in the decision tree.
class DecisionTreeNode:
	## Constructor
	def __init__(self, columnIndex = -1, value = None, results = None, leftChild = None, rightChild = None):
		self.columnIndex = columnIndex
		self.value = value;
		self.results = results;
		self.leftChild = leftChild
		self.rightChild = rightChild


## Recursively constructs the decision tree.
# @param data	This is a list of rows containing attributes for a particular employee from the .csv file
# @return		TO BE POPULATED
def buildTree(data):
	if len(data) == 0:
		return DecisionTreeNode()
	else:
		# Set initial values.
		bestInformationGain = 0.0
		bestCriteria = None
		bestSets = None
		currentScore = entropy(data)

		# Find the column that best divides the set.
		for columnIndex in range(0, len(data[0])):
			if columnIndex != ATTRITION_COLUMN_INDEX:
				# Partition the data by this column.
				columnValues = []
				for row in data:
					columnValues.append(row[columnIndex])

				for value in columnValues:
					# Divide the set.
					(set1, set2) = divideSet(data, columnIndex, value)

					# Compute the information gain.
					ratio = float(len(set1)) / float(len(data))
					informationGain = currentScore - ratio * entropy(set1) - (1.0 - ratio) * entropy(set2)
					if informationGain > bestInformationGain and len(set1) > 0 and len(set2) > 0:
						bestInformationGain = informationGain
						bestCriteria = (columnIndex, value)
						bestSets = (set1, set2)

		# Recurse and split the current node.
		if bestInformationGain > 0.0:
			leftChild = buildTree(bestSets[0])
			rightChild = buildTree(bestSets[1])
			return DecisionTreeNode(bestCriteria[0], bestCriteria[1], None, leftChild, rightChild)
		else:
			return DecisionTreeNode(results = uniqueCounts(data))


def classify(observation, tree):
	if tree.results != None:
		return tree.results
	else:
		value = observation[tree.columnIndex]
		if isNumber(value):
			if float(value) >= tree.value:
				return classify(observation, tree.leftChild)
			else:
				return classify(observation, tree.rightChild)
		else:
			if value == tree.value:
				return classify(observation, tree.leftChild)
			else:
				return classify(observation, tree.rightChild)


def classifyAll(data, tree):
	for row in data:
		print row[EMPLOYEE_ID_COLUMN_INDEX] + " " + str(classify(row, tree))


## Partitions and returns a data set in two.
# @param data			TO BE POPULATED
# @param columnIndex	TO BE POPULATED
# @param value			TO BE POPULATED
# @return				TO BE POPULATED
def divideSet(data, columnIndex, value):
	# Create a function that partitions a row into the left set (true) or the right set (false).
	splitFunction = None
	if isNumber(value):
		splitFunction = lambda row : row[columnIndex] >= float(value)
	else:
		splitFunction = lambda row : row[columnIndex] == value

	# Divide the data into two sets and return them.
	set1 = [row for row in data if splitFunction(row)]
	set2 = [row for row in data if not splitFunction(row)]
	return (set1, set2)


## Draws the current node in the tree.
# @param draw		TO BE POPULATED
# @param headers	TO BE POPULATED
# @param tree		TO BE POPULATED
# @param x			TO BE POPULATED
# @param y			TO BE POPULATED
def drawNode(draw, headers, tree, x, y):
	if tree.results == None:
		# Get the width of each branch.
		leftWidth = 100 * getWidth(tree.leftChild)
		rightWidth = 100 * getWidth(tree.rightChild)

		# Determine the total space required by this node.
		left = x - (leftWidth + rightWidth) / 2
		right = x + (leftWidth + rightWidth) / 2

		# Draw the condition string.
		if isNumber(tree.value):
			draw.text((x - 20, y - 10), str(headers[tree.columnIndex]) + " >= " + str(tree.value), (0, 0, 0))
		else:
			draw.text((x - 20, y - 10), str(headers[tree.columnIndex]) + " is " + str(tree.value), (0, 0, 0))

		# Draw links to the children.
		draw.line((x, y, left + leftWidth / 2, y + 100), fill=(255, 0, 0))
		draw.line((x, y, right - rightWidth / 2, y + 100), fill=(255, 0, 0))

		# Draw the branch nodes.
		drawNode(draw, headers, tree.leftChild, left + leftWidth / 2, y + 100)
		drawNode(draw, headers, tree.rightChild, right - rightWidth / 2, y + 100)
	else:
		# Draw the leaf label.
		leafText = "\n".join(['%s:%d'%v for v in tree.results.items()])
		draw.text((x - 20, y), leafText, (0, 0, 0))


## Draws the tree to "DecisionTree.png".
# @param tree		TO BE POPULATED
# @param headers	TO BE POPULATED
def drawTree(tree, headers):
	width = 100 * getWidth(tree)
	height = 100 * getHeight(tree) + 220

	image = Image.new("RGB", (width, height), (255, 255, 255))
	draw = ImageDraw.Draw(image)
	drawNode(draw, headers, tree, width / 2, 20)
	image.save("DecisionTree.png", "PNG")


## Calculates how different outcomes are from each other.
# @param data	TO BE POPULATED
# @return		TO BE POPULATED
def entropy(data):
	logBase2 = lambda x : math.log(x) / math.log(2)
	results = uniqueCounts(data)

	entropy = 0.0
	for result in results.keys():
		resultRatio = float(results[result]) / float(len(data))
		entropy = entropy - resultRatio * logBase2(resultRatio)
	return entropy


## Calculates the height (depth) of the tree.
# @param tree		TO BE POPULATED
# @return
def getHeight(tree):
	if tree.leftChild == None and tree.rightChild == None:
		return 0
	else:
		return max(getHeight(tree.leftChild), getHeight(tree.rightChild)) + 1


## Calculates the number of leaves in the tree.
# @param tree		TO BE POPULATED
# @return
def getWidth(tree):
	if tree.leftChild == None and tree.rightChild == None:
		return 1
	else:
		return getWidth(tree.leftChild) + getWidth(tree.rightChild)


## Returns True if x is a number and False otherwise.
# @param x		TO BE POPULATED
# @return
def isNumber(x):
	try:
		float(x)
		return True
	except ValueError:
		return False


## Loads and returns CSV data from the provided file path.
# @param filePath		TO BE POPULATED
# @return				TO BE POPULATED
def loadCSVData(filePath):
	csvFile = open(filePath, 'rb')
	reader = csv.reader(csvFile)
	data = []
	for row in reader:
		data.append([])
		for column in row:
			data[len(data) - 1].append(column);
	return data[0], data[1:]


## Computes how mixed the set is and returns the results.
# @param data		TO BE POPULATED
# @return
def uniqueCounts(dataSet):
	results = {}
	for row in dataSet:
		attritionValue = row[ATTRITION_COLUMN_INDEX]
		if attritionValue not in results:
			results[attritionValue] = 0
		results[attritionValue] += 1
	return results



## The starting point for this program's execution.
if __name__ == "__main__":
	if len(sys.argv) != 3:
		print "Usage: python DecisionTree.py <train csv file> <test csv file>"
	else:
		(headers, data) = loadCSVData(sys.argv[1])
		tree = buildTree(data)

		(testHeaders, testData) = loadCSVData(sys.argv[2])
		classifyAll(testData, tree)

		drawTree(tree, headers)
