# Name      :   Pranav Bhandari
# Student ID:   1001551132
# Date      :   10/27/2020

import sys, numpy as np, random, math

# Class that represents a Decision Tree, with each node containing the attribute and the threshold
class Tree:
    def __init__(self):
        self.left = None
        self.right = None
        self.attribute = -1
        self.threshold = -1
        self.informationGain = 0
        self.classLabelDistribution = None

# Function that calculates entropy of a set of training examples
def entropy(output):
    unique = []
    counts = []
    numExamples = len(output)
    for i in range(numExamples):
        try:
            index = unique.index(output[i])
            counts[index] += 1
        except ValueError:
            unique.append(output[i])
            counts.append(1)
    result = 0
    for i in range(len(unique)):
        ratio = float(counts[i])/numExamples
        result +=  -1 * ratio * math.log(ratio, 2)
    return result

# Function that calculates the Information Gain when an attribute and a threshold is applied to
# a set of training examples
def informationGain(attributes, output, i, threshold):
    attributes_left, output_left, attributes_right, output_right = splitExamples(i, 
                                                                    attributes, output, threshold)
    length = len(output)
    weight_left = float(len(output_left))/length
    weight_right = float(len(output_right))/length
    info_gain = entropy(output) - weight_left*entropy(output_left) - weight_right*entropy(output_right)
    return info_gain

# This function chooses the best threshold and attribute pair for the most information gain.    
def optimizedChooseAttribute(attributes, output):
    max_gain = best_attribute = best_threshold = -1
    for i in range(len(attributes[0])):
        L = M = attributes[0][i]
        for j in range(len(attributes)):
            if attributes[j][i] < L:
                L = attributes[j][i]
            if attributes[j][i] > M:
                M = attributes[j][i]
        for K in range(1, 51):
            threshold = L + K*(M-L)/51
            gain = informationGain(attributes, output, i, threshold)
            if gain > max_gain:
                max_gain = gain
                best_attribute = i
                best_threshold = threshold
    return best_attribute, best_threshold, max_gain      
    
# This function choses an attribute randomly and finds the best threshold with the most
# information gain for that attribute
def randomizedChooseAttribute(attributes, output):
    max_gain = best_attribute = best_threshold = -1
    best_attribute = random.randint(0, len(attributes[0])-1)
    attribute_values = [row[best_attribute] for row in attributes]
    L = min(attribute_values)
    M = max(attribute_values)
    for K in range(1, 51):
        threshold = L + K*(M-L)/51
        gain = informationGain(attributes, output, best_attribute, threshold)
        if gain > max_gain:
            max_gain = gain
            best_threshold = threshold
    return best_attribute, best_threshold, max_gain

# This function splits the training examples based on the threshold and the attribute.
def splitExamples(curr_attribute, attributes, output, curr_threshold):
    attributes_left = []
    attributes_right = []
    output_left = []
    output_right = []
    for i in range(len(attributes)):
        if attributes[i][curr_attribute] < curr_threshold:
            attributes_left.append(attributes[i])
            output_left.append(output[i])
        else:
            attributes_right.append(attributes[i])
            output_right.append(output[i])
    return attributes_left, output_left, attributes_right, output_right 

# Recursive function which builds the tree
def DTL(attributes, output, default, pruning_thr, option, numClasses):
    # If there are not enough examples, then we return a Tree with the root as the default class
    # or
    # If all examples belong to the same class, we return a Tree with the root as that class
    if len(attributes) < pruning_thr or len(np.unique(np.array(output))) == 1:
        tree = Tree()
        tree.classLabelDistribution = default
        return tree
    else:
        if option == "optimized":
            best_attribute, best_threshold, max_gain = optimizedChooseAttribute(attributes, output)
        else:
            best_attribute, best_threshold, max_gain = randomizedChooseAttribute(attributes, output)
        tree = Tree()
        tree.attribute = best_attribute+1
        tree.threshold = best_threshold
        tree.informationGain = max_gain
        dist = distribution(output, numClasses)
        attributes_left, output_left, attributes_right, output_right = splitExamples(best_attribute, attributes, 
                                                                                    output, best_threshold)
        tree.left = DTL(attributes_left, output_left, dist, pruning_thr, option, numClasses)
        tree.right = DTL(attributes_right, output_right, dist, pruning_thr, option, numClasses)
        return tree

# Calculates the probability of each class label, and returns the most probable class label
def distribution(output, numClasses):
    result = [0.0 for i in range(numClasses)]
    total = len(output)
    for i in range(len(output)):
        result[output[i]] += 1
    return [num/total for num in result]

# Reads the file and returns the attributes and class labels for each test/training example
def readFile(file):
    file = open(file, "r")
    attributes = []
    output = []
    for row in file:
        temp = row.split()
        length = len(temp)
        intermediate = []
        for i in range(length-1):
            intermediate.append(float(temp[i]))
        output.append(int(temp[length-1]))
        attributes.append(intermediate)
    return attributes, output

# Top layer function which gets the attributes and output and calls the recursive DTL function
# based on the option given by the user    
def trainingTree(training_file, option, pruning_thr):
    attributes, output = readFile(training_file)
    np_output = np.array(output)
    unique = np.unique(np_output)
    numClasses = len(unique)
    default = distribution(output, numClasses)
    trees = []
    if option == "optimized" or option == "randomized":
        trees.append(DTL(attributes, output, default, pruning_thr, option, numClasses))
    elif option == "forest3":
        for i in range(3):
            trees.append(DTL(attributes, output, default, pruning_thr, option, numClasses))
    elif option == "forest15":
        for i in range(15):
            trees.append(DTL(attributes, output, default, pruning_thr, option, numClasses))
    return trees

# This function prints the tree in Breath-First manner
def printTree(trees):
    for i in range(len(trees)):
        queue = []
        queue.append(trees[i])
        node_id = 1
        while len(queue) !=0:
            temp = queue.pop()
            print("tree={:2d}, node={:3d}, feature={:2d}, thr={:6.2f}, gain={:f}".format(i+1, node_id, temp.attribute, temp.threshold, temp.informationGain))
            node_id+=1
            if temp.left != None and temp.right!=None:
                queue.append(temp.right)
                queue.append(temp.left)

# This function tests the Decision trees created with test examples in the test file
# and prints the classification accuracy            
def testTree(test_file, trees):
    classificationAccuracy = 0.0
    attributes, output = readFile(test_file)
    for i in range(len(attributes)):
        results = []
        for tree in trees:
            while tree != None:
                # Leaf Node
                if tree.threshold == -1:
                    if len(results) == 0:
                        results = tree.classLabelDistribution
                    else:
                        results = [a+b for a,b in zip(results, tree.classLabelDistribution)]
                    break
                if attributes[i][tree.attribute-1] < tree.threshold:
                    tree = tree.left
                else:
                    tree = tree.right
        predicted = np.argmax(results)
        accuracy = 1.0 if predicted == output[i] else 0.0
        classificationAccuracy += accuracy
        print("ID={:5d}, predicted={:3d}, true={:3d}, accuracy={:4.2f}".format(i+1, predicted, output[i], accuracy))
    classificationAccuracy = classificationAccuracy/len(attributes)
    print('classification accuracy={:6.4f}'.format(classificationAccuracy))

def decision_tree(training_file, test_file, option, pruning_thr):
    trees = trainingTree(training_file, option, pruning_thr)
    printTree(trees)
    testTree(test_file, trees)
    return

if __name__ == '__main__':
    decision_tree(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
