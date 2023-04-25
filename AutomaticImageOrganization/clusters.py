import os
import shutil

def putSamplesInClusters(modded, sampleList):
    new = []

    # Modify text file to be a 2d array
    for line in modded:
        if line == "":
            continue
        line = line.split(" ")
        for value in line:
            if value == "":
                line.remove(value)
        new.append(line)

    clusters = [[], [], []]

    # put ids of even indexes into separate lists according to the value in the odd indexes at the same location
    for i in range(len(new)):
        for j in range(len(new[i])):
            if i % 2 == 0:
                clusters[int(new[i + 1][j]) - 1].append(sampleList[int(new[i][j]) - 1])

    return clusters


def printClusters(clusters):
    # print clusters
    for i in range(len(clusters)):
        print("Cluster " + str(i + 1) + " (" + str(len(clusters[i])) + ")" + ": " )
        for j in range(len(clusters[i])):
            print(clusters[i][j])
        print("")


def getCluster(sample, clusters):
    for i in range(len(clusters)):
        res = [ele for ele in clusters[i] if(sample in ele)]
        if len(res) != 0:
            return i


def listCluster(clusters):
    while True:
        userInput = input("Enter a sample ID: ")
        answers = []

        if userInput == "exit":
            break
        elif userInput == "print":
            printClusters(clusters)

        cluster = getCluster(userInput, clusters)
        
        if cluster == None:
            print("Sample not found")
        else:
            print("Sample " + userInput + " is in cluster " + str(cluster + 1))
            return

        for i in range(len(clusters)):
            res = [ele for ele in clusters[i] if(userInput in ele)]
            answers.append(res)

        for i in range(len(answers)):
            if len(answers[i]) != 0:
                print("Cluster " + str(i + 1) + ": " + str(answers[i]))
                return i

def putImageInClusterFolder(clusters):
    directory = '/Users/jbahn/Documents/Cropped-Images/ScaledClusters/'
    clusterPaths = [
            '/Users/jbahn/Documents/Cropped-Images/ScaledClusters/cluster1',
            '/Users/jbahn/Documents/Cropped-Images/ScaledClusters/cluster2',
            '/Users/jbahn/Documents/Cropped-Images/ScaledClusters/cluster3'
        ]

    files = os.listdir(directory)

    for file in files:
        sample = []
        if "a" in file:
            sample = file.split("a")[0]
        elif "b" in file:
            sample = file.split("b")[0]
        else:
            print("Not a correct file name: " + file)
            continue
         
        cluster = getCluster(sample, clusters)
        if cluster == None:
            print("Cluster not found for " + sample)
            continue
        else:
            shutil.move(directory + "/" + file, clusterPaths[cluster])

def checkCluster(clusters):
    cluster1 = '/Users/jbahn/Documents/Cropped-Images/Clusters/Cluster1/'
    cluster2 = '/Users/jbahn/Documents/Cropped-Images/Clusters/Cluster2/'
    cluster3 = '/Users/jbahn/Documents/Cropped-Images/Clusters/Cluster3/'

    for file in os.listdir(cluster1):
        sample = getSampleFromFileName(file)
        if sample == None:
            continue

        cluster = getCluster(sample, clusters)
        if cluster != 0:
            print("Sample " + file + " is incorrectly in cluster 1")

    for file in os.listdir(cluster2):
        sample = getSampleFromFileName(file)
        if sample == None:
            continue

        cluster = getCluster(sample, clusters)
        if cluster != 1:
            print("Sample " + file + " is incorrectly in cluster 2")

    for file in os.listdir(cluster3):
        sample = getSampleFromFileName(file)
        if sample == None:
            continue

        cluster = getCluster(sample, clusters)
        if cluster != 2:
            print("Sample " + file + " is incorrectly in cluster 3")

    print("Done")

def getSampleFromFileName(file):
    sample = []

    if "a" in file:
        sample = file.split("a")[0]
    elif "b" in file:
        sample = file.split("b")[0]
    else:
        print("Not a correct file name: " + file)
        return None

    return sample

def main():
    file = open("scaled-clusters.txt", "r")
    sampleFile = open("samples.txt", "r")

    samples = sampleFile.read()

    sampleList = samples.split("\n")

    contents = file.read()

    contents = contents.replace("  ", " ")
    contents = contents.replace("   ", " ")

    modded = contents.split("\n")

    clusters = putSamplesInClusters(modded, sampleList)

    #listCluster(clusters)
    #printClusters(clusters)
    
    putImageInClusterFolder(clusters)

    #checkCluster(clusters)

if __name__ == '__main__':
    main()
