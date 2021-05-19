import csv
def readcsv(patternfile, targetfile):
    with open(patternfile) as csvfile:
        csvdata = csv.reader(csvfile)
        patterndata = []
        for csvrow in csvdata:
            row = []
            for csvnum in csvrow:
                row.append(float(csvnum))
            patterndata.append(row)
    with open(targetfile) as csvfile:
        csvdata = csv.reader(csvfile)
        targetdata = []
        for csvrow in csvdata:
            row = []
            for csvnum in csvrow:
                row.append(float(csvnum))
            targetdata.append(row)

    return patterndata, targetdata