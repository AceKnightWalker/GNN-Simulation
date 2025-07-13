import csv
import os

filename = "/home/abdulfatai/Downloads/Internship/GNN-Simulation/Rxncgr/dataset/full_rdb7/val.csv"
temp_filename = "temp.csv"

# Read original, write to temp without the first column
with open(filename, newline='', encoding='utf-8') as infile, \
     open(temp_filename, 'w', newline='', encoding='utf-8') as outfile:
    
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    for row in reader:
        writer.writerow(row[1:])  # Skip the first column

# Replace original file with the modified one
os.replace(temp_filename, filename)
