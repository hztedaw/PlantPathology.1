import csv

input_file = 'D:/Users/3005/Desktop/PlantPathology.1/data/train.csv'  # Replace with your original CSV file name
output_file = 'D:/Users/3005/Desktop/PlantPathology.1/data/binary_train.csv'

with open(input_file, mode='r') as infile, open(output_file, mode='w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # Write the header for the new CSV
    writer.writerow(['Image_ID', 'Label'])

    # Skip the header row in the original CSV
    next(reader)

    for row in reader:
        image_id = row[0]
        # Check if any disease column has '1'
        if '1' in row[2:]:  # Assuming disease columns start from index 2
            label = '0'
        else:
            label = '1'
        writer.writerow([image_id, label])
