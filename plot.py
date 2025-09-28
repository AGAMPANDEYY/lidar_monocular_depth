import matplotlib.pyplot as plt

# Class names and counts
classes = [
    "Car", "MTW", "HV", "Pedestrians", "Rickshaw", "LCV", "Cycle", "Others"
]
counts = [5194, 1234, 934, 593, 397, 298, 84, 2]

plt.figure(figsize=(10, 6))
bars = plt.bar(classes, counts, color='skyblue', edgecolor='black')
plt.xlabel("Class", fontsize=14)
plt.ylabel("Number of Annotations", fontsize=14)
#plt.title("YOLOv11 Training Data Annotation Class Distribution", fontsize=16)
plt.xticks(rotation=30, fontsize=12)
plt.yticks(fontsize=12)

# Annotate bar values
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
             f"{int(bar.get_height())}", ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.show()