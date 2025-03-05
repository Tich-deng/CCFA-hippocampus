import os

import ultralytics

# Initialize
model = ultralytics.YOLO('yolo11x-seg.pt')

files = os.listdir('./patch')

for file in files:
    # Inference
    results = model('patch/' + file)
    results = results[0]
    results.save('output/' + file)
