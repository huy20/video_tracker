import os

import cv2
import numpy as np
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Paths
input_video_path = 'Feel_Special - Trim.mp4'
output_video_path = 'output.avi'

# Video I/O setup
stream = cv2.VideoCapture(input_video_path)
if not stream.isOpened():
    print("Error: Could not open video.")
    exit()

frame_width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = stream.get(cv2.CAP_PROP_FPS)
target_size = (832, 480)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, target_size)

# Workflow
wf = Workflow()

# YOLOv8 detection
detector = wf.add_task(name="infer_yolo_v8", auto_connect=True)
detector.set_parameters({"conf_thres": "0.4"})

# DeepSORT tracking
tracking = wf.add_task(name="infer_deepsort", auto_connect=True)
tracking.set_parameters({"categories": "person", "conf_thres": "0.3"})

# ORB for visual Re-ID
orb = cv2.ORB_create(nfeatures=300)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Re-ID storage
trackid_to_personid = {}        # deepSORT_id -> consistent person ID
personid_descriptors = {}       # person ID -> descriptors
next_person_id = 0
MAX_PERSONS = 10
frame_count = 0

def find_matching_person_id(desc_new):
    for pid, desc_old in personid_descriptors.items():
        if desc_old is None or desc_new is None:
            continue
        matches = bf.match(desc_new, desc_old)
        matches = sorted(matches, key=lambda x: x.distance)
        if len(matches) >= 10 and matches[0].distance < 0.2 * 100:
            return pid
    return None

# Process each frame
while True:
    ret, frame = stream.read()
    if not ret:
        print("Info: End of video or error.")
        break
    
    # Resize frame for speed
    frame = cv2.resize(frame, target_size)
    frame_count += 1
    
    # Run workflow
    wf.run_on(array=frame)
    
    # Get detection and tracking results
    image_out = tracking.get_output(0)
    obj_detect_out = tracking.get_output(1)
    objects = obj_detect_out.get_objects()
    
    for obj in objects:
        deepsort_id = obj.label  # Convert to int for consistency
        x, y, w, h = map(int, obj.box)
        x, y = max(0, x), max(0, y)
        
        person_crop = frame[y:y+h, x:x+w]
        if person_crop.size == 0:
            continue
        
        # Extract descriptor
        gray_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = orb.detectAndCompute(gray_crop, None)
        
        if descriptors is None:
            continue
        
        # Assign consistent person ID
        if deepsort_id not in trackid_to_personid:
            # First time seeing this DeepSORT ID
            match_id = find_matching_person_id(descriptors)
            if match_id is not None:
                # Found a visual match with existing person
                trackid_to_personid[deepsort_id] = match_id
                # Update descriptors for this person
                personid_descriptors[match_id] = descriptors
            elif next_person_id < MAX_PERSONS:
                # New person, assign new ID
                trackid_to_personid[deepsort_id] = next_person_id
                personid_descriptors[next_person_id] = descriptors
                next_person_id += 1
            else:
                # Max limit reached - skip this detection
                print(f"Warning: Max persons ({MAX_PERSONS}) reached. Skipping DeepSORT ID {deepsort_id}")
                continue
        else:
            # Already seen this DeepSORT ID, update descriptors
            pid = trackid_to_personid[deepsort_id]
            personid_descriptors[pid] = descriptors
        
        # Get the assigned person ID (guaranteed to be 0-9)
        assigned_person_id = trackid_to_personid[deepsort_id]
        
        # Verify the assigned ID is within bounds
        if assigned_person_id >= MAX_PERSONS:
            print(f"Error: Assigned person ID {assigned_person_id} exceeds MAX_PERSONS {MAX_PERSONS}")
            continue
        
        # Set the label to the consistent person ID (0-9)
        obj.label = str(assigned_person_id)
    
    # Render and save
    img_out = image_out.get_image_with_graphics(obj_detect_out)
    img_res = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)
    out.write(img_out)  # Fixed: write img_out instead of img_res
    
    # Display
    display(img_res, title="DeepSORT with Consistent IDs", viewer="opencv")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
stream.release()
out.release()
cv2.destroyAllWindows()

print(f"Processing complete. Total unique persons tracked: {len(personid_descriptors)}")
print(f"DeepSORT ID mappings: {trackid_to_personid}")