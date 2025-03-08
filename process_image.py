import sys
from mpi4py import MPI
import cv2
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def apply_gaussian_blur(image, kernel_size=(7, 7)):
    """
    Efficient Gaussian Blur function with optimized parameters.
    
    - Uses a larger kernel size for better smoothing.
    - Handles boundary artifacts in MPI by adding mirrored padding.
    """
    if image is None or image.size == 0:
        raise ValueError("❌ ERROR: Invalid image!")

    # Add padding to prevent boundary artifacts in parallel processing
    padded_image = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_REFLECT)

    # Apply Gaussian Blur
    blurred_image = cv2.GaussianBlur(padded_image, kernel_size, 0)

    # Remove padding
    return blurred_image[10:-10, 10:-10]

def apply_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def detect_objects(image):
    print(f"🔎 Process {rank}: Running Object Detection...")

    # Load Haar cascade
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Check if cascade file loaded correctly
    if face_cascade.empty():
        print("❌ ERROR: Haar cascade XML file not found! Check OpenCV installation.")
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces with adjusted parameters
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3)

    print(f"✅ Process {rank}: Detected {len(faces)} faces.")

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    return image

def process_image_chunk(chunk, task):
    if task == "blur":
        return apply_gaussian_blur(chunk)
    elif task == "edge":
        return apply_edge_detection(chunk)
    elif task == "object":
        return detect_objects(chunk)
    else:
        raise ValueError("❌ ERROR: Invalid task!")

def process_image(input_path, output_path, task):
    if rank == 0:
        image = cv2.imread(input_path)
        if image is None:
            print(f"❌ ERROR: Could not load {input_path}!")
            sys.exit(1)

        # If task is object detection, run before splitting the image
        if task == "object":
            image = detect_objects(image)

        height, width, _ = image.shape
        chunk_size = height // size
        chunks = [image[i * chunk_size:(i + 1) * chunk_size, :] for i in range(size)]
    else:
        chunks = None

    chunk = comm.scatter(chunks, root=0)
    processed_chunk = process_image_chunk(chunk, task)
    processed_chunks = comm.gather(processed_chunk, root=0)

    if rank == 0:
        final_image = np.vstack(processed_chunks)
        cv2.imwrite(output_path, final_image)
        print(f"✅ Processed image saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        if rank == 0:
            print("❌ ERROR: Usage: mpirun -np 4 python process_image.py <input_path> <output_path> <task>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    task = sys.argv[3]
    
    process_image(input_path, output_path, task)
