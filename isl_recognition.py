import cv2
import mediapipe as mp
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle
import os
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,       # Process video stream
    max_num_hands=2,             # Detect up to two hands
    min_detection_confidence=0.5, # Minimum confidence for initial detection
    min_tracking_confidence=0.5  # Minimum confidence for tracking
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles # For nicer landmark drawing

# Constants
NUM_LANDMARKS = 21  # Number of hand landmarks in MediaPipe
LANDMARK_DIMS = 3   # x, y, z coordinates for each landmark
MAX_HANDS = 2       # Maximum number of hands the model expects features for
# --- Corrected: Defined SINGLE_HAND_FEATURE_SIZE globally ---
SINGLE_HAND_FEATURE_SIZE = NUM_LANDMARKS * LANDMARK_DIMS # Size for one hand's landmarks
FEATURE_VECTOR_SIZE = SINGLE_HAND_FEATURE_SIZE * MAX_HANDS # Total size of the input vector for the model

# --- Utility Functions ---

def get_screen_resolution():
    """Tries to get screen resolution. Falls back to Full HD."""
    try:
        import screeninfo
        # Get the primary monitor
        primary_monitor = None
        for m in screeninfo.get_monitors():
            if m.is_primary:
                primary_monitor = m
                break
        # If no primary, just take the first one
        screen = primary_monitor if primary_monitor else screeninfo.get_monitors()[0]
        return screen.width, screen.height
    except ImportError:
        print("Warning: 'screeninfo' library not found. Falling back to 1920x1080.")
        print("Install using: pip install screeninfo")
        return 1920, 1080 # Fallback resolution
    except Exception as e:
        print(f"Warning: Error getting screen info ({e}). Falling back to 1920x1080.")
        return 1920, 1080

SCREEN_WIDTH, SCREEN_HEIGHT = get_screen_resolution()

# Function to extract hand landmarks
def extract_landmarks(image):
    """
    Processes an image to detect hands and extract landmarks.

    Args:
        image: The input image (BGR format).

    Returns:
        A tuple containing:
        - left_landmarks (np.array): Landmarks for the left hand (size SINGLE_HAND_FEATURE_SIZE, padded with zeros if not detected).
        - right_landmarks (np.array): Landmarks for the right hand (size SINGLE_HAND_FEATURE_SIZE, padded with zeros if not detected).
        - image_with_drawings (np.array): The input image with landmarks drawn.
        - hands_detected (list): A list [bool, bool] indicating if [left, right] hands were detected.
    """
    image_to_process = image.copy() # Work on a copy
    image_rgb = cv2.cvtColor(image_to_process, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False # Improve performance
    results = hands.process(image_rgb)
    image_rgb.flags.writeable = True # Back to writeable
    image_with_drawings = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR) # Convert back for drawing

    # Initialize arrays using the global constant
    left_landmarks = np.zeros(SINGLE_HAND_FEATURE_SIZE)
    right_landmarks = np.zeros(SINGLE_HAND_FEATURE_SIZE)
    hands_detected = [False, False]  # [left, right]

    # --- Removed redundant local definition: single_hand_feature_size = NUM_LANDMARKS * LANDMARK_DIMS ---

    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if hand_idx >= MAX_HANDS: # Should not happen with max_num_hands=2, but good practice
                break

            # Determine if it's left or right hand (requires multi_handedness)
            hand_type = "Unknown"
            if results.multi_handedness and len(results.multi_handedness) > hand_idx:
                 hand_type = results.multi_handedness[hand_idx].classification[0].label

            # Draw hand landmarks on the image
            mp_drawing.draw_landmarks(
                image_with_drawings,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Extract coordinates into a temporary flat array
            # Ensure we handle cases where landmarks might be missing (though unlikely with default settings)
            landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            # Verify size before assignment (optional safety check)
            if landmarks_array.shape[0] != SINGLE_HAND_FEATURE_SIZE:
                print(f"Warning: Landmark array size mismatch for hand {hand_type}. Expected {SINGLE_HAND_FEATURE_SIZE}, got {landmarks_array.shape[0]}. Skipping hand.")
                continue # Skip this hand if size is wrong


            # Store landmarks based on hand type
            if hand_type == "Left":
                left_landmarks = landmarks_array
                hands_detected[0] = True
            elif hand_type == "Right":
                right_landmarks = landmarks_array
                hands_detected[1] = True
            # else: Handle "Unknown" if necessary, maybe assign based on position? For now, ignore.

    return left_landmarks, right_landmarks, image_with_drawings, hands_detected

# --- Data Collection ---

def list_available_cameras():
    """Lists all available camera devices."""
    available_cameras = []
    for i in range(10):  # Check first 10 indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(i)
            cap.release()
    return available_cameras

def select_camera():
    """Allows user to select a camera device."""
    available_cameras = list_available_cameras()
    if not available_cameras:
        print("No cameras found!")
        return 0
    
    print("\nAvailable cameras:")
    for cam_idx in available_cameras:
        print(f"Camera {cam_idx}")
    
    while True:
        try:
            choice = int(input("\nSelect camera number (0-9): "))
            if choice in available_cameras:
                return choice
            print("Invalid camera number. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

def collect_training_data(gesture_class, num_samples=100):
    """Collects training samples for a given gesture class."""
    camera_idx = select_camera()
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_idx}.")
        return [], []

    data = []
    labels = []
    count = 0

    # Ask if the gesture requires both hands
    print(f"\nGesture: {gesture_class}")
    print("Does this gesture require both hands?")
    print("1. Yes - Both hands together")
    print("2. No - One hand at a time")
    while True:
        choice = input("Enter your choice (1/2): ")
        if choice in ['1', '2']:
            break
        print("Invalid choice. Please enter 1 or 2.")
    requires_both_hands = (choice == '1')

    print("-" * 30)
    print(f"Collecting data for gesture: {gesture_class} ({num_samples} samples)")
    print("Press 'q' to stop early.")
    if requires_both_hands:
        print("Instruction: Show the gesture using BOTH hands simultaneously.")
    else:
        print("Instruction: Show the gesture using EITHER your left OR right hand.")
    print("Ensure the hand(s) are clearly visible in the center.")
    print("-" * 30)
    time.sleep(2) # Give user time to read

    # --- Fullscreen Window Setup ---
    window_name = f'Collecting Data: {gesture_class}'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # Attempt fullscreen - might not work perfectly on all OS/Window Managers
    # cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) # Can be glitchy
    cv2.resizeWindow(window_name, SCREEN_WIDTH, SCREEN_HEIGHT) # Often more reliable
    cv2.moveWindow(window_name, 0, 0)


    # Get initial frame dimensions for scaling calculation
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read initial frame from camera.")
        cap.release()
        cv2.destroyAllWindows()
        return [], []
    frame_h, frame_w = frame.shape[:2]

    # Calculate scaling to fit frame within screen while preserving aspect ratio
    scale_w = SCREEN_WIDTH / frame_w if frame_w > 0 else 1
    scale_h = SCREEN_HEIGHT / frame_h if frame_h > 0 else 1
    scale = min(scale_w, scale_h)
    disp_w = int(frame_w * scale)
    disp_h = int(frame_h * scale)

    # Calculate offsets to center the display frame
    offset_x = (SCREEN_WIDTH - disp_w) // 2
    offset_y = (SCREEN_HEIGHT - disp_h) // 2

    # Info panel configuration (relative to the *display* area)
    panel_width = 400
    panel_start_x = disp_w - panel_width if disp_w > panel_width else 0 # Position within displayed frame
    panel_start_x_screen = offset_x + panel_start_x                  # Absolute screen X
    panel_end_x_screen = offset_x + disp_w                            # Absolute screen X
    panel_start_y_screen = offset_y                                   # Absolute screen Y
    panel_end_y_screen = offset_y + disp_h                            # Absolute screen Y
    text_x_start = panel_start_x_screen + 20                          # Text start position on screen
    text_panel_width_screen = panel_end_x_screen - text_x_start - 20  # Max text width on screen


    last_collection_time = time.time()
    collection_delay = 0.1 # Minimum seconds between collecting samples

    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Dropped frame.")
            continue

        # --- Landmark Extraction (on original frame) ---
        # Process the *original* frame for accurate landmarks
        left_landmarks, right_landmarks, _, hands_detected = extract_landmarks(frame) # We don't need the drawn image from here

        # --- Display Frame Preparation ---
        display_canvas = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8) # Black canvas
        resized_frame = cv2.resize(frame, (disp_w, disp_h)) # Resize for display

        # Place resized frame onto the canvas
        display_canvas[offset_y:offset_y+disp_h, offset_x:offset_x+disp_w] = resized_frame

        # --- Draw Landmarks on Display Frame ---
        # We need to re-run detection/drawing on the *displayed portion* or scale landmarks.
        # Re-running is simpler for display purposes.
        display_frame_slice = display_canvas[offset_y:offset_y+disp_h, offset_x:offset_x+disp_w]
        # Make a copy to avoid modifying the canvas directly if drawing fails
        display_slice_copy = display_frame_slice.copy()
        display_slice_rgb = cv2.cvtColor(display_slice_copy, cv2.COLOR_BGR2RGB)
        display_slice_rgb.flags.writeable = False
        results_display = hands.process(display_slice_rgb)
        display_slice_rgb.flags.writeable = True
        # Draw onto the slice *within* the main canvas
        if results_display.multi_hand_landmarks:
            for hand_landmarks_disp in results_display.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    display_frame_slice, # Draw directly on the slice in the canvas
                    hand_landmarks_disp,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
        # No need to put slice back if we drew directly on it


        # --- Information Panel Overlay ---
        overlay = display_canvas.copy()
        cv2.rectangle(overlay, (panel_start_x_screen, panel_start_y_screen),
                      (panel_end_x_screen, panel_end_y_screen), (0, 0, 0), -1) # Black rectangle
        alpha = 0.6 # Transparency
        display_canvas = cv2.addWeighted(overlay, alpha, display_canvas, 1 - alpha, 0)


        # --- Data Collection Logic ---
        current_time = time.time()
        collected_this_frame = False
        can_collect = (current_time - last_collection_time) > collection_delay

        feature_vector = None
        collection_message = ""

        if requires_both_hands:
            if all(hands_detected):
                if can_collect:
                    feature_vector = np.concatenate([left_landmarks, right_landmarks])
                    collection_message = f"Collected {count + 1}/{num_samples} (Both Hands)"
                else:
                     collection_message = "Hold steady..."
            else:
                collection_message = "Waiting for BOTH hands..."
        else: # Single hand mode
            if hands_detected[0]: # Left hand detected
                if can_collect:
                    # --- Corrected: Use global constant ---
                    feature_vector = np.concatenate([left_landmarks, np.zeros(SINGLE_HAND_FEATURE_SIZE)])
                    collection_message = f"Collected {count + 1}/{num_samples} (Left Hand)"
                else:
                     collection_message = "Hold steady..."
            elif hands_detected[1]: # Right hand detected (and left wasn't)
                 if can_collect:
                    # --- Corrected: Use global constant ---
                    feature_vector = np.concatenate([np.zeros(SINGLE_HAND_FEATURE_SIZE), right_landmarks])
                    collection_message = f"Collected {count + 1}/{num_samples} (Right Hand)"
                 else:
                     collection_message = "Hold steady..."
            else:
                collection_message = "Waiting for ONE hand..."

        if feature_vector is not None:
            # Verify feature vector size before appending
            if feature_vector.shape[0] == FEATURE_VECTOR_SIZE:
                data.append(feature_vector)
                labels.append(gesture_class)
                count += 1
                collected_this_frame = True
                last_collection_time = current_time # Reset timer
                print(collection_message) # Print to console as well
            else:
                 print(f"Error: Feature vector size mismatch! Expected {FEATURE_VECTOR_SIZE}, got {feature_vector.shape[0]}")
                 collection_message = "Size Error!" # Show error on screen


        # --- Display Information Panel Text ---
        y_pos = panel_start_y_screen + 50
        line_height = 40
        text_color = (0, 255, 0) # Green

        def draw_text(text, y, size=0.8, color=text_color, bold=False):
            thickness = 2 if bold else 1
            try:
                cv2.putText(display_canvas, text, (text_x_start, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness, cv2.LINE_AA)
            except Exception as e:
                print(f"Error drawing text '{text}': {e}") # Handle potential errors during drawing
            return y + line_height

        y_pos = draw_text('ISL GESTURE COLLECTION', y_pos, size=1.0, bold=True)
        cv2.line(display_canvas, (text_x_start, y_pos - (line_height // 2)), (panel_end_x_screen - 20, y_pos - (line_height // 2)), text_color, 1)

        y_pos = draw_text(f'Gesture: {gesture_class}', y_pos, size=0.9, bold=True)
        y_pos = draw_text(f'Progress: {count}/{num_samples}', y_pos)
        y_pos += line_height // 2 # Extra space

        y_pos = draw_text('INSTRUCTIONS:', y_pos, bold=True)
        cv2.line(display_canvas, (text_x_start, y_pos - (line_height // 2)), (panel_end_x_screen - 20, y_pos - (line_height // 2)), text_color, 1)
        instr = ['Show gesture clearly.', 'Hold steady for collection.', 'Press "q" to quit.']
        if requires_both_hands: instr.insert(0,'Use BOTH hands.')
        else: instr.insert(0, 'Use ONE hand (Left or Right).')
        for line in instr: y_pos = draw_text(line, y_pos, size=0.7)
        y_pos += line_height // 2

        y_pos = draw_text('HAND STATUS:', y_pos, bold=True)
        cv2.line(display_canvas, (text_x_start, y_pos - (line_height // 2)), (panel_end_x_screen - 20, y_pos - (line_height // 2)), text_color, 1)
        status_color_left = (0, 255, 0) if hands_detected[0] else (0, 0, 255)
        status_color_right = (0, 255, 0) if hands_detected[1] else (0, 0, 255)
        y_pos = draw_text(f'Left Hand: {"Detected" if hands_detected[0] else "Not Detected"}', y_pos, color=status_color_left)
        y_pos = draw_text(f'Right Hand: {"Detected" if hands_detected[1] else "Not Detected"}', y_pos, color=status_color_right)
        y_pos += line_height // 2

        status_color = (0, 255, 0) if collected_this_frame else (0, 255, 255) # Green if collected, Yellow otherwise
        if "Error" in collection_message: status_color = (0,0,255) # Red for errors
        y_pos = draw_text(collection_message, y_pos, color=status_color, bold=True)

        # --- Show Frame ---
        cv2.imshow(window_name, display_canvas)

        # --- Handle Keypress ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nCollection stopped early by user.")
            break
        # elif key == ord('f'): # Fullscreen toggle can be unreliable
        #      current_mode = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
        #      cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
        #                            cv2.WINDOW_NORMAL if current_mode == cv2.WINDOW_FULLSCREEN else cv2.WINDOW_FULLSCREEN)

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    print(f"Finished collecting data for {gesture_class}.")
    return data, labels

# --- Model Training ---

def train_model(data, labels):
    """Trains an SVM model on the collected data."""
    if not data or not labels:
        print("Error: No data provided for training.")
        return None

    X = np.array(data)
    y = np.array(labels)
    unique_labels = np.unique(y)

    if len(unique_labels) < 2:
        print(f"Error: Need data for at least two different gestures to train. Found only: {unique_labels}")
        return None

    # Check if any class has fewer samples than required for stratification (default is 2 for test_split)
    min_samples_per_class = 2
    counts = {label: np.sum(y == label) for label in unique_labels}
    can_stratify = all(count >= min_samples_per_class for count in counts.values())

    # Determine test size (e.g., 20%, but ensure at least 1 sample per class in test set if possible)
    test_size = 0.2

    print(f"Total samples: {len(X)}")
    print(f"Gestures: {counts}")


    if len(X) < 5 or not can_stratify: # Not enough samples overall or per class for reliable split/stratification
        print(f"Warning: Few samples or classes with < {min_samples_per_class} samples. Using all data for training, accuracy based on training data.")
        X_train, X_test, y_train, y_test = X, [], y, [] # No test set
    else:
        try:
             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
             print(f"Training with {len(X_train)} samples, Testing with {len(X_test)} samples (Stratified).")
        except ValueError as e:
             print(f"Warning: Stratified split failed ({e}). Using non-stratified split.")
             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
             print(f"Training with {len(X_train)} samples, Testing with {len(X_test)} samples (Non-Stratified).")


    print("Training SVM model...")
    # Use probability=True to get confidence scores later
    model = SVC(kernel='linear', probability=True, random_state=42, C=1.0) # C is regularization parameter
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"Error during model training: {e}")
        return None

    # Evaluate accuracy
    if len(X_test) > 0:
        accuracy = model.score(X_test, y_test)
        print(f"Model accuracy on TEST set: {accuracy:.4f}")
    else:
        # If no test set, show training accuracy (likely optimistic)
        accuracy = model.score(X_train, y_train)
        print(f"Model accuracy on TRAINING set: {accuracy:.4f}")


    return model

# --- Data Management ---

MODEL_FILENAME = 'isl_gesture_model.pkl'
DATA_FILENAME = 'isl_gesture_data.pkl' # Store collected data separately

def save_data(data, labels, path=DATA_FILENAME):
    """Saves collected data and labels."""
    # Ensure data is serializable (convert numpy arrays if necessary, though pickle handles them)
    try:
        with open(path, 'wb') as f:
            pickle.dump({'data': data, 'labels': labels}, f)
        print(f"Data saved to {path}")
    except Exception as e:
        print(f"Error saving data to {path}: {e}")

def load_data(path=DATA_FILENAME):
    """Loads collected data and labels."""
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                saved_data = pickle.load(f)
            print(f"Data loaded from {path}")
            # Basic validation
            data = saved_data.get('data', [])
            labels = saved_data.get('labels', [])
            if isinstance(data, list) and isinstance(labels, list):
                 if len(data) == len(labels):
                     return data, labels
                 else:
                     print("Warning: Loaded data and labels have different lengths. Returning empty.")
                     return [], []
            else:
                 print("Warning: Loaded data is not in the expected list format. Returning empty.")
                 return [], []
        except Exception as e:
            print(f"Error loading data from {path}: {e}")
            return [], []
    else:
        # print("Data file not found.") # Reduce noise on startup
        return [], []

def save_model(model, path=MODEL_FILENAME):
    """Saves the trained model."""
    if model:
        try:
            with open(path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Model saved to {path}")
        except Exception as e:
            print(f"Error saving model to {path}: {e}")

def load_model(path=MODEL_FILENAME):
    """Loads the trained model."""
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                model = pickle.load(f)
            print(f"Model loaded from {path}")
            # Check if the loaded object looks like a scikit-learn estimator
            if hasattr(model, 'predict') and hasattr(model, 'score'):
                return model
            else:
                print("Error: Loaded file does not appear to be a valid scikit-learn model.")
                return None
        except Exception as e:
            print(f"Error loading model from {path}: {e}")
            # Optionally delete corrupted file
            # try: os.remove(path) except OSError: pass
            return None
    else:
        # print("Model file not found.") # Reduce noise on startup
        return None

# --- Main Application Logic ---

def main():
    """Main function to handle data collection, training, and recognition."""

    model = None
    all_data, all_labels = [], []
    available_gestures = []

    # --- Initial Setup ---
    print("\n" + "="*30)
    print("  ISL Gesture Recognition")
    print("="*30)

    # Try loading existing data and model first
    model = load_model('static/isl_gesture_model.pkl')
    all_data, all_labels = load_data('static/isl_gesture_data.pkl')

    if all_data and all_labels:
        available_gestures = sorted(list(np.unique(all_labels)))
        print(f"Loaded {len(all_data)} samples for gestures: {', '.join(available_gestures)}")
    elif all_data or all_labels:
         print("Warning: Inconsistent data loaded (data/labels mismatch). Ignoring loaded data.")
         all_data, all_labels = [], [] # Reset

    if model:
         # If we loaded data AND a model, assume the model corresponds to the data for now
         # A more robust system might store metadata (e.g., gestures included, feature size) with the model
         if not all_data: # Model exists but no data file - risky, model might be for different data
              print("Warning: Model file exists but no data file found. Model might be outdated or incompatible.")
              proceed = input("Use loaded model anyway? (y/n): ").lower()
              if proceed == 'y':
                  model = model
                  print("Proceeding with loaded model. Gesture labels might be inaccurate.")
                  # We don't know the gestures this model was trained on. Get them from the model itself.
                  if hasattr(model, 'classes_'):
                       available_gestures = sorted(list(model.classes_))
                       print(f"Model expects gestures: {', '.join(available_gestures)}")
                  else:
                       print("Could not determine gestures from loaded model.")
                       available_gestures = []

              else:
                  model = None # Discard the loaded model
         else:
              model = model # Both loaded, assume they match
              # Verify model classes match loaded data labels
              if hasattr(model, 'classes_'):
                    model_gestures = sorted(list(model.classes_))
                    if set(model_gestures) != set(available_gestures):
                         print("Warning: Model classes do not match loaded data labels!")
                         print(f"  Model has: {model_gestures}")
                         print(f"  Data has: {available_gestures}")
                         retrain = input("Retrain model with current data? (y/n): ").lower()
                         if retrain == 'y':
                              model = None # Force retrain later if chosen
                         else:
                              print("Keeping existing model. Predictions might be inconsistent.")
              else:
                    print("Warning: Could not verify classes of loaded model.")


    while True:
        print("\n--- Main Menu ---")
        if available_gestures:
             print(f"Current Gestures: {', '.join(available_gestures)}")
        else:
             print("No gesture data loaded or collected yet.")
        if model:
             print("Model is loaded/trained.")
        else:
             print("Model needs training.")

        print("-" * 17)
        print("1. Collect/Add Gesture Data")
        print("2. Train/Retrain Model")
        print("3. Start Real-time Recognition")
        print("4. View Collected Data Summary")
        print("5. Clear ALL Data and Model")
        print("q. Quit")
        print("-" * 17)

        choice = input("Enter your choice: ").strip().lower()

        if choice == '1':
            gesture_name = input("Enter name for the gesture (e.g., 'Hello', 'A'): ").strip()
            if not gesture_name:
                print("Gesture name cannot be empty.")
                continue

            # Check if overwriting or adding new
            is_new_gesture = gesture_name not in available_gestures
            if not is_new_gesture:
                 overwrite = input(f"Gesture '{gesture_name}' exists. Overwrite samples? (y/n): ").lower()
                 if overwrite != 'y':
                     print(f"Keeping existing samples for '{gesture_name}'.")
                     continue # Go back to menu if not overwriting
                 else: # Remove existing data for this specific gesture
                     # Create new lists keeping only data for other gestures
                     temp_data = []
                     temp_labels = []
                     for i, label in enumerate(all_labels):
                         if label != gesture_name:
                             temp_data.append(all_data[i])
                             temp_labels.append(label)
                     all_data = temp_data
                     all_labels = temp_labels
                     print(f"Removed existing samples for '{gesture_name}'. Ready to collect new.")
                     # Update available gestures immediately if all samples were removed
                     available_gestures = sorted(list(np.unique(all_labels)))


            # Get number of samples
            while True:
                try:
                    num_samples_str = input(f"Enter number of samples for '{gesture_name}' (e.g., 100): ").strip()
                    if not num_samples_str: # Default if empty
                         num_samples = 100
                         print("Using default: 100 samples.")
                         break
                    num_samples = int(num_samples_str)
                    if num_samples > 0:
                        break
                    else:
                        print("Please enter a positive number.")
                except ValueError:
                    print("Invalid input. Please enter a number.")

            # Collect data
            new_data, new_labels = collect_training_data(gesture_name, num_samples)

            # Process collected data
            if new_data:
                all_data.extend(new_data)
                all_labels.extend(new_labels)
                available_gestures = sorted(list(np.unique(all_labels))) # Update list
                print(f"Finished collecting. Total samples for '{gesture_name}': {np.sum(np.array(all_labels) == gesture_name)}")
                print(f"Total samples overall: {len(all_labels)}")
                save_data(all_data, all_labels, 'static/isl_gesture_data.pkl') # Save updated data
                model = None # Mark model as needing retraining
                print("-> Model needs retraining (use Option 2).")
            else:
                print(f"No new data was collected for '{gesture_name}'.")


        elif choice == '2':
            if not all_data:
                print("No data available to train. Please collect data first (Option 1).")
                continue
            if len(available_gestures) < 2:
                print("Need data for at least TWO different gestures to train a classifier.")
                continue

            print("Starting model training...")
            temp_model = train_model(all_data, all_labels)
            if temp_model:
                model = temp_model # Update the main model variable
                save_model(model, 'static/isl_gesture_model.pkl') # Save the newly trained model
                print("Model training complete and saved.")
            else:
                 print("Model training failed.")

        elif choice == '3':
            if not model:
                print("Model not trained or loaded.")
                if all_data and len(available_gestures) >= 2:
                     retrain_now = input("Train model now with available data? (y/n): ").lower()
                     if retrain_now == 'y':
                          print("Starting model training...")
                          temp_model = train_model(all_data, all_labels)
                          if temp_model:
                              model = temp_model
                              save_model(model, 'static/isl_gesture_model.pkl')
                              print("Model training complete and saved.")
                          else:
                               print("Model training failed. Cannot start recognition.")
                               continue # Back to menu
                     else:
                         print("Please train the model first (Option 2).")
                         continue # Back to menu
                else:
                     print("Insufficient data to train. Please collect more data (Option 1).")
                     continue # Back to menu


            if model: # Check again in case training just finished
                start_recognition(model) # Pass the trained/loaded model
            else:
                 print("Cannot start recognition without a valid model.")


        elif choice == '4':
             if not all_labels:
                 print("No data collected yet.")
             else:
                 print("\n--- Collected Data Summary ---")
                 unique_labels_summary, counts_summary = np.unique(all_labels, return_counts=True)
                 print(f"Total Samples: {len(all_labels)}")
                 print(f"Gestures ({len(unique_labels_summary)}):")
                 for label, count in zip(unique_labels_summary, counts_summary):
                     print(f"  - '{label}': {count} samples")
                 print(f"Expected feature vector size: {FEATURE_VECTOR_SIZE}")


        elif choice == '5':
             confirm = input("WARNING: This will delete saved data ("+DATA_FILENAME+") and model ("+MODEL_FILENAME+").\nAre you sure? (yes/no): ").lower()
             if confirm == 'yes':
                 deleted_files = []
                 try:
                     if os.path.exists(DATA_FILENAME):
                         os.remove(DATA_FILENAME)
                         deleted_files.append(DATA_FILENAME)
                     if os.path.exists(MODEL_FILENAME):
                         os.remove(MODEL_FILENAME)
                         deleted_files.append(MODEL_FILENAME)

                     if deleted_files:
                          print(f"Deleted: {', '.join(deleted_files)}")
                     else:
                          print("No files found to delete.")

                     # Reset runtime variables
                     all_data, all_labels = [], []
                     available_gestures = []
                     model = None
                     print("Runtime data and model cleared.")

                 except OSError as e:
                     print(f"Error deleting files: {e}")
             else:
                 print("Operation cancelled.")

        elif choice == 'q':
            print("Exiting program.")
            break

        else:
            print("Invalid choice. Please try again.")

# --- Real-time Recognition ---

def start_recognition(model):
    """Starts the real-time gesture recognition loop."""
    camera_idx = select_camera()
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_idx}.")
        return

    print("\nStarting Real-time Recognition...")
    print("Press 'q' in the window to stop.")
    time.sleep(1)

    window_name = 'ISL Gesture Recognition'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    #cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) # Optional
    cv2.resizeWindow(window_name, SCREEN_WIDTH, SCREEN_HEIGHT)
    cv2.moveWindow(window_name, 0, 0)

    # Get frame dimensions for scaling
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read initial frame.")
        cap.release()
        cv2.destroyAllWindows()
        return
    frame_h, frame_w = frame.shape[:2]

    # Calculate display scaling and offsets (same as collection)
    scale_w = SCREEN_WIDTH / frame_w if frame_w > 0 else 1
    scale_h = SCREEN_HEIGHT / frame_h if frame_h > 0 else 1
    scale = min(scale_w, scale_h)
    disp_w = int(frame_w * scale)
    disp_h = int(frame_h * scale)
    offset_x = (SCREEN_WIDTH - disp_w) // 2
    offset_y = (SCREEN_HEIGHT - disp_h) // 2

    # Info panel config (same as collection)
    panel_width = 400
    panel_start_x = disp_w - panel_width if disp_w > panel_width else 0
    panel_start_x_screen = offset_x + panel_start_x
    panel_end_x_screen = offset_x + disp_w
    panel_start_y_screen = offset_y
    panel_end_y_screen = offset_y + disp_h
    text_x_start = panel_start_x_screen + 20
    text_panel_width_screen = panel_end_x_screen - text_x_start - 20

    # Recognition variables
    last_high_conf_prediction = "None"
    last_confidence = 0.0
    display_prediction = "None" # What is shown on screen

    confidence_threshold = 0.99 # Only show prediction if confidence is >= this
    smoothing_buffer = [] # Optional: for temporal smoothing
    buffer_size = 5       # Optional: number of frames for smoothing


    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Dropped frame during recognition.")
            continue

        # --- Landmark Extraction ---
        left_landmarks, right_landmarks, _, hands_detected = extract_landmarks(frame)

        # --- Prepare Display Canvas ---
        display_canvas = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
        resized_frame = cv2.resize(frame, (disp_w, disp_h))
        display_canvas[offset_y:offset_y+disp_h, offset_x:offset_x+disp_w] = resized_frame

        # --- Draw Landmarks on Display ---
        display_frame_slice = display_canvas[offset_y:offset_y+disp_h, offset_x:offset_x+disp_w]
        display_slice_copy = display_frame_slice.copy()
        display_slice_rgb = cv2.cvtColor(display_slice_copy, cv2.COLOR_BGR2RGB)
        display_slice_rgb.flags.writeable = False
        results_display = hands.process(display_slice_rgb)
        display_slice_rgb.flags.writeable = True
        if results_display.multi_hand_landmarks:
            for hand_landmarks_disp in results_display.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    display_frame_slice,
                    hand_landmarks_disp,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        # --- Information Panel Overlay ---
        overlay = display_canvas.copy()
        cv2.rectangle(overlay, (panel_start_x_screen, panel_start_y_screen),
                      (panel_end_x_screen, panel_end_y_screen), (0, 0, 0), -1)
        alpha = 0.6
        display_canvas = cv2.addWeighted(overlay, alpha, display_canvas, 1 - alpha, 0)

        # --- Prediction Logic ---
        feature_vector = None
        current_raw_prediction = "None"
        current_confidence = 0.0

        # Construct feature vector based on detected hands (matching training format)
        # --- Corrected: Use global constant ---
        if all(hands_detected):
            feature_vector = np.concatenate([left_landmarks, right_landmarks])
        elif hands_detected[0]:
            feature_vector = np.concatenate([left_landmarks, np.zeros(SINGLE_HAND_FEATURE_SIZE)])
        elif hands_detected[1]:
            feature_vector = np.concatenate([np.zeros(SINGLE_HAND_FEATURE_SIZE), right_landmarks])


        if feature_vector is not None:
            # Ensure correct shape and size before predicting
            if feature_vector.shape[0] == FEATURE_VECTOR_SIZE:
                feature_vector_reshaped = feature_vector.reshape(1, -1)
                try:
                    # Check if model supports predict_proba
                    if hasattr(model, 'predict_proba'):
                         probabilities = model.predict_proba(feature_vector_reshaped)[0]
                         max_prob_index = np.argmax(probabilities)
                         current_raw_prediction = model.classes_[max_prob_index]
                         current_confidence = probabilities[max_prob_index]
                    else: # Fallback if no probability (e.g., standard LinearSVC)
                         current_raw_prediction = model.predict(feature_vector_reshaped)[0]
                         current_confidence = 1.0 # Assign full confidence as it's a direct prediction

                except Exception as e:
                    print(f"Error during prediction: {e}")
                    current_raw_prediction = "Error"
                    current_confidence = 0.0
            else:
                print(f"Error: Feature vector size mismatch in recognition! Expected {FEATURE_VECTOR_SIZE}, got {feature_vector.shape[0]}")
                current_raw_prediction = "Size Error"
                current_confidence = 0.0


        # --- Filtering / Smoothing (Optional) ---
        # Method 1: Simple Thresholding
        if current_confidence >= confidence_threshold:
            display_prediction = current_raw_prediction
            last_confidence = current_confidence
            last_high_conf_prediction = current_raw_prediction # Store the last good one
        #elif any(hands_detected): # Hand detected, but low confidence
             # Option: Keep showing the last high-confidence prediction
             #display_prediction = f"{last_high_conf_prediction}?" # Add '?' to indicate uncertainty
             #last_confidence = current_confidence # Show the low confidence value
             # Option: Show 'Uncertain' or 'None'
             # display_prediction = "Uncertain"
             # last_confidence = current_confidence
        else: # No hands detected
            display_prediction = "None"
            last_confidence = 0.0
            last_high_conf_prediction = "None"


        # --- Display Information ---
        y_pos = panel_start_y_screen + 50
        line_height = 40
        text_color = (0, 255, 0)

        def draw_text(text, y, size=0.8, color=text_color, bold=False):
            thickness = 2 if bold else 1
            try:
                cv2.putText(display_canvas, text, (text_x_start, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness, cv2.LINE_AA)
            except Exception as e:
                print(f"Error drawing text '{text}': {e}")
            return y + line_height

        y_pos = draw_text('ISL GESTURE RECOGNITION', y_pos, size=1.0, bold=True)
        cv2.line(display_canvas, (text_x_start, y_pos - (line_height // 2)), (panel_end_x_screen - 20, y_pos - (line_height // 2)), text_color, 1)

        pred_color = (0,255,0) if last_confidence >= confidence_threshold and display_prediction not in ["Error", "Size Error", "None"] else (0,255,255) # Green for high conf, Yellow for low/uncertain
        if "?" in display_prediction: pred_color = (0, 255, 255) # Yellow for uncertain
        if display_prediction in ["Error", "Size Error", "None"]: pred_color = (0,0,255) # Red for errors/none

        y_pos = draw_text(f'Prediction: {display_prediction}', y_pos, size=1.1, color=pred_color, bold=True)
        y_pos = draw_text(f'Confidence: {last_confidence:.2f}', y_pos, size=0.7, color=pred_color)
        y_pos += line_height // 2


        y_pos = draw_text('HAND STATUS:', y_pos, bold=True)
        cv2.line(display_canvas, (text_x_start, y_pos - (line_height // 2)), (panel_end_x_screen - 20, y_pos - (line_height // 2)), text_color, 1)
        status_color_left = (0, 255, 0) if hands_detected[0] else (0, 0, 255)
        status_color_right = (0, 255, 0) if hands_detected[1] else (0, 0, 255)
        y_pos = draw_text(f'Left Hand: {"Detected" if hands_detected[0] else "Not Detected"}', y_pos, color=status_color_left)
        y_pos = draw_text(f'Right Hand: {"Detected" if hands_detected[1] else "Not Detected"}', y_pos, color=status_color_right)
        y_pos += line_height // 2

        y_pos = draw_text('CONTROLS:', y_pos, bold=True)
        cv2.line(display_canvas, (text_x_start, y_pos - (line_height // 2)), (panel_end_x_screen - 20, y_pos - (line_height // 2)), text_color, 1)
        y_pos = draw_text('Press "q" to Quit Window', y_pos, size=0.7)


        # --- Show Frame ---
        cv2.imshow(window_name, display_canvas)

        # --- Handle Keypress ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Stopping recognition window.")
            break
        # elif key == ord('f'): # Fullscreen toggle
        #      current_mode = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
        #      cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
        #                            cv2.WINDOW_NORMAL if current_mode == cv2.WINDOW_FULLSCREEN else cv2.WINDOW_FULLSCREEN)

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()


# --- Entry Point ---
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n--- An Unexpected Error Occurred ---")
        import traceback
        traceback.print_exc()
        print("------------------------------------")
    finally:
        # Ensure mediapipe hands resources are released if an error occurs mid-operation
        if 'hands' in globals() and hasattr(hands, 'close'):
             hands.close()
        print("\nProgram finished.")