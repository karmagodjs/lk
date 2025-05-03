from flask import Flask, render_template_string, request, jsonify, send_from_directory
import cv2
import numpy as np
import base64
import io
from PIL import Image
import threading
from flask_cors import CORS
import os

# Import recognition logic and model loading from isl_recognition.py
from isl_recognition import extract_landmarks, load_model, save_model, save_data, load_data, train_model, SINGLE_HAND_FEATURE_SIZE, FEATURE_VECTOR_SIZE

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Runtime data
model = load_model('static/isl_gesture_model.pkl')
print("Model loaded:", model)
all_data, all_labels = load_data('static/isl_gesture_data.pkl')
available_gestures = sorted(list(np.unique(all_labels))) if all_labels else []

# --- Temp buffer for real-time collection ---
collect_buffer = {'data': [], 'labels': [], 'target': None, 'needed': 0}

# HTML template for the web page
HTML_PAGE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gesture Scribe - Modern Sign Language Translator</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary-color: #7c3aed; /* Violet 600 */
            --primary-light: #a78bfa; /* Violet 400 */
            --primary-lighter: #ede9fe; /* Violet 100 */
            --primary-dark: #6d28d9; /* Violet 700 */

            --danger-color: #ef4444; /* Red 500 */
            --danger-light: #fee2e2; /* Red 100 */
            --danger-dark: #dc2626; /* Red 600 */

            --success-color: #22c55e; /* Green 500 */

            --text-dark: #1f2937; /* Gray 800 */
            --text-medium: #6b7280; /* Gray 500 */
            --text-light: #9ca3af; /* Gray 400 */
            --text-on-primary: #ffffff;
            --text-on-danger: #ffffff;

            --bg-main: #f9fafb; /* Gray 50 */
            --bg-card: #ffffff;
            --bg-nav-active: var(--primary-lighter);
            --bg-nav-hover: #f3f4f6; /* Gray 100 */
            --bg-input: #ffffff;
            --bg-disabled: #e5e7eb; /* Gray 200 */
            --text-disabled: #9ca3af; /* Gray 400 */

            --border-color: #e5e7eb; /* Gray 200 */
            --border-focus: var(--primary-light);

            --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -4px rgba(0, 0, 0, 0.1);
            --shadow-inner: inset 0 2px 4px 0 rgba(0,0,0,0.05);

            --radius-sm: 4px;
            --radius-md: 8px;
            --radius-lg: 16px;

            --font-main: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol";

            --transition-speed: 0.2s;
            --transition-func: ease-in-out;
        }

        *, *::before, *::after {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: var(--font-main);
            background-color: var(--bg-main);
            color: var(--text-dark);
            line-height: 1.6;
        }

        .header {
            text-align: center;
            margin: 40px 0 24px 0;
        }

        .header h1 {
            color: var(--primary-color);
            font-size: 2.8em;
            margin-bottom: 0.3em;
            font-weight: 700;
            letter-spacing: -0.02em;
        }

        .header p {
            color: var(--text-medium);
            font-size: 1.2em;
            margin-top: 0;
            max-width: 500px;
            margin-left: auto;
            margin-right: auto;
        }

        .nav {
            display: flex;
            justify-content: center;
            gap: 8px;
            margin-bottom: 32px;
            border-bottom: 1px solid var(--border-color);
            padding: 0 16px;
        }

        .nav-btn {
            background: transparent;
            border: none;
            border-bottom: 3px solid transparent; /* Indicator line */
            border-radius: var(--radius-md) var(--radius-md) 0 0;
            padding: 14px 28px;
            font-size: 1.1em;
            color: var(--text-medium);
            cursor: pointer;
            font-weight: 500;
            transition: background-color var(--transition-speed) var(--transition-func),
                        color var(--transition-speed) var(--transition-func),
                        border-color var(--transition-speed) var(--transition-func);
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: -1px; /* Overlap border */
        }

        .nav-btn:hover {
            background: var(--bg-nav-hover);
            color: var(--text-dark);
        }
        .nav-btn.active {
            color: var(--primary-color);
            font-weight: 600;
            border-bottom-color: var(--primary-color);
        }
        .nav-btn.active, .nav-btn.active:hover {
             background: transparent; /* Keep bg transparent when active */
        }


        main {
            display: flex;
            flex-direction: column; /* Stack video and tabs */
            align-items: center;
            gap: 32px;
            padding: 0 16px; /* Add some horizontal padding */
        }

        .video-container { /* Renamed from video-card for clarity */
            background: var(--bg-card);
            border-radius: var(--radius-lg);
            box-shadow: var(--shadow-md);
            padding: 24px;
            display: flex;
            flex-direction: column;
            align-items: center;
            width: 100%;
            max-width: 520px; /* Control max width */
        }

        #video {
            border-radius: var(--radius-md);
            display: block; /* Remove extra space below video */
            max-width: 100%; /* Ensure video scales */
            height: auto;
            background: #111;
        }

        .tab-content-wrapper {
            width: 100%;
            max-width: 520px; /* Match video container or adjust */
            position: relative; /* Needed for absolute positioning of tabs if used */
        }

        .tab-content {
            background: var(--bg-card);
            border-radius: var(--radius-lg);
            box-shadow: var(--shadow-md);
            padding: 32px 36px;
            width: 100%;
            opacity: 0;
            visibility: hidden;
            position: absolute; /* Stack tabs for fade */
            top: 0;
            left: 0;
            transition: opacity var(--transition-speed) var(--transition-func),
                        visibility var(--transition-speed) var(--transition-func);
            /* Removed transform for simpler fade */
        }

        .tab-content.active-tab {
            opacity: 1;
            visibility: visible;
            position: relative; /* Take up space in layout */
        }


        .card h2, .result-title { /* Shared styles */
            color: var(--primary-color);
            margin-top: 0;
            margin-bottom: 24px;
            font-size: 1.6em;
            font-weight: 600;
        }

        .input-group { margin-bottom: 20px; }

        label {
            display: block;
            font-weight: 500;
            margin-bottom: 8px;
            color: var(--text-medium);
            font-size: 0.95em;
        }

        input[type="text"], input[type="number"], input[type="range"] {
            width: 100%;
            padding: 12px 14px;
            border-radius: var(--radius-md);
            border: 1px solid var(--border-color);
            font-size: 1em;
            background-color: var(--bg-input);
            transition: border-color var(--transition-speed) var(--transition-func),
                        box-shadow var(--transition-speed) var(--transition-func);
        }
        input[type="text"]:focus, input[type="number"]:focus, input[type="range"]:focus {
            outline: none;
            border-color: var(--border-focus);
            box-shadow: 0 0 0 2px var(--primary-lighter);
        }
        /* Specific style adjustments for range */
        input[type="range"] {
             padding: 0; /* Remove padding for range */
             height: 8px; /* Consistent height */
             cursor: pointer;
             appearance: none; /* Override default look */
             background: var(--primary-lighter);
             border-radius: var(--radius-md);
             outline: none;
        }
        /* Style range thumb */
        input[type="range"]::-webkit-slider-thumb {
            appearance: none;
            width: 20px;
            height: 20px;
            background: var(--primary-color);
            border-radius: 50%;
            cursor: pointer;
            transition: background-color var(--transition-speed) var(--transition-func);
        }
        input[type="range"]::-moz-range-thumb {
            width: 20px;
            height: 20px;
            background: var(--primary-color);
            border-radius: 50%;
            cursor: pointer;
            border: none;
            transition: background-color var(--transition-speed) var(--transition-func);
        }
         input[type="range"]:hover::-webkit-slider-thumb { background: var(--primary-dark); }
         input[type="range"]:hover::-moz-range-thumb { background: var(--primary-dark); }


        .slider-group { display: flex; align-items: center; gap: 16px; }
        #numSamplesValue { font-weight: 600; color: var(--primary-color); min-width: 3ch; text-align: right;}


        .btn { /* Base Button Style */
             border: none;
             border-radius: var(--radius-md);
             padding: 14px 24px;
             font-size: 1.1em;
             font-weight: 600;
             width: 100%;
             margin-top: 18px;
             cursor: pointer;
             transition: background-color var(--transition-speed) var(--transition-func),
                         transform var(--transition-speed) var(--transition-func),
                         box-shadow var(--transition-speed) var(--transition-func);
             display: inline-flex; /* Align icon and text */
             align-items: center;
             justify-content: center;
             gap: 8px;
        }
        .btn:hover {
            transform: translateY(-1px);
            box-shadow: var(--shadow-sm);
        }
        .btn:active {
             transform: translateY(0px);
             box-shadow: none;
        }
        .btn:disabled {
            background-color: var(--bg-disabled);
            color: var(--text-disabled);
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        .btn-primary {
             background: var(--primary-light);
             color: var(--text-on-primary);
        }
        .btn-primary:not(:disabled):hover { background: var(--primary-color); }
        .btn-primary:not(:disabled):active { background: var(--primary-dark); }

        .btn-danger {
             background: var(--danger-color);
             color: var(--text-on-danger);
        }
        .btn-danger:not(:disabled):hover { background: var(--danger-dark); }
        .btn-danger:not(:disabled):active { background: #b91c1c; } /* Red 700 */


        /* Collection specific */
        #modal-status, #modal-progress {
             margin-top: 16px;
             font-weight: 500;
             color: var(--text-medium);
             min-height: 1.5em; /* Prevent layout jump */
        }
        #modal-status { font-size: 1.1em; color: var(--text-dark); }
        #progress-bar-bg {
             background: var(--primary-lighter);
             border-radius: var(--radius-md);
             height: 10px; /* Thinner bar */
             margin-top: 8px;
             overflow: hidden; /* Contain inner bar */
        }
        #progress-bar {
             height: 100%;
             background: var(--primary-light);
             border-radius: var(--radius-md); /* Match parent */
             width: 0%;
             transition: width 0.3s ease-out;
        }
        #hand-status span {
            display: inline-block;
            margin-top: 12px;
            padding: 4px 10px;
            border-radius: var(--radius-sm);
            color: white;
            font-size: 0.9em;
            font-weight: 500;
        }


        /* Train Section */
         #train-summary {
             margin-bottom: 24px;
             background-color: var(--primary-lighter);
             padding: 16px;
             border-radius: var(--radius-md);
             color: var(--primary-dark);
             font-size: 0.95em;
         }
        #train-summary h3 { margin-top: 0; margin-bottom: 10px; color: var(--primary-color);}
        #train-summary .bar { background: white; } /* Adjust bar bg */

        /* Recognize Section */
        .recognize-flex {
             display: flex;
             gap: 32px; /* Spacing between video preview (if added back) and results */
             justify-content: center;
             align-items: flex-start;
             margin-bottom: 24px;
        }
        .result-card {
            /* Uses .card base styles, add specifics if needed */
            padding: 32px 36px;
            flex-grow: 1; /* Allow it to take space */
             background: var(--bg-card); /* Ensure bg */
             border-radius: var(--radius-lg); /* Ensure radius */
             box-shadow: var(--shadow-md); /* Ensure shadow */
             min-width: 300px; /* Minimum width */
        }
        .result-title { /* Already styled above */ }
        #prediction, #confidence {
            font-size: 1.1em;
            color: var(--text-medium);
            margin-top: 8px;
        }
         #prediction span, #confidence span {
            font-weight: 600;
            color: var(--primary-color);
            margin-left: 5px;
         }
        .recognize-btns {
            display: flex;
            flex-direction: column; /* Stack buttons */
            align-items: center;
            gap: 12px; /* Space between buttons */
            margin-top: 24px;
        }
        .recognize-btns .btn { margin-top: 0; } /* Remove top margin when in this group */


        /* Summary Section */
        #summary { font-size: 0.95em; }
        #summary h3 { color: var(--primary-color); margin-bottom: 16px; }
        .summary-box { /* For potential future use if summary gets complex */
            display: flex;
            gap: 24px;
            margin-bottom: 24px;
            flex-wrap: wrap;
        }
        .summary-item {
            background: var(--primary-lighter);
            color: var(--primary-color);
            border-radius: var(--radius-md);
            padding: 16px 24px;
            font-size: 1.3em;
            font-weight: 700;
            text-align: center;
            flex: 1; /* Distribute space */
            min-width: 120px; /* Prevent shrinking too much */
        }
        .summary-label {
            font-size: 0.8em;
            color: var(--text-medium);
            font-weight: 400;
            margin-top: 4px;
            display: block;
        }
        .bar { /* Shared bar style */
            background: var(--primary-lighter);
            border-radius: var(--radius-md);
            height: 18px;
            margin: 8px 0 16px 0;
            position: relative;
            overflow: hidden;
        }
        .bar-inner {
            background: var(--primary-light);
            height: 100%;
            border-radius: var(--radius-md);
            position: absolute;
            left: 0;
            top: 0;
            transition: width 0.5s var(--transition-func); /* Animate width changes */
        }
        .bar-label {
            position: absolute;
            left: 12px;
            top: 0;
            color: var(--primary-dark); /* Darker text on light bar */
            font-weight: 600;
            font-size: 0.9em;
            line-height: 18px; /* Match bar height */
            white-space: nowrap;
        }
        .bar-count {
            position: absolute;
            right: 12px;
            top: 0;
            color: var(--primary-dark);
            font-size: 0.9em;
            line-height: 18px; /* Match bar height */
            font-weight: 500;
        }

        /* Clear Section */
         #clear-section p {
             margin-bottom: 20px;
             color: var(--text-medium);
         }

        /* Notification */
        #notification {
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background-color: var(--primary-color);
            color: white;
            padding: 12px 24px;
            border-radius: var(--radius-md);
            box-shadow: var(--shadow-lg);
            z-index: 1000;
            font-weight: 500;
            opacity: 0;
            visibility: hidden;
            transition: opacity 0.3s ease-in-out, visibility 0.3s ease-in-out, background-color 0.3s ease-in-out;
        }
         #notification.show {
             opacity: 1;
             visibility: visible;
         }


        /* Spinner Overlay */
        #spinner {
            position: fixed;
            inset: 0;
            background-color: rgba(255, 255, 255, 0.7);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1001;
            opacity: 0;
            visibility: hidden;
             transition: opacity 0.2s ease-in-out, visibility 0.2s ease-in-out;
        }
         #spinner.show {
            opacity: 1;
            visibility: visible;
         }

        /* Simple CSS Spinner */
        .lds-dual-ring {
            display: inline-block;
            width: 80px;
            height: 80px;
        }
        .lds-dual-ring:after {
            content: " ";
            display: block;
            width: 64px;
            height: 64px;
            margin: 8px;
            border-radius: 50%;
            border: 6px solid var(--primary-color);
            border-color: var(--primary-color) transparent var(--primary-color) transparent;
            animation: lds-dual-ring 1.2s linear infinite;
        }
        @keyframes lds-dual-ring {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }


        /* Responsiveness */
        @media (max-width: 768px) {
             .header h1 { font-size: 2.2em; }
             .header p { font-size: 1.1em; padding: 0 10px;}
             .nav { gap: 2px; overflow-x: auto; padding: 0 8px; justify-content: flex-start;}
             .nav-btn { padding: 12px 16px; font-size: 1em;}

             main { gap: 24px; padding: 0 8px;}
             .video-container, .tab-content-wrapper { max-width: 100%; }
             .tab-content { padding: 24px 16px; }
             .card h2, .result-title { font-size: 1.4em; margin-bottom: 16px;}

             .recognize-flex { flex-direction: column; align-items: stretch; gap: 16px; }
             .result-card { min-width: unset; padding: 24px 16px;}

             .btn { padding: 12px 18px; font-size: 1em;}
             #notification { width: 90%; bottom: 10px; text-align: center;}
        }

    </style>
</head>
<body>
    <header class="header">
        <h1>Gesture Scribe</h1>
        <p>Convert sign language to text with machine learning</p>
    </header>

    <nav class="nav" role="tablist">
        <button class="nav-btn active" id="tab-collect" onclick="showTab('collect')" role="tab" aria-selected="true" aria-controls="collect-section">
             Collect Data
        </button>
        <button class="nav-btn" id="tab-train" onclick="showTab('train')" role="tab" aria-selected="false" aria-controls="train-section">
            ↻ Train Model
        </button>
        <button class="nav-btn" id="tab-recognize" onclick="showTab('recognize')" role="tab" aria-selected="false" aria-controls="recognize-section">
            ▶ Recognize
        </button>
        <button class="nav-btn" id="tab-summary" onclick="showTab('summary')" role="tab" aria-selected="false" aria-controls="summary-section">
             Data Summary
        </button>
        <button class="nav-btn" id="tab-clear" onclick="showTab('clear')" role="tab" aria-selected="false" aria-controls="clear-section">
             Clear All
        </button>
    </nav>

    <main>
        <!-- Shared Video Section -->
        <div class="video-container" id="video-section">
            <video id="video" width="480" height="360" autoplay muted playsinline></video>
             <!-- Added playsinline for better mobile -->
        </div>

        <!-- Tab Content Area -->
        <div class="tab-content-wrapper">
            <section class="card tab-content active-tab" id="collect-section" role="tabpanel" aria-labelledby="tab-collect">
                <h2>Collect Gesture Data</h2>
                <div class="input-group">
                    <label for="gestureName">Gesture Name</label>
                    <input type="text" id="gestureName" placeholder="e.g., Hello, Thank You, Yes, No">
                </div>
                <div class="input-group">
                    <label for="numSamples">Number of samples (<span id="numSamplesValue">30</span>)</label>
                    <div class="slider-group">
                        <input type="range" id="numSamples" min="10" max="200" value="30" oninput="document.getElementById('numSamplesValue').textContent = this.value">
                        <!-- <span id="numSamplesValue">30</span> --> <!-- Moved label span above -->
                    </div>
                </div>
                <button class="btn btn-primary" id="btnStartCollection" onclick="startRealCollection()">
                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="12" cy="12" r="3"></circle><line x1="16.5" x2="16.5" y1="7.5" y2="7.501"></line></svg>
                    Start Collection
                </button>
                <div id="modal-status" aria-live="polite"></div>
                <div id="modal-progress" aria-live="polite"></div>
                <div id="progress-bar-bg" aria-hidden="true">
                    <div id="progress-bar" style="width:0%;"></div>
                </div>
                <div id="hand-status" aria-live="polite"></div>
            </section>

            <section class="card tab-content" id="train-section" style="position: absolute;" role="tabpanel" aria-labelledby="tab-train">
                <h2>Model Training</h2>
                <div id="train-summary">Loading summary...</div>
                <button class="btn btn-primary" onclick="trainModel()">
                     <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"></polyline><polyline points="1 20 1 14 7 14"></polyline><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path></svg>
                     Start Training
                </button>
            </section>

            <section class="tab-content" id="recognize-section" style="position: absolute;" role="tabpanel" aria-labelledby="tab-recognize">
                <!-- Video is now displayed globally above -->
                 <div class="recognize-flex">
                    <!-- Removed nested video card -->
                    <div class="result-card">
                        <h2 class="result-title">Real-time Recognition</h2>
                        <div id="prediction" aria-live="polite">Prediction: <span id="pred">None*</span></div>
                        <div id="confidence">Confidence: <span id="conf">0.00</span></div>
                    </div>
                 </div>
                 <div class="recognize-btns">
                    <button class="btn btn-primary" onclick="startRecognition()">
                        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
                        Start Recognition
                    </button>
                    <button class="btn btn-primary" onclick="stopRecognition()" style="background-color: var(--text-medium);"> <!-- Secondary button style -->
                        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18"></rect></svg>
                         Stop Recognition
                    </button>
                </div>
            </section>

            <section class="card tab-content" id="summary-section" style="position: absolute;" role="tabpanel" aria-labelledby="tab-summary">
                <h2>Data Summary</h2>
                <div id="summary">Loading summary...</div>
            </section>

            <section class="card tab-content" id="clear-section" style="position: absolute;" role="tabpanel" aria-labelledby="tab-clear">
                <h2>Clear All Data and Model</h2>
                <p>This will permanently delete all collected gesture data and the trained model weights.</p>
                <button class="btn btn-danger" onclick="clearData()">
                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path><line x1="10" y1="11" x2="10" y2="17"></line><line x1="14" y1="11" x2="14" y2="17"></line></svg>
                    Clear All Data
                </button>
            </section>
        </div>
    </main>

    <div id="notification" aria-live="assertive"></div>
    <div id="spinner"><div class="lds-dual-ring"></div></div>

    <script>
        // Tab switching logic (Updated for class-based animation)
        function showTab(tabId) {
            const tabs = ['collect', 'train', 'recognize', 'summary', 'clear'];
            const tabContentWrapper = document.querySelector('.tab-content-wrapper');

            tabs.forEach(t => {
                const section = document.getElementById(`${t}-section`);
                const button = document.getElementById(`tab-${t}`);
                const isActive = t === tabId;

                section.classList.toggle('active-tab', isActive);
                button.classList.toggle('active', isActive);
                button.setAttribute('aria-selected', isActive);

                // Adjust wrapper height dynamically - Optional but good for layout
                if (isActive) {
                     // Use setTimeout to allow fade-out before height change
                     setTimeout(() => {
                         //tabContentWrapper.style.height = section.offsetHeight + 'px'; // Causes jumpiness, maybe better without
                         // Make sure the active tab is not positioned absolutely
                         section.style.position = 'relative';
                     }, parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--transition-speed') || '0.2') * 1000);

                } else {
                    // Reset inactive tabs to absolute position for stacking/fade
                    section.style.position = 'absolute';
                }
            });

            // Trigger backend updates if needed
            if(tabId === 'summary') viewSummary();
            if(tabId === 'train') updateTrainSummary();

            // Handle video visibility based on tab (if needed - currently always visible)
            // const videoSection = document.getElementById('video-section');
            // if (tabId === 'collect' || tabId === 'recognize') {
            //    videoSection.style.display = '';
            // } else {
            //    videoSection.style.display = 'none'; // Or adjust as needed
            // }
        }

        // --- Existing JS Logic (Mostly Unchanged) ---
        let video = document.getElementById('video');
        let predSpan = document.getElementById('pred');
        let confSpan = document.getElementById('conf');
        let running = false;
        let intervalId = null;
        let summaryDiv = document.getElementById('summary');
        let modalStatus = document.getElementById('modal-status');
        let modalProgress = document.getElementById('modal-progress');
        let progressBar = document.getElementById('progress-bar');
        let handStatus = document.getElementById('hand-status');
        let btnStartCollection = document.getElementById('btnStartCollection');
        let collecting = false;
        let collectCount = 0;
        let collectTarget = 0;
        let collectGesture = '';
        let collectInterval = null;
        let countdownTimer = null;
        let lastHandDetected = false;
        let failedCollects = 0;
        let notification = document.getElementById('notification');
        let spinner = document.getElementById('spinner');
        let notificationTimeout = null;


        function showNotification(msg, type = 'info', timeout = 3000) {
            // Clear any existing timeout
             if (notificationTimeout) clearTimeout(notificationTimeout);

             notification.textContent = msg;
             notification.classList.remove('success', 'error', 'info'); // Remove old types

             // Apply new type class for styling (optional, but good practice)
             let bgColor;
             switch(type) {
                 case 'success':
                     bgColor = getComputedStyle(document.documentElement).getPropertyValue('--success-color');
                     break;
                 case 'error':
                     bgColor = getComputedStyle(document.documentElement).getPropertyValue('--danger-color');
                     break;
                 case 'info':
                 default:
                     bgColor = getComputedStyle(document.documentElement).getPropertyValue('--primary-color');
                     break;
             }
             notification.style.backgroundColor = bgColor;

             notification.classList.add('show'); // Trigger fade-in

             // Set timeout to hide notification
             notificationTimeout = setTimeout(() => {
                 notification.classList.remove('show'); // Trigger fade-out
             }, timeout);
        }


        function showSpinner(show) {
            spinner.classList.toggle('show', show);
        }

        function setMenuEnabled(enabled) {
             // Disable all major action buttons during processing
             document.querySelectorAll('.btn-primary, .btn-danger, .nav-btn').forEach(btn => {
                 // Don't disable the stop button during recognition
                 if (btn.textContent.includes('Stop Recognition') && running) {
                     // Keep stop button enabled
                 } else {
                    btn.disabled = !enabled;
                 }
             });
             // Re-enable specific buttons based on context if needed,
             // but generally disabling all during long operations is safer.
             // Ensure range input is also disabled/enabled
             document.getElementById('numSamples').disabled = !enabled;
             document.getElementById('gestureName').disabled = !enabled;
        }

        function startRecognition() {
            if (running) return;
            running = true;
            setMenuEnabled(false); // Disable buttons except stop
            document.querySelector('.recognize-btns button:nth-child(2)').disabled = false; // Ensure stop is enabled
            intervalId = setInterval(captureAndSend, 200); // 5 FPS
            showNotification('Recognition started', 'info');
            predSpan.textContent = 'Starting...';
            confSpan.textContent = '0.00';
        }

        function stopRecognition() {
            if (!running) return;
            running = false;
            if (intervalId) clearInterval(intervalId);
            setMenuEnabled(true); // Re-enable all buttons
            showNotification('Recognition stopped', 'info');
             predSpan.textContent = 'Stopped';
             confSpan.textContent = '0.00';
        }


        function captureAndSend() {
             if (!running || !video.srcObject || video.paused || video.ended || video.readyState < 3) {
                 console.warn("Recognition frame skipped: Video not ready.");
                 return; // Ensure video is ready
             }
             let canvas = document.createElement('canvas');
             // Use naturalWidth/Height for potentially non-rendered dimensions
             canvas.width = video.videoWidth || video.width;
             canvas.height = video.videoHeight || video.height;
             if (canvas.width === 0 || canvas.height === 0) {
                 console.warn("Recognition frame skipped: Video dimensions are zero.");
                 return; // Cannot draw if dimensions are 0
             }

             let ctx = canvas.getContext('2d');
             try {
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                let dataURL = canvas.toDataURL('image/jpeg', 0.8); // Quality 0.8
                fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: dataURL })
                })
                .then(response => {
                    if (!response.ok) { throw new Error(`HTTP error! status: ${response.status}`); }
                    return response.json();
                 })
                .then(data => {
                    // Display prediction even if low confidence, but style differently?
                    predSpan.textContent = data.prediction || 'None';
                    confSpan.textContent = (data.confidence !== undefined && data.confidence !== null) ? data.confidence.toFixed(2) : '0.00';

                    // Example: Visual cue for high confidence
                     if (data.confidence >= 0.90) {
                         predSpan.style.color = 'var(--success-color)';
                     } else if (data.confidence >= 0.70) {
                         predSpan.style.color = 'var(--primary-color)';
                     } else {
                        predSpan.style.color = 'var(--text-medium)';
                     }

                })
                .catch(err => {
                    console.error("Prediction error:", err);
                    predSpan.textContent = 'Error';
                    confSpan.textContent = '0.00';
                     predSpan.style.color = 'var(--danger-color)';
                    // Maybe stop recognition on repeated errors?
                    // stopRecognition();
                    // showNotification('Recognition stopped due to error.', 'error');
                });
            } catch (e) {
                console.error("Error drawing video to canvas:", e);
                 predSpan.textContent = 'Canvas Error';
                 confSpan.textContent = '0.00';
                 predSpan.style.color = 'var(--danger-color)';
            }
        }

        // --- Webcam Setup ---
         navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 } } }) // Request specific size
            .then(stream => {
                video.srcObject = stream;
                video.onloadedmetadata = () => {
                     console.log(`Webcam started: ${video.videoWidth}x${video.videoHeight}`);
                     // Adjust video element size if needed, though CSS max-width should handle it
                     // video.width = video.videoWidth;
                     // video.height = video.videoHeight;
                 };
            })
            .catch(err => {
                console.error('Could not access webcam:', err);
                alert('Could not access webcam: ' + err.message);
                // Display error message in the UI
                 modalStatus.textContent = 'Error: Could not access webcam. Please grant permission and refresh.';
                 modalStatus.style.color = 'var(--danger-color)';
                 setMenuEnabled(false); // Disable collection etc.
            });


        function startRealCollection() {
            let gesture = document.getElementById('gestureName').value.trim();
            let num = parseInt(document.getElementById('numSamples').value);
            if (!gesture) {
                showNotification('Please enter a gesture name.', 'error');
                document.getElementById('gestureName').focus();
                return;
            }
             if (isNaN(num) || num < 10) {
                showNotification('Please set number of samples (minimum 10).', 'error');
                 document.getElementById('numSamples').focus();
                return;
            }
            if (!video.srcObject) {
                 showNotification('Webcam not available.', 'error');
                 return;
            }

            setMenuEnabled(false); // Disable buttons/inputs
            collectCount = 0;
            collectTarget = num;
            collectGesture = gesture;
            collecting = false; // Will be set true after countdown
            failedCollects = 0;
            modalStatus.textContent = '';
            modalProgress.textContent = '';
            progressBar.style.width = '0%';
            handStatus.innerHTML = '';

            // Countdown before starting
            let countdown = 3;
            modalStatus.textContent = `Get ready... ${countdown}`;
            modalStatus.style.color = 'var(--primary-color)';
            countdownTimer = setInterval(() => {
                countdown--;
                if (countdown > 0) {
                    modalStatus.textContent = `Get ready... ${countdown}`;
                } else {
                    clearInterval(countdownTimer);
                    modalStatus.textContent = `Collecting "${collectGesture}"...`;
                    collecting = true;

                    // Tell backend to init buffer
                    fetch('/start_collection', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ gesture: collectGesture, num_samples: collectTarget }) // Send target num
                    })
                    .then(response => {
                         if (!response.ok) { throw new Error(`HTTP error! status: ${response.status}`); }
                         return response.json();
                     })
                    .then(data => {
                        if (data.success) {
                             modalStatus.textContent = `Collecting "${collectGesture}" (0/${collectTarget})...`;
                             collectInterval = setInterval(captureAndSendSample, 100); // 10 FPS collection
                        } else {
                            throw new Error(data.message || 'Failed to start collection on server.');
                        }
                    })
                     .catch(err => {
                         console.error("Start Collection Error:", err);
                         modalStatus.textContent = `Error starting: ${err.message}`;
                         modalStatus.style.color = 'var(--danger-color)';
                         showNotification(`Error starting collection: ${err.message}`, 'error', 5000);
                         setMenuEnabled(true);
                         collecting = false;
                     });
                }
            }, 1000);
        }


        function finishCollection(success = true, message = 'Collection complete!') {
             if (collectInterval) clearInterval(collectInterval);
             if (countdownTimer) clearInterval(countdownTimer); // Clear countdown if stopped early
             collecting = false;

             // Tell backend to finish and save, regardless of client-side completion status
             fetch('/finish_collection', { method: 'POST' })
                .then(r => r.json())
                .then(data => {
                     // Use the server's final status message if available
                     const finalMessage = data.status || message;
                     modalStatus.textContent = finalMessage;
                     modalStatus.style.color = success ? 'var(--success-color)' : 'var(--danger-color)';
                     modalProgress.textContent = `${collectCount} / ${collectTarget} samples gathered.`; // Final count
                     progressBar.style.width = '100%'; // Visually complete
                     handStatus.innerHTML = '';
                     setMenuEnabled(true);
                     showNotification(finalMessage, success ? 'success' : 'error');
                     updateTrainSummary(); // Update summary on train tab
                })
                .catch(err => {
                     console.error("Finish Collection Error:", err);
                     modalStatus.textContent = 'Error finalizing collection.';
                     modalStatus.style.color = 'var(--danger-color)';
                     setMenuEnabled(true);
                     showNotification('Error finalizing collection: ' + err, 'error', 4000);
                });
        }


         function captureAndSendSample() {
            if (!collecting) return; // Stop if flag is false
            // Check target reached client-side (safety check, backend should also limit)
            if (collectCount >= collectTarget) {
                 finishCollection(true, `Collected ${collectTarget} samples for "${collectGesture}".`);
                 return;
             }
             if (!video.srcObject || video.paused || video.ended || video.readyState < 3) {
                console.warn("Collection frame skipped: Video not ready.");
                failedCollects++; // Count as failure? Maybe not, just wait.
                handStatus.innerHTML = `<span style="background:var(--text-light);color:white;">Waiting for video...</span>`;
                return; // Ensure video is ready
             }

             let canvas = document.createElement('canvas');
             canvas.width = video.videoWidth || video.width;
             canvas.height = video.videoHeight || video.height;
              if (canvas.width === 0 || canvas.height === 0) {
                 console.warn("Collection frame skipped: Video dimensions are zero.");
                 failedCollects++;
                 handStatus.innerHTML = `<span style="background:var(--danger-color);">Video Error</span>`;
                 return; // Cannot draw if dimensions are 0
             }

             let ctx = canvas.getContext('2d');
             try {
                 ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                 let dataURL = canvas.toDataURL('image/jpeg', 0.8); // Quality 0.8

                 fetch('/collect_sample', {
                     method: 'POST',
                     headers: { 'Content-Type': 'application/json' },
                     body: JSON.stringify({ image: dataURL, gesture: collectGesture }) // Send gesture name with each sample
                 })
                 .then(response => {
                     if (!response.ok) { throw new Error(`HTTP error! status: ${response.status}`); }
                     return response.json();
                  })
                 .then(data => {
                     if (!collecting) return; // Check again, might have been stopped

                     if (data.success) {
                         collectCount++;
                         failedCollects = 0; // Reset fail counter on success
                         modalStatus.textContent = `Collecting "${collectGesture}" (${collectCount}/${collectTarget})...`;
                         modalProgress.textContent = ''; // Clear detailed progress, status line has count
                         let percent = Math.min(100, Math.round(100 * collectCount / collectTarget));
                         progressBar.style.width = percent + '%';
                         lastHandDetected = true;
                         handStatus.innerHTML = `<span style="background:var(--success-color);">Hand Detected</span>`;

                         // Check if target is met AFTER incrementing
                         if (collectCount >= collectTarget) {
                             finishCollection(true, `Collected ${collectTarget} samples for "${collectGesture}".`);
                         }
                     } else {
                         // Handle specific backend errors if provided
                         if (data.error === 'Sample limit reached') {
                              finishCollection(true, 'Sample collection limit reached on server.');
                              return;
                         } else if (data.error === 'No hand detected') {
                             failedCollects++;
                             lastHandDetected = false;
                             handStatus.innerHTML = `<span style="background:var(--danger-color);">No Hand Detected</span>`;
                         } else {
                             // Generic failure from backend
                             failedCollects++;
                             lastHandDetected = false;
                             handStatus.innerHTML = `<span style="background:var(--danger-color);">Detection Failed</span>`;
                             console.warn("Sample collection failed:", data.message || 'Unknown reason');
                         }

                         // Stop if too many consecutive failures
                         if (failedCollects >= 25) { // Increased threshold slightly
                            finishCollection(false, 'Collection stopped: No hand detected for too long.');
                         }
                     }
                 })
                 .catch(err => {
                     console.error("Collect Sample Error:", err);
                     if (collecting) { // Only stop if still in collecting state
                         finishCollection(false, `Collection error: ${err.message}`);
                     }
                 });
            } catch(e) {
                console.error("Error drawing video to canvas during collection:", e);
                failedCollects++;
                handStatus.innerHTML = `<span style="background:var(--danger-color);">Canvas Error</span>`;
                 if (failedCollects >= 10) { // Stop quicker on canvas errors
                    finishCollection(false, 'Collection stopped due to repeated canvas errors.');
                 }
            }
         }


        function trainModel() {
            showSpinner(true);
            setMenuEnabled(false);
            showNotification('Starting model training... this may take a while.', 'info', 5000);
            fetch('/train_model', { method: 'POST' })
            .then(response => {
                 if (!response.ok) { throw new Error(`HTTP error! status: ${response.status}`); }
                 return response.json();
              })
            .then(data => {
                showSpinner(false);
                setMenuEnabled(true);
                const message = data.status || 'Training request processed.';
                const success = message.toLowerCase().includes('trained') || message.toLowerCase().includes('success');
                showNotification(message, success ? 'success' : 'error', success ? 5000 : 7000);
                updateTrainSummary(); // Refresh summary after training attempt
            })
            .catch(err => {
                 console.error("Train Model Error:", err);
                 showSpinner(false);
                 setMenuEnabled(true);
                 showNotification(`Training failed: ${err.message}`, 'error', 7000);
                 updateTrainSummary(); // Still refresh summary
            });
        }

        function viewSummary() {
            summaryDiv.innerHTML = 'Loading summary...'; // Show loading state
            fetch('/data_summary')
             .then(response => {
                 if (!response.ok) { throw new Error(`HTTP error! status: ${response.status}`); }
                 return response.json();
              })
            .then(data => {
                if (data.summary) {
                    summaryDiv.innerHTML = data.summary; // Expecting HTML summary from backend
                } else {
                     summaryDiv.innerHTML = '<p>No data collected yet or error fetching summary.</p>';
                }
            })
            .catch(err => {
                 console.error("View Summary Error:", err);
                 summaryDiv.innerHTML = `<p style="color:var(--danger-color)">Error loading summary: ${err.message}</p>`;
            });
        }

        function updateTrainSummary() {
            const trainSummaryDiv = document.getElementById('train-summary');
            trainSummaryDiv.innerHTML = 'Loading summary...'; // Show loading state
             fetch('/data_summary') // Reuse the same endpoint
            .then(response => {
                 if (!response.ok) { throw new Error(`HTTP error! status: ${response.status}`); }
                 return response.json();
              })
            .then(data => {
                 if (data.summary) {
                     // Modify summary slightly for training context if needed
                     trainSummaryDiv.innerHTML = '<h3>Current Data Status</h3>' + data.summary + '<p style="margin-top:15px; font-size:0.9em; color: var(--text-medium);">Click "Start Training" to train a model on this data.</p>';
                 } else {
                     trainSummaryDiv.innerHTML = '<h3>Current Data Status</h3><p>No data collected yet.</p>';
                 }
            })
            .catch(err => {
                 console.error("Update Train Summary Error:", err);
                 trainSummaryDiv.innerHTML = `<h3 style="color:var(--danger-color)">Error loading summary</h3><p>${err.message}</p>`;
            });
        }

        function clearData() {
            if (!confirm('Are you sure you want to permanently clear all collected data and the trained model? This action cannot be undone.')) return;
            showSpinner(true);
            setMenuEnabled(false);
            fetch('/clear_data', { method: 'POST' })
             .then(response => {
                 if (!response.ok) { throw new Error(`HTTP error! status: ${response.status}`); }
                 return response.json();
              })
            .then(data => {
                showSpinner(false);
                setMenuEnabled(true);
                showNotification(data.status || 'Data cleared.', 'info'); // Use info for clear action
                summaryDiv.innerHTML = ''; // Clear summary view
                updateTrainSummary(); // Update train summary view
                predSpan.textContent = 'None*'; // Reset prediction state
                confSpan.textContent = '0.00';
                if (running) stopRecognition(); // Stop recognition if running
            })
             .catch(err => {
                 console.error("Clear Data Error:", err);
                 showSpinner(false);
                 setMenuEnabled(true);
                 showNotification(`Error clearing data: ${err.message}`, 'error', 5000);
            });
        }

        // Set default tab on load
        document.addEventListener('DOMContentLoaded', () => {
             showTab('collect');
        });

    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if not model:
        return jsonify({'prediction': 'No Model', 'confidence': 0.0})
    try:
        data = request.get_json()
        img_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(img_data)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # Extract landmarks
        left_landmarks, right_landmarks, _, hands_detected = extract_landmarks(frame)
        feature_vector = None
        if all(hands_detected):
            feature_vector = np.concatenate([left_landmarks, right_landmarks])
        elif hands_detected[0]:
            feature_vector = np.concatenate([left_landmarks, np.zeros(SINGLE_HAND_FEATURE_SIZE)])
        elif hands_detected[1]:
            feature_vector = np.concatenate([np.zeros(SINGLE_HAND_FEATURE_SIZE), right_landmarks])
        if feature_vector is not None and feature_vector.shape[0] == FEATURE_VECTOR_SIZE:
            feature_vector_reshaped = feature_vector.reshape(1, -1)
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(feature_vector_reshaped)[0]
                max_prob_index = np.argmax(probabilities)
                prediction = model.classes_[max_prob_index]
                confidence = float(probabilities[max_prob_index])
            else:
                prediction = model.predict(feature_vector_reshaped)[0]
                confidence = 1.0
        else:
            prediction = 'None'
            confidence = 0.0
        return jsonify({'prediction': str(prediction), 'confidence': confidence})
    except Exception as e:
        return jsonify({'prediction': 'Error', 'confidence': 0.0, 'error': str(e)})

# --- Real gesture collection endpoints ---
@app.route('/start_collection', methods=['POST'])
def start_collection():
    global collect_buffer
    data = request.get_json()
    gesture = data.get('gesture')
    num_samples = int(data.get('num_samples', 100))
    test_mode = request.args.get('test', '0') == '1'
    collect_buffer = {'data': [], 'labels': [], 'target': gesture, 'needed': num_samples, 'test': test_mode}
    if test_mode:
        print('[start_collection] TEST MODE ENABLED')
    return jsonify({'success': True})

@app.route('/collect_sample', methods=['POST'])
def collect_sample():
    global collect_buffer
    data = request.get_json()
    gesture = data.get('gesture')
    test_mode = collect_buffer.get('test', False)
    # --- FIX: Prevent collecting more than needed ---
    if len(collect_buffer['data']) >= collect_buffer['needed']:
        print(f"[collect_sample] Sample limit reached for gesture '{gesture}'. Ignoring extra sample.")
        return jsonify({'success': False, 'error': 'Sample limit reached'})
    img_data = data['image'].split(',')[1]
    img_bytes = base64.b64decode(img_data)
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    frame = np.array(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    left_landmarks, right_landmarks, _, hands_detected = extract_landmarks(frame)
    feature_vector = None
    print(f"[collect_sample] Received sample for gesture '{gesture}'. Hands detected: {hands_detected}")
    if test_mode:
        feature_vector = np.random.rand(FEATURE_VECTOR_SIZE)
        print('[collect_sample] TEST MODE: Accepting random sample.')
    else:
        if all(hands_detected):
            feature_vector = np.concatenate([left_landmarks, right_landmarks])
        elif hands_detected[0]:
            feature_vector = np.concatenate([left_landmarks, np.zeros(SINGLE_HAND_FEATURE_SIZE)])
        elif hands_detected[1]:
            feature_vector = np.concatenate([np.zeros(SINGLE_HAND_FEATURE_SIZE), right_landmarks])
    if feature_vector is not None and feature_vector.shape[0] == FEATURE_VECTOR_SIZE:
        collect_buffer['data'].append(feature_vector)
        collect_buffer['labels'].append(gesture)
        print(f"[collect_sample] Sample accepted. Total for this gesture: {len(collect_buffer['data'])}")
        return jsonify({'success': True})
    else:
        print(f"[collect_sample] Sample rejected (no hand detected or wrong size).")
        return jsonify({'success': False})

@app.route('/finish_collection', methods=['POST'])
def finish_collection():
    global all_data, all_labels, available_gestures, collect_buffer
    test_mode = collect_buffer.get('test', False)
    all_data.extend(collect_buffer['data'])
    all_labels.extend(collect_buffer['labels'])
    available_gestures = sorted(list(set(all_labels)))
    save_data(all_data, all_labels, 'static/isl_gesture_data.pkl')
    count = len(collect_buffer['data'])
    gesture = collect_buffer['target']
    collect_buffer = {'data': [], 'labels': [], 'target': None, 'needed': 0, 'test': False}
    if test_mode:
        print('[finish_collection] TEST MODE: Finished collection.')
    return jsonify({'status': f'Added {count} real samples for gesture "{gesture}".'})

@app.route('/train_model', methods=['POST'])
def train_model_route():
    global model, all_data, all_labels
    if not all_data or not all_labels or len(set(all_labels)) < 2:
        return jsonify({'status': 'Need at least 2 gestures with data to train.'})
    model = train_model(all_data, all_labels)
    if model:
        save_model(model, 'static/isl_gesture_model.pkl')
        return jsonify({'status': 'Model trained and saved.'})
    else:
        return jsonify({'status': 'Model training failed.'})

@app.route('/data_summary')
def data_summary():
    global all_labels
    if not all_labels:
        return jsonify({'summary': 'No data collected yet.'})
    unique, counts = np.unique(all_labels, return_counts=True)
    summary = f"<ul>" + ''.join([f'<li>{u}: {c} samples</li>' for u, c in zip(unique, counts)]) + f"</ul>"
    summary += f"<p>Total samples: {len(all_labels)}</p>"
    return jsonify({'summary': summary})

@app.route('/clear_data', methods=['POST'])
def clear_data():
    global all_data, all_labels, model, available_gestures
    all_data.clear()
    all_labels.clear()
    available_gestures.clear()
    model = None
    import os
    try:
        if os.path.exists('isl_gesture_data.pkl'):
            os.remove('isl_gesture_data.pkl')
        if os.path.exists('isl_gesture_model.pkl'):
            os.remove('isl_gesture_model.pkl')
    except Exception:
        pass
    return jsonify({'status': 'All data and model cleared.'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 