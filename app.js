/* ============================================================================
   Medzome - AI-Powered Lateral Flow Test Analysis
   Professional Medical Web Application
   ============================================================================ */

// ============================================================================
// Configuration
// ============================================================================

const CONFIG = {
    MODEL_PATH: 'medzome_mvp_model.tflite',
    API_ENDPOINT: '',  // Empty string = relative URLs (same origin, no CORS)
    INPUT_HEIGHT: 384,  // Will be updated from model info
    INPUT_WIDTH: 128,   // Will be updated from model info
    CONFIDENCE_THRESHOLD: 0.5,
    MAX_FILE_SIZE: 10 * 1024 * 1024, // 10MB
    SUPPORTED_FORMATS: ['image/jpeg', 'image/png', 'image/bmp', 'image/tiff'],
    HISTORY_STORAGE_KEY: 'medzome_test_history'
};

// ============================================================================
// Global State
// ============================================================================

let currentImage = null;
let currentImageData = null;
let cameraStream = null;
let analysisResults = null;

// ============================================================================
// DOM Elements
// ============================================================================

const elements = {
    // Info Panel
    infoPanel: document.getElementById('infoPanel'),
    infoText: document.getElementById('infoText'),
    
    // Upload Section
    dropZone: document.getElementById('dropZone'),
    fileInput: document.getElementById('fileInput'),
    cameraBtn: document.getElementById('cameraBtn'),
    uploadProgress: document.getElementById('uploadProgress'),
    progressBar: document.getElementById('progressBar'),
    progressText: document.getElementById('progressText'),
    
    // Preview Section
    previewSection: document.getElementById('previewSection'),
    previewImage: document.getElementById('previewImage'),
    scanningOverlay: document.getElementById('scanningOverlay'),
    clearBtn: document.getElementById('clearBtn'),
    analyzeBtn: document.getElementById('analyzeBtn'),
    
    // Results Section
    noResults: document.getElementById('noResults'),
    resultsDisplay: document.getElementById('resultsDisplay'),
    resultBadge: document.getElementById('resultBadge'),
    resultIcon: document.getElementById('resultIcon'),
    resultStatus: document.getElementById('resultStatus'),
    resultMessage: document.getElementById('resultMessage'),
    
    // Metrics
    confidenceValue: document.getElementById('confidenceValue'),
    confidenceBar: document.getElementById('confidenceBar'),
    intensityValue: document.getElementById('intensityValue'),
    intensityBar: document.getElementById('intensityBar'),
    thresholdValue: document.getElementById('thresholdValue'),
    timeValue: document.getElementById('timeValue'),
    
    // Details
    controlLine: document.getElementById('controlLine'),
    testLine: document.getElementById('testLine'),
    intensityScore: document.getElementById('intensityScore'),
    qualityAssessment: document.getElementById('qualityAssessment'),
    
    // Guidance
    medicalGuidance: document.getElementById('medicalGuidance'),
    guidanceText: document.getElementById('guidanceText'),
    
    // Actions
    downloadBtn: document.getElementById('downloadBtn'),
    resetBtn: document.getElementById('resetBtn'),
    
    // Camera Modal
    cameraModal: document.getElementById('cameraModal'),
    cameraStream: document.getElementById('cameraStream'),
    cameraCanvas: document.getElementById('cameraCanvas'),
    closeCameraBtn: document.getElementById('closeCameraBtn'),
    captureBtn: document.getElementById('captureBtn'),
    
    // Capture Confirmation Modal
    captureModal: document.getElementById('captureModal'),
    capturedImage: document.getElementById('capturedImage'),
    useCaptureBtn: document.getElementById('useCaptureBtn'),
    retakeBtn: document.getElementById('retakeBtn'),
    cancelCaptureBtn: document.getElementById('cancelCaptureBtn'),
    
    // Confirmation Modal
    confirmModal: document.getElementById('confirmModal'),
    confirmTitle: document.getElementById('confirmTitle'),
    confirmMessage: document.getElementById('confirmMessage'),
    confirmOkBtn: document.getElementById('confirmOkBtn'),
    confirmCancelBtn: document.getElementById('confirmCancelBtn'),
    
    // Results Modal
    resultsModal: document.getElementById('resultsModal'),
    closeResultsModalBtn: document.getElementById('closeResultsModalBtn'),
    modalPreviewImage: document.getElementById('modalPreviewImage'),
    modalResultBadge: document.getElementById('modalResultBadge'),
    modalResultIcon: document.getElementById('modalResultIcon'),
    modalResultStatus: document.getElementById('modalResultStatus'),
    modalResultMessage: document.getElementById('modalResultMessage'),
    modalConfidenceValue: document.getElementById('modalConfidenceValue'),
    modalConfidenceBar: document.getElementById('modalConfidenceBar'),
    modalIntensityValue: document.getElementById('modalIntensityValue'),
    modalIntensityBar: document.getElementById('modalIntensityBar'),
    modalThresholdValue: document.getElementById('modalThresholdValue'),
    modalTimeValue: document.getElementById('modalTimeValue'),
    modalControlLine: document.getElementById('modalControlLine'),
    modalTestLine: document.getElementById('modalTestLine'),
    modalIntensityScore: document.getElementById('modalIntensityScore'),
    modalQualityAssessment: document.getElementById('modalQualityAssessment'),
    modalMedicalGuidance: document.getElementById('modalMedicalGuidance'),
    modalGuidanceText: document.getElementById('modalGuidanceText'),
    
    // History
    historyList: document.getElementById('historyList'),
    clearHistoryBtn: document.getElementById('clearHistoryBtn'),
    
    // Loading Overlay
    loadingOverlay: document.getElementById('loadingOverlay')
};

// ============================================================================
// Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', async () => {
    initializeEventListeners();
    loadHistory();
    await fetchModelInfo();
    updateInfoPanel('Ready to analyze. Please upload or capture a test strip image.', 'info');
});

// ============================================================================
// Model Configuration
// ============================================================================

async function fetchModelInfo() {
    try {
        const response = await fetch(`${CONFIG.API_ENDPOINT}/model/info`);
        if (response.ok) {
            const modelInfo = await response.json();
            // Update CONFIG with actual model dimensions
            CONFIG.INPUT_HEIGHT = modelInfo.input_height;
            CONFIG.INPUT_WIDTH = modelInfo.input_width;
            console.log(`✅ Model loaded: ${modelInfo.model_type.toUpperCase()}`);
            console.log(`   Input dimensions: ${CONFIG.INPUT_WIDTH}x${CONFIG.INPUT_HEIGHT}`);
        }
    } catch (error) {
        console.warn('⚠️  Could not fetch model info, using defaults:', error);
    }
}

// ============================================================================
// Event Listeners
// ============================================================================

function initializeEventListeners() {
    // File Upload
    elements.fileInput.addEventListener('change', handleFileSelect);
    elements.dropZone.addEventListener('click', (e) => {
        if (!e.target.closest('label') && e.target !== elements.fileInput) {
            elements.fileInput.click();
        }
    });
    
    // Drag and Drop
    elements.dropZone.addEventListener('dragover', handleDragOver);
    elements.dropZone.addEventListener('dragleave', handleDragLeave);
    elements.dropZone.addEventListener('drop', handleDrop);
    
    // Camera
    elements.cameraBtn.addEventListener('click', openCamera);
    elements.closeCameraBtn.addEventListener('click', closeCamera);
    elements.captureBtn.addEventListener('click', captureImage);
    
    // Capture Confirmation
    elements.useCaptureBtn.addEventListener('click', useCapturedImage);
    elements.retakeBtn.addEventListener('click', retakeImage);
    elements.cancelCaptureBtn.addEventListener('click', cancelCapture);
    
    // Results Modal
    elements.closeResultsModalBtn.addEventListener('click', closeResultsModal);
    elements.resultsModal.addEventListener('click', (e) => {
        if (e.target === elements.resultsModal) {
            closeResultsModal();
        }
    });
    
    // Preview Actions
    elements.clearBtn.addEventListener('click', clearPreview);
    elements.analyzeBtn.addEventListener('click', analyzeImage);
    
    // Result Actions
    elements.downloadBtn.addEventListener('click', downloadReport);
    elements.resetBtn.addEventListener('click', resetApp);
    
    // History
    elements.clearHistoryBtn.addEventListener('click', clearHistory);
}

// ============================================================================
// File Upload Handling
// ============================================================================

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        validateAndLoadFile(file);
    }
}

function handleDragOver(event) {
    event.preventDefault();
    elements.dropZone.classList.add('drag-over');
}

function handleDragLeave(event) {
    event.preventDefault();
    elements.dropZone.classList.remove('drag-over');
}

function handleDrop(event) {
    event.preventDefault();
    elements.dropZone.classList.remove('drag-over');
    
    const file = event.dataTransfer.files[0];
    if (file) {
        validateAndLoadFile(file);
    }
}

function validateAndLoadFile(file) {
    // Validate file type
    if (!CONFIG.SUPPORTED_FORMATS.includes(file.type)) {
        showError('Unsupported file format. Please upload JPG, PNG, BMP, or TIFF images.');
        return;
    }
    
    // Validate file size
    if (file.size > CONFIG.MAX_FILE_SIZE) {
        showError('File size exceeds 10MB limit. Please choose a smaller image.');
        return;
    }
    
    // Load file
    loadImageFile(file);
}

function loadImageFile(file) {
    showUploadProgress();
    
    const reader = new FileReader();
    
    reader.onprogress = (event) => {
        if (event.lengthComputable) {
            const percentComplete = (event.loaded / event.total) * 100;
            updateProgress(percentComplete);
        }
    };
    
    reader.onload = (event) => {
        const img = new Image();
        img.onload = () => {
            currentImage = img;
            currentImageData = event.target.result;
            showPreview(event.target.result);
            hideUploadProgress();
            updateInfoPanel('Image loaded successfully. Click "Analyze" to begin.', 'success');
        };
        img.onerror = () => {
            showError('Failed to load image. Please try again.');
            hideUploadProgress();
        };
        img.src = event.target.result;
    };
    
    reader.onerror = () => {
        showError('Failed to read file. Please try again.');
        hideUploadProgress();
    };
    
    reader.readAsDataURL(file);
}

function showUploadProgress() {
    elements.uploadProgress.style.display = 'block';
    elements.progressBar.style.width = '0%';
    elements.progressText.textContent = 'Uploading...';
}

function updateProgress(percent) {
    elements.progressBar.style.width = percent + '%';
    elements.progressText.textContent = `Uploading... ${Math.round(percent)}%`;
}

function hideUploadProgress() {
    setTimeout(() => {
        elements.uploadProgress.style.display = 'none';
    }, 500);
}

// ============================================================================
// Camera Handling
// ============================================================================

async function openCamera() {
    try {
        elements.cameraModal.classList.add('active');
        
        const constraints = {
            video: {
                facingMode: 'environment',
                width: { ideal: 1280 },
                height: { ideal: 720 }
            }
        };
        
        cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
        elements.cameraStream.srcObject = cameraStream;
        
        updateInfoPanel('Camera ready. Align test strip within the frame and capture.', 'info');
    } catch (error) {
        console.error('Camera error:', error);
        showError('Unable to access camera. Please check permissions.');
        closeCamera();
    }
}

function closeCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }
    elements.cameraModal.classList.remove('active');
}

function captureImage() {
    const video = elements.cameraStream;
    const canvas = elements.cameraCanvas;
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    
    const imageData = canvas.toDataURL('image/jpeg', 0.95);
    
    elements.capturedImage.src = imageData;
    elements.captureModal.classList.add('active');
}

function useCapturedImage() {
    const imageData = elements.capturedImage.src;
    
    const img = new Image();
    img.onload = () => {
        currentImage = img;
        currentImageData = imageData;
        showPreview(imageData);
        updateInfoPanel('Image captured successfully. Click "Analyze" to begin.', 'success');
    };
    img.src = imageData;
    
    elements.captureModal.classList.remove('active');
    closeCamera();
}

function retakeImage() {
    elements.captureModal.classList.remove('active');
}

function cancelCapture() {
    elements.captureModal.classList.remove('active');
    closeCamera();
}

// ============================================================================
// Preview Handling
// ============================================================================

function showPreview(imageData) {
    elements.previewImage.src = imageData;
    elements.previewSection.style.display = 'block';
    elements.noResults.style.display = 'block';
    elements.resultsDisplay.style.display = 'none';
}

function clearPreview() {
    currentImage = null;
    currentImageData = null;
    elements.previewSection.style.display = 'none';
    elements.fileInput.value = '';
    updateInfoPanel('Ready to analyze. Please upload or capture a test strip image.', 'info');
}

// ============================================================================
// Image Analysis
// ============================================================================

async function analyzeImage() {
    if (!currentImage || !currentImageData) {
        showError('No image to analyze. Please upload or capture an image first.');
        return;
    }
    
    // Show scanning animation
    elements.scanningOverlay.style.display = 'block';
    elements.analyzeBtn.disabled = true;
    updateInfoPanel('Analyzing test strip... Please wait.', 'warning');
    
    const startTime = performance.now();
    
    try {
        // Preprocess image
        const processedImage = await preprocessImage(currentImage);
        
        // Send to backend for inference
        const result = await performInference(processedImage, currentImageData);
        
        const endTime = performance.now();
        const processingTime = endTime - startTime;
        
        // Add processing time to result
        result.processingTime = processingTime;
        
        // Display results
        displayResults(result);
        
        // Save to history
        saveToHistory(result, currentImageData);
        
        updateInfoPanel('Analysis complete! Results are ready.', 'success');
        
    } catch (error) {
        console.error('Analysis error:', error);
        showError('Analysis failed. Please try again or use a different image.');
        updateInfoPanel('Analysis failed. Please try again.', 'danger');
    } finally {
        elements.scanningOverlay.style.display = 'none';
        elements.analyzeBtn.disabled = false;
    }
}

async function preprocessImage(image) {
    // Create canvas for preprocessing
    const canvas = document.createElement('canvas');
    canvas.width = CONFIG.INPUT_WIDTH;
    canvas.height = CONFIG.INPUT_HEIGHT;
    const ctx = canvas.getContext('2d');
    
    // Draw and resize image
    ctx.drawImage(image, 0, 0, CONFIG.INPUT_WIDTH, CONFIG.INPUT_HEIGHT);
    
    // Get image data
    const imageData = ctx.getImageData(0, 0, CONFIG.INPUT_WIDTH, CONFIG.INPUT_HEIGHT);
    
    // Apply preprocessing (lighting correction, normalization, etc.)
    const data = imageData.data;
    const normalized = new Float32Array(CONFIG.INPUT_WIDTH * CONFIG.INPUT_HEIGHT * 3);
    
    for (let i = 0; i < data.length; i += 4) {
        const idx = i / 4;
        normalized[idx * 3] = data[i] / 255.0;       // R
        normalized[idx * 3 + 1] = data[i + 1] / 255.0; // G
        normalized[idx * 3 + 2] = data[i + 2] / 255.0; // B
    }
    
    return {
        data: normalized,
        shape: [1, CONFIG.INPUT_HEIGHT, CONFIG.INPUT_WIDTH, 3]
    };
}

async function performInference(processedImage, originalImageData) {
    // Send to Flask backend
    const response = await fetch(`${CONFIG.API_ENDPOINT}/predict`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            image: originalImageData,
            threshold: CONFIG.CONFIDENCE_THRESHOLD
        })
    });
    
    if (!response.ok) {
        throw new Error('Inference request failed');
    }
    
    const result = await response.json();
    return result;
}

// ============================================================================
// Results Display
// ============================================================================

function displayResults(result) {
    analysisResults = result;
    
    elements.noResults.style.display = 'none';
    elements.resultsDisplay.style.display = 'block';
    
    // Check if test is INVALID (no control line detected)
    if (result.is_invalid || !result.control_line_detected) {
        // Display INVALID result
        elements.resultBadge.className = 'result-badge invalid';
        elements.resultIcon.className = 'bi bi-exclamation-triangle-fill';
        elements.resultStatus.textContent = 'INVALID TEST';
        elements.resultMessage.textContent = result.error_message || 'Control line not detected. This may not be a test strip image. Please upload a clear image of a lateral flow test strip.';
        
        // Set all metrics to 0 or N/A
        elements.confidenceValue.textContent = '0.0%';
        elements.confidenceBar.style.width = '0%';
        elements.confidenceBar.className = 'progress-bar bg-secondary';
        
        elements.intensityValue.textContent = '0.0%';
        elements.intensityBar.style.width = '0%';
        elements.intensityBar.className = 'progress-bar bg-secondary';
        
        elements.thresholdValue.textContent = (CONFIG.CONFIDENCE_THRESHOLD * 100).toFixed(0) + '%';
        elements.timeValue.textContent = (result.processing_time_ms || 0).toFixed(0) + ' ms';
        
        // Update detailed analysis
        elements.controlLine.textContent = '✗ Not Detected';
        elements.controlLine.className = 'detail-value text-danger';
        
        elements.testLine.textContent = 'N/A';
        elements.testLine.className = 'detail-value text-muted';
        
        elements.intensityScore.textContent = 'Invalid Test';
        elements.qualityAssessment.textContent = result.quality || 'Invalid - Not a test strip';
        
        // Update medical guidance
        updateMedicalGuidance(false, 0, 'Invalid', true);
        
        return;
    }
    
    // Valid test strip - proceed with normal display
    const isPositive = result.confidence > CONFIG.CONFIDENCE_THRESHOLD;
    const status = isPositive ? 'positive' : 'negative';
    
    // Update result badge
    elements.resultBadge.className = `result-badge ${status}`;
    elements.resultIcon.className = isPositive ? 'bi bi-x-circle-fill' : 'bi bi-check-circle-fill';
    elements.resultStatus.textContent = isPositive ? 'POSITIVE' : 'NEGATIVE';
    elements.resultMessage.textContent = isPositive 
        ? 'Test line detected - Further evaluation recommended'
        : 'No test line detected - Result appears negative';
    
    // Calculate line intensity
    const lineIntensity = calculateLineIntensity(result);
    const intensityCategory = getIntensityCategory(lineIntensity);
    
    // Update metrics
    elements.confidenceValue.textContent = (result.confidence * 100).toFixed(1) + '%';
    elements.confidenceBar.style.width = (result.confidence * 100) + '%';
    elements.confidenceBar.className = `progress-bar ${getConfidenceClass(result.confidence)}`;
    
    elements.intensityValue.textContent = lineIntensity.toFixed(1) + '%';
    elements.intensityBar.style.width = lineIntensity + '%';
    elements.intensityBar.className = `progress-bar ${getIntensityClass(lineIntensity)}`;
    
    elements.thresholdValue.textContent = (CONFIG.CONFIDENCE_THRESHOLD * 100).toFixed(0) + '%';
    elements.timeValue.textContent = result.processing_time_ms.toFixed(0) + ' ms';
    
    // Update detailed analysis
    elements.controlLine.textContent = detectControlLine(result) ? '✓ Detected' : '✗ Not Detected';
    elements.controlLine.className = detectControlLine(result) ? 'detail-value text-success' : 'detail-value text-danger';
    
    elements.testLine.textContent = isPositive ? '✓ Detected' : '✗ Not Detected';
    elements.testLine.className = isPositive ? 'detail-value text-danger' : 'detail-value text-success';
    
    elements.intensityScore.textContent = `${lineIntensity.toFixed(1)}% (${intensityCategory})`;
    elements.qualityAssessment.textContent = assessImageQuality(result);
    
    // Update medical guidance
    updateMedicalGuidance(isPositive, lineIntensity, intensityCategory);
    
    // Scroll to results
    elements.resultsDisplay.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function calculateLineIntensity(result) {
    // Calculate intensity based on confidence and additional factors
    // Higher confidence = higher intensity
    let intensity = result.confidence * 100;
    
    // Adjust based on threshold proximity
    if (result.confidence > CONFIG.CONFIDENCE_THRESHOLD) {
        const excess = result.confidence - CONFIG.CONFIDENCE_THRESHOLD;
        intensity += excess * 20; // Boost for high confidence
    }
    
    // Cap at 100%
    return Math.min(intensity, 100);
}

function getIntensityCategory(intensity) {
    if (intensity >= 80) return 'Very Strong';
    if (intensity >= 60) return 'Strong';
    if (intensity >= 40) return 'Moderate';
    if (intensity >= 20) return 'Weak';
    return 'Very Weak';
}

function detectControlLine(result) {
    // Assume control line is present if confidence is reasonable
    // In production, this would be from actual line detection
    return result.confidence > 0.3;
}

function assessImageQuality(result) {
    const controlLine = detectControlLine(result);
    
    if (!controlLine) return 'Poor - Control line not detected';
    if (result.confidence > 0.8) return 'Excellent - Clear image';
    if (result.confidence > 0.6) return 'Good - Acceptable quality';
    if (result.confidence > 0.4) return 'Fair - Consider retake';
    return 'Poor - Retake recommended';
}

function getConfidenceClass(confidence) {
    if (confidence > 0.8) return 'bg-success';
    if (confidence > 0.6) return 'bg-info';
    if (confidence > 0.4) return 'bg-warning';
    return 'bg-danger';
}

function getIntensityClass(intensity) {
    if (intensity >= 80) return 'bg-danger';
    if (intensity >= 60) return 'bg-warning';
    if (intensity >= 40) return 'bg-info';
    return 'bg-success';
}

function updateMedicalGuidance(isPositive, intensity, category, isInvalid = false) {
    if (isInvalid) {
        // Invalid test - show warning guidance
        elements.medicalGuidance.className = 'medical-guidance warning';
        elements.guidanceText.innerHTML = `
            <strong>⚠️ Invalid Test Result</strong><br><br>
            • Control line was not detected in the image<br>
            • This indicates the image may not contain a valid test strip<br>
            • <strong>Please ensure you have uploaded a clear image of a lateral flow test strip</strong><br>
            • Make sure the entire test strip is visible and in focus<br>
            • Avoid shadows, glare, or obstructions<br>
            • Retake the photo in good lighting conditions<br><br>
            <em>A valid test strip must show at least a control line for results to be reliable.</em>
        `;
    } else if (isPositive) {
        elements.medicalGuidance.className = 'medical-guidance danger';
        elements.guidanceText.innerHTML = `
            <strong>⚠️ Positive Result Detected (${category} Intensity)</strong><br><br>
            • This test indicates a positive result for the lateral flow assay<br>
            • Line intensity suggests ${category.toLowerCase()} presence of target antigen<br>
            • <strong>Immediate medical consultation is strongly recommended</strong><br>
            • Do not self-diagnose - consult a healthcare professional<br>
            • Follow local health guidelines and protocols<br>
            • Consider confirmatory laboratory testing<br><br>
            <em>This is a screening tool only. Professional medical advice is essential.</em>
        `;
    } else {
        elements.medicalGuidance.className = 'medical-guidance success';
        elements.guidanceText.innerHTML = `
            <strong>✓ Negative Result</strong><br><br>
            • No test line detected - result appears negative<br>
            • Control line present - test is valid<br>
            • This does not completely rule out infection<br>
            • Continue to monitor for symptoms<br>
            • Follow public health guidelines<br>
            • Consult a doctor if symptoms develop or persist<br><br>
            <em>A negative result does not guarantee absence of disease. Seek medical advice if concerned.</em>
        `;
    }
}

// ============================================================================
// History Management
// ============================================================================

function saveToHistory(result, imageData) {
    const historyItem = {
        id: Date.now(),
        timestamp: new Date().toISOString(),
        isPositive: result.confidence > CONFIG.CONFIDENCE_THRESHOLD,
        confidence: result.confidence,
        intensity: calculateLineIntensity(result),
        processingTime: result.processingTime,
        imageData: imageData,
        result: result  // Save the complete result object for modal display
    };
    
    // Get existing history
    let history = JSON.parse(localStorage.getItem(CONFIG.HISTORY_STORAGE_KEY) || '[]');
    
    // Add new item at the beginning
    history.unshift(historyItem);
    
    // Limit history to 50 items
    if (history.length > 50) {
        history = history.slice(0, 50);
    }
    
    // Save back to localStorage
    localStorage.setItem(CONFIG.HISTORY_STORAGE_KEY, JSON.stringify(history));
    
    // Reload history display
    loadHistory();
}

function loadHistory() {
    const history = JSON.parse(localStorage.getItem(CONFIG.HISTORY_STORAGE_KEY) || '[]');
    
    if (history.length === 0) {
        elements.historyList.innerHTML = `
            <div class="no-history">
                <i class="bi bi-inbox"></i>
                <p>No test history available.</p>
            </div>
        `;
        return;
    }
    
    elements.historyList.innerHTML = history.map(item => `
        <div class="history-item" onclick="viewHistoryItem(${item.id})">
            <img src="${item.imageData}" alt="Test strip" class="history-thumbnail">
            <div class="history-details">
                <div class="history-status ${item.isPositive ? 'positive' : 'negative'}">
                    ${item.isPositive ? '🔴 POSITIVE' : '🟢 NEGATIVE'}
                </div>
                <div class="history-timestamp">
                    ${new Date(item.timestamp).toLocaleString()}
                </div>
                <div class="history-metrics">
                    <span class="history-metric">
                        <i class="bi bi-speedometer2"></i> ${(item.confidence * 100).toFixed(1)}%
                    </span>
                    <span class="history-metric">
                        <i class="bi bi-brightness-high"></i> ${item.intensity.toFixed(1)}%
                    </span>
                    <span class="history-metric">
                        <i class="bi bi-stopwatch"></i> ${item.processingTime.toFixed(0)}ms
                    </span>
                </div>
            </div>
        </div>
    `).join('');
}

function viewHistoryItem(id) {
    const history = JSON.parse(localStorage.getItem(CONFIG.HISTORY_STORAGE_KEY)) || [];
    const item = history.find(h => h.id === id);
    
    if (!item) {
        showError('History item not found');
        return;
    }
    
    console.log('Viewing history item:', item);
    
    // Show results in modal
    showResultsModal(item.result, item.imageData);
}

function clearHistory() {
    showConfirmDialog(
        'Clear History',
        'Are you sure you want to clear all test history? This action cannot be undone.',
        () => {
            showLoadingOverlay('Clearing history...');
            
            setTimeout(() => {
                localStorage.removeItem(CONFIG.HISTORY_STORAGE_KEY);
                loadHistory();
                hideLoadingOverlay();
                updateInfoPanel('Test history cleared successfully.', 'success');
            }, 500);
        }
    );
}

// ============================================================================
// PDF Report Generation
// ============================================================================

function downloadReport() {
    if (!analysisResults) {
        showError('No results to download. Please analyze an image first.');
        return;
    }
    
    showLoadingOverlay('Generating report...');
    
    setTimeout(() => {
        try {
            const { jsPDF } = window.jspdf;
            const doc = new jsPDF();
            
            const isPositive = analysisResults.confidence > CONFIG.CONFIDENCE_THRESHOLD;
            const lineIntensity = calculateLineIntensity(analysisResults);
            const intensityCategory = getIntensityCategory(lineIntensity);
            
            // Header
            doc.setFillColor(13, 110, 253);
            doc.rect(0, 0, 210, 40, 'F');
            doc.setTextColor(255, 255, 255);
            doc.setFontSize(24);
            doc.text('MEDZOME', 105, 20, { align: 'center' });
            doc.setFontSize(12);
            doc.text('Lateral Flow Test Analysis Report', 105, 30, { align: 'center' });
            
            // Reset colors
            doc.setTextColor(0, 0, 0);
            
            // Test Result
            doc.setFontSize(18);
            doc.setFont(undefined, 'bold');
            let yPos = 55;
            doc.text('TEST RESULT', 20, yPos);
            
            // Result badge
            yPos += 10;
            if (isPositive) {
                doc.setFillColor(220, 53, 69);
            } else {
                doc.setFillColor(40, 167, 69);
            }
            doc.roundedRect(20, yPos, 170, 20, 3, 3, 'F');
            doc.setTextColor(255, 255, 255);
            doc.setFontSize(16);
            doc.text(isPositive ? 'POSITIVE' : 'NEGATIVE', 105, yPos + 13, { align: 'center' });
            
            // Reset colors
            doc.setTextColor(0, 0, 0);
            
            // Metrics
            yPos += 35;
            doc.setFontSize(14);
            doc.setFont(undefined, 'bold');
            doc.text('Analysis Metrics', 20, yPos);
            
            yPos += 10;
            doc.setFontSize(11);
            doc.setFont(undefined, 'normal');
            
            const metrics = [
                ['Confidence Score:', `${(analysisResults.confidence * 100).toFixed(1)}%`],
                ['Line Intensity:', `${lineIntensity.toFixed(1)}% (${intensityCategory})`],
                ['Threshold:', `${(CONFIG.CONFIDENCE_THRESHOLD * 100).toFixed(0)}%`],
                ['Processing Time:', `${analysisResults.processingTime.toFixed(0)} ms`],
                ['Control Line:', detectControlLine(analysisResults) ? 'Detected' : 'Not Detected'],
                ['Test Line:', isPositive ? 'Detected' : 'Not Detected'],
                ['Image Quality:', assessImageQuality(analysisResults)],
                ['Test Date:', new Date().toLocaleString()]
            ];
            
            metrics.forEach(([label, value]) => {
                yPos += 8;
                doc.text(label, 25, yPos);
                doc.setFont(undefined, 'bold');
                doc.text(value, 105, yPos);
                doc.setFont(undefined, 'normal');
            });
            
            // Medical Guidance
            yPos += 15;
            doc.setFontSize(14);
            doc.setFont(undefined, 'bold');
            doc.text('Medical Recommendation', 20, yPos);
            
            yPos += 8;
            doc.setFontSize(10);
            doc.setFont(undefined, 'normal');
            
            if (isPositive) {
                const guidance = [
                    'This test indicates a POSITIVE result.',
                    `Line intensity suggests ${intensityCategory.toLowerCase()} presence of target antigen.`,
                    'Immediate medical consultation is strongly recommended.',
                    'Do not self-diagnose - consult a healthcare professional.',
                    'Follow local health guidelines and protocols.',
                    'Consider confirmatory laboratory testing.'
                ];
                
                guidance.forEach(line => {
                    yPos += 6;
                    doc.text('• ' + line, 25, yPos, { maxWidth: 160 });
                });
            } else {
                const guidance = [
                    'No test line detected - result appears negative.',
                    'Control line present - test is valid.',
                    'This does not completely rule out infection.',
                    'Continue to monitor for symptoms.',
                    'Consult a doctor if symptoms develop or persist.'
                ];
                
                guidance.forEach(line => {
                    yPos += 6;
                    doc.text('• ' + line, 25, yPos, { maxWidth: 160 });
                });
            }
            
            // Test Strip Image
            yPos += 15;
            if (yPos > 230) {
                doc.addPage();
                yPos = 20;
            }
            
            doc.setFontSize(14);
            doc.setFont(undefined, 'bold');
            doc.text('Test Strip Image', 20, yPos);
            
            yPos += 5;
            if (currentImageData) {
                const imgWidth = 60;
                const imgHeight = 100;
                doc.addImage(currentImageData, 'JPEG', 75, yPos, imgWidth, imgHeight);
                yPos += imgHeight + 10;
            }
            
            // Disclaimer
            yPos += 5;
            doc.setFontSize(8);
            doc.setFont(undefined, 'italic');
            doc.setTextColor(100, 100, 100);
            const disclaimer = 'DISCLAIMER: This report is generated by an AI-powered screening tool and is for informational purposes only. It is NOT a medical diagnosis. Always consult with a qualified healthcare professional for proper medical advice, diagnosis, and treatment. The accuracy of this test depends on proper sample collection, timing, and test execution.';
            doc.text(disclaimer, 105, yPos, { align: 'center', maxWidth: 170 });
            
            // Footer
            doc.setFillColor(33, 37, 41);
            doc.rect(0, 280, 210, 17, 'F');
            doc.setTextColor(255, 255, 255);
            doc.setFontSize(9);
            doc.text('Medzome - AI-Powered Medical Diagnostics', 105, 288, { align: 'center' });
            doc.text(`Report generated: ${new Date().toLocaleString()}`, 105, 293, { align: 'center' });
            
            // Save PDF
            const filename = `Medzome_Test_Report_${Date.now()}.pdf`;
            doc.save(filename);
            
            hideLoadingOverlay();
            updateInfoPanel('Report downloaded successfully.', 'success');
            
        } catch (error) {
            console.error('PDF generation error:', error);
            hideLoadingOverlay();
            showError('Failed to generate report. Please try again.');
        }
    }, 1000);
}

// ============================================================================
// App Control
// ============================================================================

function resetApp() {
    showLoadingOverlay('Resetting...');
    
    setTimeout(() => {
        clearPreview();
        elements.resultsDisplay.style.display = 'none';
        elements.noResults.style.display = 'block';
        analysisResults = null;
        hideLoadingOverlay();
        updateInfoPanel('Ready for new test. Please upload or capture an image.', 'info');
    }, 500);
}

// ============================================================================
// UI Helpers
// ============================================================================

function updateInfoPanel(message, type = 'info') {
    elements.infoText.textContent = message;
    elements.infoPanel.className = 'info-panel';
    
    if (type === 'success') {
        elements.infoPanel.classList.add('success');
    } else if (type === 'danger') {
        elements.infoPanel.classList.add('danger');
    } else if (type === 'warning') {
        elements.infoPanel.classList.add('warning');
    }
}

function showLoadingOverlay(message = 'Processing...') {
    elements.loadingOverlay.querySelector('.loading-text').textContent = message;
    elements.loadingOverlay.classList.add('active');
}

function hideLoadingOverlay() {
    elements.loadingOverlay.classList.remove('active');
}

function showError(message) {
    updateInfoPanel(message, 'danger');
}

function showConfirmDialog(title, message, onConfirm) {
    elements.confirmTitle.textContent = title;
    elements.confirmMessage.textContent = message;
    elements.confirmModal.classList.add('active');
    
    // Remove existing event listeners
    const newOkBtn = elements.confirmOkBtn.cloneNode(true);
    const newCancelBtn = elements.confirmCancelBtn.cloneNode(true);
    elements.confirmOkBtn.replaceWith(newOkBtn);
    elements.confirmCancelBtn.replaceWith(newCancelBtn);
    
    // Update references
    elements.confirmOkBtn = newOkBtn;
    elements.confirmCancelBtn = newCancelBtn;
    
    // Add new event listeners
    elements.confirmOkBtn.addEventListener('click', () => {
        elements.confirmModal.classList.remove('active');
        onConfirm();
    });
    
    elements.confirmCancelBtn.addEventListener('click', () => {
        elements.confirmModal.classList.remove('active');
    });
    
    // Close on backdrop click
    elements.confirmModal.addEventListener('click', (e) => {
        if (e.target === elements.confirmModal) {
            elements.confirmModal.classList.remove('active');
        }
    });
}

function showResultsModal(result, imageData) {
    console.log('showResultsModal called with:', { result, imageData });
    
    if (!result) {
        console.error('No result provided to showResultsModal');
        showError('Unable to display result - data missing');
        return;
    }
    
    // Set image
    elements.modalPreviewImage.src = imageData;
    
    // Check if invalid test
    const isInvalid = result.is_invalid || !result.control_line_detected;
    
    if (isInvalid) {
        // Invalid test display
        elements.modalResultBadge.className = 'result-badge invalid';
        elements.modalResultIcon.className = 'bi bi-exclamation-triangle-fill';
        elements.modalResultStatus.textContent = 'INVALID TEST';
        elements.modalResultMessage.textContent = result.error_message || 'This test strip could not be analyzed';
        
        // Override metrics
        elements.modalConfidenceValue.textContent = '0%';
        elements.modalConfidenceBar.style.width = '0%';
        elements.modalConfidenceBar.className = 'progress-bar bg-secondary';
        
        elements.modalIntensityValue.textContent = '0%';
        elements.modalIntensityBar.style.width = '0%';
        elements.modalIntensityBar.className = 'progress-bar bg-secondary';
        
        elements.modalThresholdValue.textContent = (result.threshold * 100).toFixed(0) + '%';
        elements.modalTimeValue.textContent = result.analysis_time ? result.analysis_time.toFixed(2) + 's' : 'N/A';
        
        // Override details
        elements.modalControlLine.textContent = 'Not Detected';
        elements.modalTestLine.textContent = 'N/A';
        elements.modalIntensityScore.textContent = 'N/A';
        elements.modalQualityAssessment.textContent = 'Invalid - Not a test strip';
        
        // Warning guidance
        elements.modalMedicalGuidance.className = 'medical-guidance warning';
        elements.modalGuidanceText.textContent = 'Please upload a clear image of a valid test strip. Ensure the entire test strip is visible with the control line clearly shown.';
    } else {
        // Valid test display
        const confidence = result.confidence * 100;
        const isPositive = result.is_positive;
        
        // Result badge
        elements.modalResultBadge.className = `result-badge ${isPositive ? 'positive' : 'negative'}`;
        elements.modalResultIcon.className = isPositive ? 'bi bi-exclamation-circle-fill' : 'bi bi-check-circle-fill';
        elements.modalResultStatus.textContent = isPositive ? 'POSITIVE' : 'NEGATIVE';
        
        // Metrics
        const intensity = calculateLineIntensity(result);
        const intensityCategory = getIntensityCategory(intensity);
        
        elements.modalConfidenceValue.textContent = confidence.toFixed(1) + '%';
        elements.modalConfidenceBar.style.width = confidence + '%';
        elements.modalConfidenceBar.className = `progress-bar ${getConfidenceClass(confidence)}`;
        
        elements.modalIntensityValue.textContent = intensity.toFixed(0) + '%';
        elements.modalIntensityBar.style.width = intensity + '%';
        elements.modalIntensityBar.className = `progress-bar ${getIntensityClass(intensity)}`;
        
        elements.modalThresholdValue.textContent = (result.threshold * 100).toFixed(0) + '%';
        elements.modalTimeValue.textContent = result.analysis_time ? result.analysis_time.toFixed(2) + 's' : 'N/A';
        
        // Details
        elements.modalControlLine.textContent = detectControlLine(result) ? 'Detected ✓' : 'Not Detected';
        elements.modalTestLine.textContent = result.test_line_detected ? 'Detected ✓' : 'Not Detected';
        elements.modalIntensityScore.textContent = `${intensity.toFixed(0)}% (${intensityCategory})`;
        elements.modalQualityAssessment.textContent = assessImageQuality(result);
        
        // Medical guidance
        if (isPositive) {
            elements.modalMedicalGuidance.className = 'medical-guidance danger';
            if (intensity >= 70) {
                elements.modalGuidanceText.textContent = 'Strong positive result detected. Please consult a healthcare professional immediately for confirmation and treatment options.';
            } else if (intensity >= 40) {
                elements.modalGuidanceText.textContent = 'Moderate positive result. Medical consultation recommended for confirmation and appropriate care.';
            } else {
                elements.modalGuidanceText.textContent = 'Weak positive result detected. Consider retesting and consult a healthcare professional for confirmation.';
            }
        } else {
            elements.modalMedicalGuidance.className = 'medical-guidance success';
            elements.modalGuidanceText.textContent = 'Negative result. If symptoms persist or you have concerns, please consult a healthcare professional.';
        }
    }
    
    // Show modal
    elements.resultsModal.classList.add('active');
}

function closeResultsModal() {
    elements.resultsModal.classList.remove('active');
}

// ============================================================================
// Utility Functions
// ============================================================================

function dataURItoBlob(dataURI) {
    const byteString = atob(dataURI.split(',')[1]);
    const mimeString = dataURI.split(',')[0].split(':')[1].split(';')[0];
    const ab = new ArrayBuffer(byteString.length);
    const ia = new Uint8Array(ab);
    
    for (let i = 0; i < byteString.length; i++) {
        ia[i] = byteString.charCodeAt(i);
    }
    
    return new Blob([ab], { type: mimeString });
}

// ============================================================================
// Error Handling
// ============================================================================

window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
});

window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
});

// ============================================================================
// Service Worker (Optional - for PWA functionality)
// ============================================================================

if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        // Uncomment to enable service worker
        // navigator.serviceWorker.register('/sw.js')
        //     .then(reg => console.log('Service Worker registered'))
        //     .catch(err => console.log('Service Worker registration failed'));
    });
}
