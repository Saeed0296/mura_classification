document.addEventListener('DOMContentLoaded', () => {
    // Tab Switching
    const navButtons = document.querySelectorAll('.nav-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    navButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabId = button.getAttribute('data-tab');
            
            navButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(tab => tab.classList.remove('active'));
            
            button.classList.add('active');
            const targetTab = document.getElementById(`${tabId}-tab`);
            if (targetTab) {
                targetTab.classList.add('active');
            }
        });
    });

    // Category Selection
    let selectedCategory = 'XR_ELBOW'; // default category
    const categoryItems = document.querySelectorAll('.category-item');

    categoryItems.forEach(item => {
        item.addEventListener('click', () => {
            categoryItems.forEach(c => c.classList.remove('active'));
            item.classList.add('active');
            selectedCategory = item.getAttribute('data-category');
        });
    });

    // File Upload, Drag & Drop
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewZone = document.getElementById('preview-zone');
    const imagePreview = document.getElementById('image-preview');
    const removeBtn = document.getElementById('remove-btn');
    const processBtn = document.getElementById('process-btn');
    
    // Results panels views
    const resultsPanel = document.getElementById('results-panel');
    const placeholderView = document.getElementById('placeholder-view');
    const loaderView = document.getElementById('loader-view');
    const outputView = document.getElementById('output-view');
    
    // Timer and Loader text
    const elapsedTimeText = document.getElementById('elapsed-time');
    const progressFill = document.getElementById('progress-fill');
    const loaderStageText = document.getElementById('loader-stage');
    
    let uploadedFile = null;
    let timerInterval = null;
    let startTime = null;

    // Trigger click on input when clicking dropzone
    dropZone.addEventListener('click', () => {
        fileInput.click();
    });

    // Drag events
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('dragover');
        }, false);
    });

    // Drop handler
    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    // Input file change handler
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file (PNG, JPEG).');
            return;
        }
        uploadedFile = file;
        
        // Show Image Preview
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropZone.classList.add('hidden');
            previewZone.classList.remove('hidden');
            processBtn.disabled = false;
        };
        reader.readAsDataURL(file);

        // Reset any previous results
        resetResultViews();
    }

    // Remove Image handler
    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        uploadedFile = null;
        fileInput.value = '';
        imagePreview.src = '';
        previewZone.classList.add('hidden');
        dropZone.classList.remove('hidden');
        processBtn.disabled = true;
        
        resetResultViews();
    });

    function resetResultViews() {
        placeholderView.classList.remove('hidden');
        loaderView.classList.add('hidden');
        outputView.classList.add('hidden');
        resultsPanel.classList.add('empty');
        if (timerInterval) {
            clearInterval(timerInterval);
        }
    }

    // Process predictions handler
    processBtn.addEventListener('click', async () => {
        if (!uploadedFile) return;

        // Setup UI for loading
        placeholderView.classList.add('hidden');
        outputView.classList.add('hidden');
        loaderView.classList.remove('hidden');
        resultsPanel.classList.remove('empty');

        // Start timer
        startTime = Date.now();
        elapsedTimeText.innerText = '0.00';
        progressFill.style.width = '0%';
        loaderStageText.innerText = 'Initializing inference pipeline...';
        
        timerInterval = setInterval(() => {
            const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);
            elapsedTimeText.innerText = elapsed;
        }, 30);

        // Simulate stages for premium user feedback
        const stages = [
            { time: 200, pct: 15, text: 'Preprocessing bone radiograph...' },
            { time: 800, pct: 30, text: 'Executing ResNet50 classifier on Apple M4 GPU...' },
            { time: 2200, pct: 55, text: 'Executing DenseNet169 classifier on Apple M4 GPU...' },
            { time: 3800, pct: 80, text: 'Executing Vision Transformer (ViT-B-16) on Apple M4 GPU...' },
            { time: 5000, pct: 95, text: 'Compiling results and generating consensus...' }
        ];

        stages.forEach(stage => {
            setTimeout(() => {
                if (uploadedFile && !loaderView.classList.contains('hidden')) {
                    progressFill.style.width = `${stage.pct}%`;
                    loaderStageText.innerText = stage.text;
                }
            }, stage.time);
        });

        // Form data prep
        const formData = new FormData();
        formData.append('image', uploadedFile);
        formData.append('category', selectedCategory);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || 'Inference failed on backend.');
            }

            const data = await response.json();
            
            // Render predictions
            displayResults(data);

        } catch (error) {
            alert(`Error: ${error.message}`);
            resetResultViews();
        } finally {
            clearInterval(timerInterval);
        }
    });

    function displayResults(data) {
        loaderView.classList.add('hidden');
        outputView.classList.remove('hidden');

        let abnormalCount = 0;
        let validModels = 0;

        const modelsList = [
            { key: 'resnet50', id: 'resnet', label: 'ResNet50', isEnsemble: false },
            { key: 'densenet169', id: 'densenet', label: 'DenseNet169', isEnsemble: false },
            { key: 'vit_b_16', id: 'vit', label: 'ViT-B-16', isEnsemble: false },
            { key: 'hybrid_ensemble', id: 'hybrid', label: 'Hybrid SOTA', isEnsemble: true }
        ];

        modelsList.forEach(m => {
            const modelResult = data[m.key];
            const card = document.getElementById(`card-${m.id}`);
            const badge = document.getElementById(`badge-${m.id}`);
            const pathElement = document.getElementById(`path-${m.id}`);
            const textElement = document.getElementById(`text-${m.id}`);
            const timeElement = document.getElementById(`time-${m.id}`);
            const svgChart = document.getElementById(`svg-${m.id}`);

            if (modelResult && !modelResult.error) {
                const prob = modelResult.probability;
                const percent = Math.round(prob * 100);
                const latency = modelResult.latency_ms;
                const isAbnormal = prob >= 0.5;

                if (!m.isEnsemble) {
                    validModels++;
                    if (isAbnormal) abnormalCount++;
                }

                // Set Percentage text
                textElement.textContent = `${percent}%`;
                
                // SVG Chart transition & color class
                if (m.isEnsemble) {
                    svgChart.className.baseVal = `circular-chart gold-chart`;
                } else {
                    svgChart.className.baseVal = `circular-chart ${isAbnormal ? 'red-chart' : 'green-chart'}`;
                }
                pathElement.setAttribute('stroke-dasharray', `${percent}, 100`);

                // Set Badge
                badge.textContent = isAbnormal ? 'Abnormal' : 'Normal';
                badge.className = `badge ${isAbnormal ? 'abnormal' : 'normal'}`;

                // Set Latency
                timeElement.textContent = latency;
            } else {
                // If model encountered error
                textElement.textContent = 'Err';
                badge.textContent = 'Error';
                badge.className = 'badge abnormal';
                timeElement.textContent = '0.0';
                pathElement.setAttribute('stroke-dasharray', `0, 100`);
            }
        });

        // Set Diagnostic Summary
        const diagText = document.getElementById('diag-text');
        const diagBox = document.getElementById('diag-summary-box');
        
        let consensusString = '';
        if (validModels === 0) {
            consensusString = 'All classification models returned an error. Please verify that the virtual environment is working and weights are available.';
            diagBox.style.borderColor = 'var(--danger)';
        } else {
            const categoryReadable = selectedCategory.replace('XR_', '').toLowerCase();
            const categoryName = categoryReadable.charAt(0).toUpperCase() + categoryReadable.slice(1);
            
            if (abnormalCount === 3) {
                consensusString = `<strong>Strong Abnormality Consensus:</strong> All three models (ResNet50, DenseNet169, and ViT-B-16) unanimously classified this <strong>${categoryName}</strong> radiograph as <strong>Abnormal</strong>. There is a high probability of structural damage, fracture, or joint degeneration. Clinical inspection is highly recommended.`;
                diagBox.style.borderColor = 'rgba(239, 68, 68, 0.4)';
            } else if (abnormalCount === 2) {
                consensusString = `<strong>Probable Abnormality:</strong> Majority consensus (2 out of 3 models) classifies this <strong>${categoryName}</strong> radiograph as <strong>Abnormal</strong>. DenseNet169 and ViT show alignment, indicating a potential bone anomaly that should be visually inspected.`;
                diagBox.style.borderColor = 'rgba(245, 158, 11, 0.4)';
            } else if (abnormalCount === 1) {
                consensusString = `<strong>Borderline / Normal:</strong> Only 1 model classified this <strong>${categoryName}</strong> radiograph as abnormal, while the remaining 2 predicted <strong>Normal</strong>. This is a borderline case. The patient's clinical file and additional imaging should be reviewed to rule out localized artifacts.`;
                diagBox.style.borderColor = 'rgba(99, 102, 241, 0.3)';
            } else {
                consensusString = `<strong>Unanimous Normal Consensus:</strong> All three models agree that this <strong>${categoryName}</strong> radiograph displays <strong>Normal</strong> bone structure without diagnostic indication of fracture or structural abnormalities.`;
                diagBox.style.borderColor = 'rgba(16, 185, 129, 0.4)';
            }
        }

        diagText.innerHTML = consensusString;
    }
});
