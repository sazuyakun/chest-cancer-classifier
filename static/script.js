class ChestCancerClassifier {
    constructor() {
        this.baseData = '';
        this.initializeElements();
        this.bindEvents();
        this.setupDragAndDrop();
    }

    initializeElements() {
        this.fileInput = document.getElementById('fileInput');
        this.uploadBtn = document.getElementById('uploadBtn');
        this.changeBtn = document.getElementById('changeBtn');
        this.predictBtn = document.getElementById('predictBtn');
        this.uploadArea = document.getElementById('uploadArea');
        this.uploadPlaceholder = document.getElementById('uploadPlaceholder');
        this.imagePreview = document.getElementById('imagePreview');
        this.previewImg = document.getElementById('previewImg');
        this.resultsSection = document.getElementById('resultsSection');
        this.resultsContent = document.getElementById('resultsContent');
        this.loadingOverlay = document.getElementById('loadingOverlay');
    }

    bindEvents() {
        this.uploadBtn.addEventListener('click', () => this.triggerFileInput());
        this.changeBtn.addEventListener('click', () => this.triggerFileInput());
        this.predictBtn.addEventListener('click', () => this.predict());
        this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
    }

    setupDragAndDrop() {
        const dragEvents = ['dragenter', 'dragover', 'dragleave', 'drop'];

        dragEvents.forEach(eventName => {
            this.uploadArea.addEventListener(eventName, this.preventDefaults, false);
            document.body.addEventListener(eventName, this.preventDefaults, false);
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            this.uploadArea.addEventListener(eventName, () => {
                this.uploadArea.classList.add('dragover');
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            this.uploadArea.addEventListener(eventName, () => {
                this.uploadArea.classList.remove('dragover');
            }, false);
        });

        this.uploadArea.addEventListener('drop', (e) => this.handleDrop(e), false);
    }

    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    triggerFileInput() {
        this.fileInput.click();
    }

    handleFileSelect(e) {
        const files = e.target.files;
        if (files.length > 0) {
            this.processFile(files[0]);
        }
    }

    handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            this.processFile(files[0]);
        }
    }

    processFile(file) {
        if (!file.type.startsWith('image/')) {
            this.showError('Please select a valid image file.');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.crossOrigin = 'Anonymous';

            img.onload = () => {
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');

                // Resize image for better performance
                const maxSize = 800;
                let { width, height } = img;

                if (width > height) {
                    if (width > maxSize) {
                        height = (height * maxSize) / width;
                        width = maxSize;
                    }
                } else {
                    if (height > maxSize) {
                        width = (width * maxSize) / height;
                        height = maxSize;
                    }
                }

                canvas.width = width;
                canvas.height = height;
                ctx.drawImage(img, 0, 0, width, height);

                this.baseData = canvas.toDataURL('image/jpeg', 0.8).replace(/^data:image.+;base64,/, '');
                this.showImagePreview(e.target.result);
                this.enablePredictButton();
            };

            img.src = e.target.result;
        };

        reader.readAsDataURL(file);
    }

    showImagePreview(src) {
        this.previewImg.src = src;
        this.uploadPlaceholder.style.display = 'none';
        this.imagePreview.classList.add('active');
        this.uploadArea.style.border = '2px solid var(--success-color)';
    }

    enablePredictButton() {
        this.predictBtn.disabled = false;
        this.predictBtn.style.background = 'var(--success-color)';
    }

    async predict() {
        if (!this.baseData) {
            this.showError('Please select an image first.');
            return;
        }

        this.showLoading(true);
        this.predictBtn.classList.add('loading');

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: this.baseData })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            this.displayResults(result);

        } catch (error) {
            console.error('Prediction error:', error);
            this.showError('Failed to analyze the image. Please try again.');
        } finally {
            this.showLoading(false);
            this.predictBtn.classList.remove('loading');
        }
    }

    displayResults(data) {
        // Assuming the API returns an array where first element contains prediction results
        const results = Array.isArray(data) ? data[0] : data; // [{"image": 'normal'}]

        let confidence = 0;
        let prediction = 'Unknown';
        let additionalInfo = '';

        // Handle different response formats
        if (results.image) {
            prediction = results.image;
            confidence = results.confidence || 0;
        } else if (results.class) {
            prediction = results.class;
            confidence = results.confidence || 0;
        }

        // Create results HTML
        const resultsHTML = `
            <div class="result-item fade-in-up">
                <div class="result-label">Prediction</div>
                <div class="result-value ${prediction.toLowerCase().includes('normal') ? 'normal' : 'cancer'}">
                    ${prediction}
                </div>
            </div>

            <div class="result-metadata">
                <div class="result-label">Metadata</div>
                ${JSON.stringify(results, null, 2)}
            </div>
        `;

        this.resultsContent.innerHTML = resultsHTML;
        this.resultsSection.classList.add('active');

        // Smooth scroll to results
        setTimeout(() => {
            this.resultsSection.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }, 300);
    }

    showLoading(show) {
        if (show) {
            this.loadingOverlay.classList.add('active');
        } else {
            this.loadingOverlay.classList.remove('active');
        }
    }

    showError(message) {
        const errorHTML = `
            <div class="result-item fade-in-up" style="border-left-color: var(--error-color);">
                <div class="result-label">Error</div>
                <div class="result-value" style="color: var(--error-color); font-size: 1.125rem;">
                    ${message}
                </div>
            </div>
        `;

        this.resultsContent.innerHTML = errorHTML;
        this.resultsSection.classList.add('active');
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new ChestCancerClassifier();
});

// Add some nice entrance animations
window.addEventListener('load', () => {
    const elements = document.querySelectorAll('.header, .upload-card, .results-card');
    elements.forEach((el, index) => {
        setTimeout(() => {
            el.style.opacity = '0';
            el.style.transform = 'translateY(20px)';
            el.style.transition = 'all 0.6s ease-out';

            requestAnimationFrame(() => {
                el.style.opacity = '1';
                el.style.transform = 'translateY(0)';
            });
        }, index * 200);
    });
});
