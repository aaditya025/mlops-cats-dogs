const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const previewImage = document.getElementById('preview-image');
const uploadContent = document.querySelector('.upload-content');
const predictBtn = document.getElementById('predict-btn');
const resultSection = document.getElementById('result-section');
const resultLabel = document.getElementById('result-label');
const resultConfidence = document.getElementById('result-confidence');
const confidenceBar = document.getElementById('confidence-bar');
const resultIcon = document.getElementById('result-icon');

let selectedFile = null;

// Handle file selection
dropZone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleFile(e.target.files[0]);
    }
});

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) return;
    
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewImage.hidden = false;
        uploadContent.style.opacity = '0';
        predictBtn.disabled = false;
        resultSection.hidden = true; // Hide previous results
    };
    reader.readAsDataURL(file);
}

// Handle Prediction
predictBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    predictBtn.innerText = "Analyzing...";
    predictBtn.disabled = true;

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
        const response = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Prediction failed');

        const data = await response.json();
        showResult(data);
    } catch (error) {
        alert("Error: " + error.message);
    } finally {
        predictBtn.innerText = "Analyze Image";
        predictBtn.disabled = false;
    }
});

function showResult(data) {
    resultSection.hidden = false;
    resultLabel.innerText = data.label;
    
    const percentage = Math.round(data.confidence * 100);
    resultConfidence.innerText = `Confidence: ${percentage}%`;
    
    // Animate bar
    setTimeout(() => {
        confidenceBar.style.width = `${percentage}%`;
    }, 100);

    // Set Icon and Color
    if (data.label === "Dog") {
        resultIcon.innerText = "🐶";
        confidenceBar.style.backgroundColor = "#3b82f6"; // Blue for Dog
    } else {
        resultIcon.innerText = "🐱";
        confidenceBar.style.backgroundColor = "#f43f5e"; // Pink for Cat
    }
}
