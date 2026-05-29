/**
 * AURA Front-End Application Logic
 * Implements interactive Drag & Drop, async REST API calls,
 * instant client-side threshold recalculation, and session-history caching.
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("fileInput");
    const thresholdSlider = document.getElementById("thresholdSlider");
    const thresholdVal = document.getElementById("thresholdVal");
    
    // Status Bar & Specs elements
    const deviceType = document.getElementById("deviceType");
    const badgeEff = document.getElementById("badgeEff");
    const badgeRes = document.getElementById("badgeRes");
    const badgeXcp = document.getElementById("badgeXcp");
    
    const effParamCount = document.getElementById("effParamCount");
    const effFileSize = document.getElementById("effFileSize");
    const effTargetDevice = document.getElementById("effTargetDevice");
    
    const resParamCount = document.getElementById("resParamCount");
    const resFileSize = document.getElementById("resFileSize");
    const resTargetDevice = document.getElementById("resTargetDevice");
    
    const xcpParamCount = document.getElementById("xcpParamCount");
    const xcpFileSize = document.getElementById("xcpFileSize");
    const xcpTargetDevice = document.getElementById("xcpTargetDevice");

    // Dashboard View States
    const welcomeCard = document.getElementById("welcomeCard");
    const loadingCard = document.getElementById("loadingCard");
    const consensusCard = document.getElementById("consensusCard");
    const individualResults = document.getElementById("individualResults");
    const previewCard = document.getElementById("previewCard");
    const historySection = document.getElementById("historySection");
    
    // Preview Panel
    const imagePreview = document.getElementById("imagePreview");
    const metaDim = document.getElementById("metaDim");
    const metaFormat = document.getElementById("metaFormat");
    const metaSize = document.getElementById("metaSize");
    const metaFilename = document.getElementById("metaFilename");
    const btnReRun = document.getElementById("btnReRun");

    // Consensus Panel
    const verdictTag = document.getElementById("verdictTag");
    const consensusGauge = document.getElementById("consensusGauge");
    const consensusConfidence = document.getElementById("consensusConfidence");
    const consensusTitle = document.getElementById("consensusTitle");
    const consensusSummary = document.getElementById("consensusSummary");
    const statMostConfident = document.getElementById("statMostConfident");
    const statAvgLatency = document.getElementById("statAvgLatency");

    // EfficientNet Elements
    const effPredictBadge = document.getElementById("effPredictBadge");
    const effRealBar = document.getElementById("effRealBar");
    const effRealVal = document.getElementById("effRealVal");
    const effFakeBar = document.getElementById("effFakeBar");
    const effFakeVal = document.getElementById("effFakeVal");
    const effLatency = document.getElementById("effLatency");
    const effDevice = document.getElementById("effDevice");

    // ResNet Elements
    const resPredictBadge = document.getElementById("resPredictBadge");
    const resRealBar = document.getElementById("resRealBar");
    const resRealVal = document.getElementById("resRealVal");
    const resFakeBar = document.getElementById("resFakeBar");
    const resFakeVal = document.getElementById("resFakeVal");
    const resLatency = document.getElementById("resLatency");
    const resDevice = document.getElementById("resDevice");

    // Xception Elements
    const xcpPredictBadge = document.getElementById("xcpPredictBadge");
    const xcpRealBar = document.getElementById("xcpRealBar");
    const xcpRealVal = document.getElementById("xcpRealVal");
    const xcpFakeBar = document.getElementById("xcpFakeBar");
    const xcpFakeVal = document.getElementById("xcpFakeVal");
    const xcpLatency = document.getElementById("xcpLatency");
    const xcpDevice = document.getElementById("xcpDevice");

    // Diagnostics Elements
    const diagnosticHeader = document.getElementById("diagnosticHeader");
    const diagnosticContent = document.getElementById("diagnosticContent");
    const jsonPreview = document.getElementById("jsonPreview");
    const btnCopyJSON = document.getElementById("btnCopyJSON");

    // Session State
    let currentPredictionData = null; // Store loaded API response
    let analysisHistory = []; // Session history cache
    let selectedImageFile = null; // Track currently selected file object (if uploaded)
    let selectedSampleName = null; // Track if we used a sample

    // Init Specs & System Status
    fetchSystemDetails();

    // ==========================================
    // Event Listeners: Drag and Drop
    // ==========================================
    dropZone.addEventListener("click", () => fileInput.click());

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleImageSelected(e.target.files[0]);
        }
    });

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("drag-over");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("drag-over");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("drag-over");
        if (e.dataTransfer.files.length > 0) {
            handleImageSelected(e.dataTransfer.files[0]);
        }
    });

    // ==========================================
    // Event Listeners: Sample Selection
    // ==========================================
    document.querySelectorAll(".sample-btn").forEach(btn => {
        btn.addEventListener("click", (e) => {
            e.stopPropagation();
            const sampleName = btn.getAttribute("data-sample");
            handleSampleSelected(sampleName);
        });
    });

    // ==========================================
    // Event Listeners: Threshold Slider
    // ==========================================
    // "input" updates the label value dynamically in real time
    thresholdSlider.addEventListener("input", (e) => {
        const val = parseFloat(e.target.value).toFixed(2);
        thresholdVal.textContent = val;
        
        // Recalculate class categories immediately in the client without network delay!
        if (currentPredictionData) {
            applyClientThreshold(parseFloat(e.target.value));
        }
    });

    // "change" triggers rerun on backend to match perfectly if we wish (optional, but client-side is faster)
    thresholdSlider.addEventListener("change", (e) => {
        // If we want a precise backend reload, we can trigger re-analysis.
        // But since client-side update handles math perfectly, we save server power!
    });

    // Re-run Button
    btnReRun.addEventListener("click", () => {
        if (selectedImageFile) {
            uploadAndAnalyze(selectedImageFile);
        } else if (selectedSampleName) {
            analyzeSample(selectedSampleName);
        }
    });

    // ==========================================
    // Collapsible Diagnostics Box
    // ==========================================
    diagnosticHeader.addEventListener("click", () => {
        diagnosticHeader.classList.toggle("active");
        diagnosticContent.classList.toggle("hidden");
    });

    // Copy JSON Utility
    btnCopyJSON.addEventListener("click", () => {
        if (!currentPredictionData) return;
        navigator.clipboard.writeText(JSON.stringify(currentPredictionData, null, 2))
            .then(() => {
                const originalText = btnCopyJSON.innerHTML;
                btnCopyJSON.innerHTML = `<i class="fa-solid fa-check"></i> Copied!`;
                btnCopyJSON.style.color = "var(--color-real)";
                setTimeout(() => {
                    btnCopyJSON.innerHTML = originalText;
                    btnCopyJSON.style.color = "";
                }, 2000);
            });
    });

    // Clear History Button
    document.getElementById("btnClearHistory").addEventListener("click", () => {
        analysisHistory = [];
        document.getElementById("historyList").innerHTML = "";
        historySection.classList.add("hidden");
    });

    // ==========================================
    // Core Application Handlers
    // ==========================================
    
    function handleImageSelected(file) {
        selectedImageFile = file;
        selectedSampleName = null;
        uploadAndAnalyze(file);
    }

    function handleSampleSelected(sampleName) {
        selectedImageFile = null;
        selectedSampleName = sampleName;
        analyzeSample(sampleName);
    }

    // Call API for system information on load
    function fetchSystemDetails() {
        fetch("/api/models-info")
            .then(res => res.json())
            .then(data => {
                if (data.status === "success") {
                    deviceType.textContent = data.device.toUpperCase();
                    
                    // EfficientNet-B0 details
                    const eff = data.models.efficientnet;
                    if (eff && eff.status.includes("Loaded")) {
                        badgeEff.classList.remove("error");
                        effParamCount.textContent = formatParams(eff.params);
                        effFileSize.textContent = eff.size_mb.toFixed(1) + " MB";
                        effTargetDevice.textContent = eff.device.toUpperCase();
                    } else {
                        badgeEff.classList.add("error");
                        effParamCount.textContent = "Error";
                        effFileSize.textContent = "N/A";
                        effTargetDevice.textContent = "N/A";
                    }

                    // ResNet50 details
                    const res = data.models.resnet50;
                    if (res && res.status.includes("Loaded")) {
                        badgeRes.classList.remove("error");
                        resParamCount.textContent = formatParams(res.params);
                        resFileSize.textContent = res.size_mb.toFixed(1) + " MB";
                        resTargetDevice.textContent = res.device.toUpperCase();
                    } else {
                        badgeRes.classList.add("error");
                        resParamCount.textContent = "Error";
                        resFileSize.textContent = "N/A";
                        resTargetDevice.textContent = "N/A";
                    }

                    // Xception details
                    const xcp = data.models.xception;
                    if (xcp && xcp.status.includes("Loaded")) {
                        badgeXcp.classList.remove("error");
                        xcpParamCount.textContent = formatParams(xcp.params);
                        xcpFileSize.textContent = xcp.size_mb.toFixed(1) + " MB";
                        xcpTargetDevice.textContent = xcp.device.toUpperCase();
                    } else {
                        badgeXcp.classList.add("error");
                        xcpParamCount.textContent = "Error";
                        xcpFileSize.textContent = "N/A";
                        xcpTargetDevice.textContent = "N/A";
                    }
                }
            })
            .catch(err => {
                console.error("Error fetching system info:", err);
                deviceType.textContent = "DISCONNECTED";
            });
    }

    // Helper: format parameters count
    function formatParams(num) {
        if (!num || isNaN(num)) return "N/A";
        if (num >= 1000000) {
            return (num / 1000000).toFixed(1) + " M";
        }
        return num.toLocaleString();
    }

    // Submit File Upload to API
    function uploadAndAnalyze(file) {
        showLoading(true);
        
        const formData = new FormData();
        formData.append("image", file);
        formData.append("threshold", thresholdSlider.value);
        formData.append("is_sample", "false");

        fetch("/api/predict", {
            method: "POST",
            body: formData
        })
        .then(res => {
            if (!res.ok) throw new Error("HTTP error " + res.status);
            return res.json();
        })
        .then(data => {
            if (data.status === "success") {
                renderAnalysisResults(data);
                addToSessionHistory(data);
            } else {
                alert("Error analyzing image: " + data.message);
                showLoading(false);
            }
        })
        .catch(err => {
            console.error("Prediction error:", err);
            alert("Failed to connect to the neural engine. Make sure Flask server is active.");
            showLoading(false);
        });
    }

    // Submit Sample Request to API
    function analyzeSample(sampleName) {
        showLoading(true);

        const formData = new FormData();
        formData.append("sample_name", sampleName);
        formData.append("threshold", thresholdSlider.value);
        formData.append("is_sample", "true");

        fetch("/api/predict", {
            method: "POST",
            body: formData
        })
        .then(res => {
            if (!res.ok) throw new Error("HTTP error " + res.status);
            return res.json();
        })
        .then(data => {
            if (data.status === "success") {
                renderAnalysisResults(data);
                addToSessionHistory(data);
            } else {
                alert("Error analyzing sample: " + data.message);
                showLoading(false);
            }
        })
        .catch(err => {
            console.error("Sample prediction error:", err);
            alert("Failed to analyze the pre-loaded sample.");
            showLoading(false);
        });
    }

    // Toggle View Cards for Loading State
    function showLoading(isLoading) {
        if (isLoading) {
            welcomeCard.classList.add("hidden");
            consensusCard.classList.add("hidden");
            individualResults.classList.add("hidden");
            loadingCard.classList.remove("hidden");
        } else {
            loadingCard.classList.add("hidden");
        }
    }

    // Render prediction parameters onto DOM
    function renderAnalysisResults(data) {
        showLoading(false);
        currentPredictionData = data;

        // 1. Populate Target Image Preview & Metadata
        imagePreview.src = data.image_url;
        metaDim.textContent = `${data.metadata.width} x ${data.metadata.height} px`;
        metaFormat.textContent = data.metadata.format.toUpperCase();
        metaSize.textContent = `${data.metadata.size_kb} KB`;
        metaFilename.textContent = data.filename;
        previewCard.classList.remove("hidden");

        // 2. Refresh Diagnostic Accordion Payload
        jsonPreview.textContent = JSON.stringify(data, null, 2);

        // 3. Render results using slider value
        applyClientThreshold(parseFloat(thresholdSlider.value));

        // Reveal Cards
        consensusCard.classList.remove("hidden");
        individualResults.classList.remove("hidden");
    }

    // Add items to Session History
    function addToSessionHistory(data) {
        // Prevent duplicate images in history list
        if (analysisHistory.some(item => item.image_url === data.image_url)) {
            return;
        }
        
        analysisHistory.push(data);
        historySection.classList.remove("hidden");

        const historyList = document.getElementById("historyList");
        const item = document.createElement("div");
        item.className = "history-item";
        item.dataset.url = data.image_url;
        
        const isReal = data.consensus.class === "real";
        const badgeClass = data.consensus.class === "mixed" ? "mixed" : (isReal ? "real" : "fake");
        const badgeText = data.consensus.class.toUpperCase();

        item.innerHTML = `
            <div class="history-thumb-wrapper">
                <img src="${data.image_url}" alt="Thumbnail">
                <span class="history-badge ${badgeClass}">${badgeText}</span>
            </div>
            <span class="history-item-label">${data.filename}</span>
        `;

        item.addEventListener("click", () => {
            // Activate style
            document.querySelectorAll(".history-item").forEach(i => i.classList.remove("active"));
            item.classList.add("active");
            
            // Re-render historical data
            renderAnalysisResults(data);
        });

        // Prepend to history container
        historyList.insertBefore(item, historyList.firstChild);
        
        // Highlight active item
        document.querySelectorAll(".history-item").forEach(i => i.classList.remove("active"));
        item.classList.add("active");
    }

    // ==========================================
    // Client-Side Dynamic Recalculator
    // ==========================================
    function applyClientThreshold(threshold) {
        if (!currentPredictionData) return;

        const results = currentPredictionData.results;
        let validResults = [];

        // --- Recalculate individual model classifications ---
        
        // EfficientNet-B0
        const eff = results.efficientnet;
        if (eff && eff.status === "Success") {
            eff.prediction = eff.prob_real >= threshold ? "real" : "fake";
            eff.confidence = eff.prediction === "real" ? eff.prob_real : eff.prob_fake;
            
            updateModelCard(
                "eff", 
                eff.prediction, 
                eff.confidence, 
                eff.prob_real, 
                eff.prob_fake, 
                eff.latency_ms, 
                eff.status
            );
            validResults.push(eff);
        } else {
            disableModelCard("eff", eff ? eff.status : "Model error");
        }

        // ResNet50
        const res = results.resnet50;
        if (res && res.status === "Success") {
            res.prediction = res.prob_real >= threshold ? "real" : "fake";
            res.confidence = res.prediction === "real" ? res.prob_real : res.prob_fake;
            
            updateModelCard(
                "res", 
                res.prediction, 
                res.confidence, 
                res.prob_real, 
                res.prob_fake, 
                res.latency_ms, 
                res.status
            );
            validResults.push(res);
        } else {
            disableModelCard("res", res ? res.status : "Model error");
        }

        // Xception
        const xcp = results.xception;
        if (xcp && xcp.status === "Success") {
            xcp.prediction = xcp.prob_real >= threshold ? "real" : "fake";
            xcp.confidence = xcp.prediction === "real" ? xcp.prob_real : xcp.prob_fake;
            
            updateModelCard(
                "xcp", 
                xcp.prediction, 
                xcp.confidence, 
                xcp.prob_real, 
                xcp.prob_fake, 
                xcp.latency_ms, 
                xcp.status
            );
            validResults.push(xcp);
        } else {
            disableModelCard("xcp", xcp ? xcp.status : "Model error");
        }

        // --- Recalculate Consensus Aggregation ---
        if (validResults.length > 0) {
            const fakes = validResults.filter(r => r.prediction === "fake").length;
            const reals = validResults.filter(r => r.prediction === "real").length;
            
            let consensusClass = "mixed";
            let agreementCount = 0;
            
            if (fakes > reals) {
                consensusClass = "fake";
                agreementCount = fakes;
            } else if (reals > fakes) {
                consensusClass = "real";
                agreementCount = reals;
            }
            
            // Render Consensus Header Badge
            verdictTag.className = "verdict-tag " + consensusClass;
            verdictTag.textContent = consensusClass.toUpperCase();

            // Render consensus card styles & background glows
            consensusCard.className = `dashboard-card consensus-card ${consensusClass}-theme`;

            // Identify most confident model in predictions
            const mostConfident = validResults.reduce((max, r) => r.confidence > max.confidence ? r : max, validResults[0]);
            
            // Compute average latency
            const totalLatency = validResults.reduce((sum, r) => sum + r.latency_ms, 0);
            const avgLatency = totalLatency / validResults.length;

            statMostConfident.textContent = `${mostConfident.name} (${(mostConfident.confidence * 100).toFixed(1)}%)`;
            statAvgLatency.textContent = `${avgLatency.toFixed(1)} ms`;

            // SVG Radial Gauge Update
            // Stroke-dasharray = 2 * PI * R = 2 * 3.14159 * 40 = 251.2
            const radius = 40;
            const circumference = 2 * Math.PI * radius;
            const percentageConfidence = mostConfident.confidence;
            const offset = circumference - (percentageConfidence * circumference);
            consensusGauge.style.strokeDashoffset = offset;
            consensusConfidence.textContent = `${(percentageConfidence * 100).toFixed(1)}%`;

            // Dynamic Descriptive Texts
            let agreeText = "";
            let summaryText = "";

            if (consensusClass === "real") {
                agreeText = `REAL (${agreementCount}/${validResults.length} Models Agree)`;
                if (agreementCount === 3) {
                    summaryText = "Triple architecture convergence! All deep networks strongly predict that this image is fully authentic and original.";
                } else {
                    summaryText = "Majority agreement (2 vs 1). Two neural models classify the target as REAL, though one model expresses dissent. Exercise standard caution.";
                }
            } else if (consensusClass === "fake") {
                agreeText = `FAKE (${agreementCount}/${validResults.length} Models Agree)`;
                if (agreementCount === 3) {
                    summaryText = "Triple engine alarm! All models converge on classification as FAKE, indicating high-probability AI generation, deepfake artifacts, or structural anomalies.";
                } else {
                    summaryText = "Majority agreement (2 vs 1). Two neural models alarm a FAKE prediction, while one network classifies it as Real. High likelihood of synthetic manipulation.";
                }
            } else {
                agreeText = "MIXED CLASSIFICATION TIE";
                summaryText = "Split decision! The loaded active models are divided equally on whether this image is authentic or generated. Check individual confidence metrics.";
            }

            consensusTitle.textContent = agreeText;
            consensusSummary.textContent = summaryText;
        }
    }

    // Helper: Update model card values dynamically
    function updateModelCard(prefix, prediction, confidence, probReal, probFake, latency, status) {
        const badge = document.getElementById(`${prefix}PredictBadge`);
        const realBar = document.getElementById(`${prefix}RealBar`);
        const realVal = document.getElementById(`${prefix}RealVal`);
        const fakeBar = document.getElementById(`${prefix}FakeBar`);
        const fakeVal = document.getElementById(`${prefix}FakeVal`);
        const latencyEl = document.getElementById(`${prefix}Latency`);
        const deviceEl = document.getElementById(`${prefix}Device`);

        // Prediction category styling
        badge.className = `model-prediction-badge ${prediction}`;
        badge.textContent = prediction.toUpperCase();

        // Horizontal Stacked Bars Animations
        realBar.style.width = `${(probReal * 100)}%`;
        realVal.textContent = `${(probReal * 100).toFixed(1)}%`;

        fakeBar.style.width = `${(probFake * 100)}%`;
        fakeVal.textContent = `${(probFake * 100).toFixed(1)}%`;

        latencyEl.textContent = `${latency.toFixed(1)} ms`;
        
        // Match specific device string
        if (prefix === "res") {
            deviceEl.textContent = "CPU"; // ResNet is isolated on CPU to avoid CUDA resource lockups
        } else {
            deviceEl.textContent = deviceType.textContent;
        }
    }

    // Helper: Disable model card on error
    function disableModelCard(prefix, errorMessage) {
        const badge = document.getElementById(`${prefix}PredictBadge`);
        badge.className = "model-prediction-badge error";
        badge.textContent = "INACTIVE";
        
        document.getElementById(`${prefix}RealBar`).style.width = "0%";
        document.getElementById(`${prefix}RealVal`).textContent = "0.0%";
        
        document.getElementById(`${prefix}FakeBar`).style.width = "0%";
        document.getElementById(`${prefix}FakeVal`).textContent = "0.0%";
        
        document.getElementById(`${prefix}Latency`).textContent = "N/A";
        document.getElementById(`${prefix}Device`).textContent = "N/A";
        
        console.warn(`[!] Model card ${prefix} disabled: ${errorMessage}`);
    }

});
