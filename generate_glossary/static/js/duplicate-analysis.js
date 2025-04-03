// Duplicate Analysis Module
class DuplicateAnalysis {
    constructor() {
        this.duplicateData = null;
        this.filteredDuplicates = [];
        this.currentLevel = 2; // Default to level 2
        this.thresholds = {
            similarity: 0.7,
            condProb: 0.3,
            mutualInfo: 0.1
        };
        this.minCoOccurrences = 0;
        this.showZeroCoOccurrences = true;
        this.parentGroups = {};
        this.dataLoaded = false;
        this.selectedDuplicate = null;
    }
    
    init() {
        // Initialize event listeners
        this.initEventListeners();
        
        // Load duplicate analysis data
        this.loadDuplicateData();
    }
    
    initEventListeners() {
        // Level selector
        const levelSelector = document.getElementById('level-selector');
        if (levelSelector) {
            levelSelector.addEventListener('change', (event) => {
                this.currentLevel = parseInt(event.target.value);
                this.loadDuplicateData();
            });
        }
        
        // Refresh button
        const refreshBtn = document.getElementById('refresh-btn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => {
                this.loadDuplicateData();
            });
        }
        
        // Export buttons
        const exportCsvBtn = document.getElementById('export-csv-btn');
        if (exportCsvBtn) {
            exportCsvBtn.addEventListener('click', () => {
                this.exportToCSV();
            });
        }
        
        const exportOverviewBtn = document.getElementById('export-overview-btn');
        if (exportOverviewBtn) {
            exportOverviewBtn.addEventListener('click', () => {
                this.exportOverview();
            });
        }
        
        // Threshold sliders
        const similarityThreshold = document.getElementById('similarity-threshold');
        const condProbThreshold = document.getElementById('cond-prob-threshold');
        const mutualInfoThreshold = document.getElementById('mutual-info-threshold');
        
        if (similarityThreshold) {
            similarityThreshold.addEventListener('input', (event) => {
                this.thresholds.similarity = parseFloat(event.target.value);
                document.getElementById('similarity-value').textContent = this.thresholds.similarity;
                this.applyFilters();
            });
        }
        
        if (condProbThreshold) {
            condProbThreshold.addEventListener('input', (event) => {
                this.thresholds.condProb = parseFloat(event.target.value);
                document.getElementById('cond-prob-value').textContent = this.thresholds.condProb;
                this.applyFilters();
            });
        }
        
        if (mutualInfoThreshold) {
            mutualInfoThreshold.addEventListener('input', (event) => {
                this.thresholds.mutualInfo = parseFloat(event.target.value);
                document.getElementById('mutual-info-value').textContent = this.thresholds.mutualInfo;
                this.applyFilters();
            });
        }
        
        // Min co-occurrences
        const minCoOccurrences = document.getElementById('min-co-occurrences');
        if (minCoOccurrences) {
            minCoOccurrences.addEventListener('change', (event) => {
                this.minCoOccurrences = parseInt(event.target.value);
                this.applyFilters();
            });
        }
        
        // Show zero co-occurrences
        const showZeroCoOccurrences = document.getElementById('show-zero-co-occurrences');
        if (showZeroCoOccurrences) {
            showZeroCoOccurrences.addEventListener('change', (event) => {
                this.showZeroCoOccurrences = event.target.checked;
                this.applyFilters();
            });
        }
        
        // Search filter
        const searchInput = document.getElementById('duplicate-search');
        if (searchInput) {
            searchInput.addEventListener('input', (event) => {
                this.filterDuplicatesBySearch(event.target.value);
            });
        }
        
        // Network parent selector
        const parentNetworkSelector = document.getElementById('parent-network-selector');
        if (parentNetworkSelector) {
            parentNetworkSelector.addEventListener('change', (event) => {
                this.renderNetworkVisualization(event.target.value);
            });
        }
        
        // Initialize tabs
        document.querySelectorAll('.nav-link').forEach(tab => {
            tab.addEventListener('click', (event) => {
                if (tab.getAttribute('href') === '#network-tab') {
                    // Delay to allow the tab to become visible
                    setTimeout(() => {
                        if (parentNetworkSelector && parentNetworkSelector.value) {
                            this.renderNetworkVisualization(parentNetworkSelector.value);
                        }
                    }, 100);
                }
            });
        });
    }
    
    loadDuplicateData() {
        // Show loading indicators
        document.querySelectorAll('.loading').forEach(el => {
            el.style.display = 'block';
        });
        
        fetch(`/api/duplicates?level=${this.currentLevel}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                this.duplicateData = data;
                this.parentGroups = data.parent_groups || {};
                this.dataLoaded = true;
                
                // Process the data
                this.processData();
                
                // Render overview
                this.renderOverview();
                
                // Populate parent network selector
                this.populateParentSelector();
                
                // Apply filters
                this.applyFilters();
                
                // Hide loading indicators
                document.querySelectorAll('.loading').forEach(el => {
                    el.style.display = 'none';
                });
            })
            .catch(error => {
                console.error('Error loading duplicate data:', error);
                this.showErrorMessage(`Failed to load duplicate analysis data: ${error.message}`);
            });
    }
    
    processData() {
        if (!this.duplicateData) return;
        
        // Convert from object to array for easier filtering
        this.allDuplicates = [];
        for (const key in this.duplicateData.parent_groups) {
            const duplicatesInGroup = this.duplicateData.parent_groups[key];
            this.allDuplicates = this.allDuplicates.concat(duplicatesInGroup);
        }
        
        this.filteredDuplicates = [...this.allDuplicates];
    }
    
    applyFilters() {
        if (!this.allDuplicates) return;
        
        this.filteredDuplicates = this.allDuplicates.filter(duplicate => {
            // Check if we should show entries with zero co-occurrences
            if (!this.showZeroCoOccurrences && duplicate.co_occurrences === 0) {
                return false;
            }
            
            // Check minimum co-occurrences
            if (duplicate.co_occurrences < this.minCoOccurrences) {
                return false;
            }
            
            return (
                duplicate.similarity >= this.thresholds.similarity &&
                Math.max(duplicate.cond_prob_1_given_2, duplicate.cond_prob_2_given_1) >= this.thresholds.condProb &&
                duplicate.mutual_information <= this.thresholds.mutualInfo
            );
        });
        
        // Update UI
        this.renderDuplicatesList();
        this.renderParentGroups();
        this.updateOverviewCounts();
        
        // Update parent selector options if needed
        this.populateParentSelector();
    }
    
    filterDuplicatesBySearch(searchTerm) {
        if (!searchTerm) {
            this.applyFilters();
            return;
        }
        
        searchTerm = searchTerm.toLowerCase();
        
        this.filteredDuplicates = this.allDuplicates.filter(duplicate => {
            // First check other filters
            if (!this.showZeroCoOccurrences && duplicate.co_occurrences === 0) {
                return false;
            }
            
            if (duplicate.co_occurrences < this.minCoOccurrences) {
                return false;
            }
            
            if (
                duplicate.similarity < this.thresholds.similarity ||
                Math.max(duplicate.cond_prob_1_given_2, duplicate.cond_prob_2_given_1) < this.thresholds.condProb ||
                duplicate.mutual_information > this.thresholds.mutualInfo
            ) {
                return false;
            }
            
            // Then check search term
            return (
                duplicate.term1.toLowerCase().includes(searchTerm) ||
                duplicate.term2.toLowerCase().includes(searchTerm) ||
                duplicate.parent.toLowerCase().includes(searchTerm)
            );
        });
        
        // Update UI
        this.renderDuplicatesList();
    }
    
    renderOverview() {
        if (!this.duplicateData) return;
        
        const metricsContainer = document.getElementById('overview-metrics');
        if (metricsContainer) {
            // Calculate statistics
            const totalDuplicates = this.allDuplicates.length;
            const uniqueParents = Object.keys(this.parentGroups).length;
            
            let avgSimilarity = 0;
            let highSimilarityCount = 0;
            let lowCondProbCount = 0;
            let zeroCoOccurrences = 0;
            let multipleCoOccurrences = 0;
            
            this.allDuplicates.forEach(duplicate => {
                avgSimilarity += duplicate.similarity;
                if (duplicate.similarity >= 0.8) highSimilarityCount++;
                if (Math.min(duplicate.cond_prob_1_given_2, duplicate.cond_prob_2_given_1) <= 0.1) {
                    lowCondProbCount++;
                }
                if (duplicate.co_occurrences === 0) {
                    zeroCoOccurrences++;
                }
                if (duplicate.co_occurrences > 1) {
                    multipleCoOccurrences++;
                }
            });
            
            avgSimilarity = totalDuplicates > 0 ? avgSimilarity / totalDuplicates : 0;
            
            metricsContainer.innerHTML = `
                <div class="metric-card">
                    <h4>Total Potential Duplicates</h4>
                    <div class="value">${totalDuplicates}</div>
                    <div class="text-muted">Level ${this.currentLevel}</div>
                </div>
                <div class="metric-card">
                    <h4>Unique Parent Categories</h4>
                    <div class="value">${uniqueParents}</div>
                    <div class="text-muted">With potential duplicates</div>
                </div>
                <div class="metric-card">
                    <h4>Average Similarity</h4>
                    <div class="value">${avgSimilarity.toFixed(2)}</div>
                    <div class="text-muted">${highSimilarityCount} pairs with similarity ≥ 0.8</div>
                </div>
                <div class="metric-card">
                    <h4>Co-occurrence Stats</h4>
                    <div class="value">${multipleCoOccurrences}</div>
                    <div class="text-muted">Pairs with multiple co-occurrences</div>
                    <div class="text-danger">${zeroCoOccurrences} pairs with zero co-occurrences</div>
                </div>
            `;
            
            // Add analysis summary content
            const summaryContent = document.getElementById('analysis-summary-content');
            if (summaryContent) {
                const highSimilarPerc = ((highSimilarityCount / totalDuplicates) * 100).toFixed(1);
                const zeroCoOccPerc = ((zeroCoOccurrences / totalDuplicates) * 100).toFixed(1);
                
                summaryContent.innerHTML = `
                <div class="card">
                    <div class="card-body">
                            <p><strong>${highSimilarityCount}</strong> pairs (${highSimilarPerc}%) have high semantic similarity (≥0.8), 
                            suggesting strong overlap in meaning.</p>
                            
                            <p><strong>${zeroCoOccurrences}</strong> pairs (${zeroCoOccPerc}%) have zero co-occurrences in the source data,
                            indicating they might be distinct concepts despite semantic similarity.</p>
                            
                            <p><strong>${multipleCoOccurrences}</strong> pairs have multiple co-occurrences, suggesting they are 
                            frequently used interchangeably or in similar contexts.</p>
                            
                            <p class="mt-3"><strong>Recommendation:</strong> Focus on pairs with both high semantic similarity (≥0.8) 
                            and multiple co-occurrences as the strongest candidates for duplicate consolidation.</p>
                    </div>
                </div>
            `;
        }
        }
        
        // Render scatter plot
        this.renderScatterPlot();
    }
    
    renderScatterPlot() {
        const scatterPlotContainer = document.getElementById('scatter-plot');
        if (!scatterPlotContainer || !this.allDuplicates || this.allDuplicates.length === 0) return;
        
        // Clear previous plot
        scatterPlotContainer.innerHTML = '';
        
        // Set dimensions and margins
        const margin = { top: 30, right: 30, bottom: 60, left: 60 };
        const width = scatterPlotContainer.clientWidth - margin.left - margin.right;
        const height = 400 - margin.top - margin.bottom;
        
        // Create SVG element
        const svg = d3.select('#scatter-plot')
            .append('svg')
            .attr('width', width + margin.left + margin.right)
            .attr('height', height + margin.top + margin.bottom)
            .append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);
        
        // Extract data for plotting
        const plotData = this.allDuplicates.map(d => ({
            x: d.similarity,
            y: Math.min(d.cond_prob_1_given_2, d.cond_prob_2_given_1),
            mi: d.mutual_information,
            term1: d.term1,
            term2: d.term2,
            parent: d.parent
        }));
        
        // X scale (similarity)
        const x = d3.scaleLinear()
            .domain([0.5, 1.0])
            .range([0, width]);
        
        // Y scale (conditional probability)
        const y = d3.scaleLinear()
            .domain([0, 0.5])
            .range([height, 0]);
        
        // Color scale (mutual information)
        const color = d3.scaleSequential()
            .domain([0, 0.3])
            .interpolator(d3.interpolateViridis);
        
        // Add X axis
        svg.append('g')
            .attr('transform', `translate(0,${height})`)
            .call(d3.axisBottom(x));
        
        // Add Y axis
        svg.append('g')
            .call(d3.axisLeft(y));
        
        // Add X axis label
        svg.append('text')
            .attr('text-anchor', 'middle')
            .attr('x', width / 2)
            .attr('y', height + margin.bottom - 10)
            .text('Embedding Similarity');
        
        // Add Y axis label
        svg.append('text')
            .attr('text-anchor', 'middle')
            .attr('transform', 'rotate(-90)')
            .attr('x', -height / 2)
            .attr('y', -margin.left + 15)
            .text('Conditional Probability (min)');
        
        // Add threshold lines
        svg.append('line')
            .attr('x1', x(this.thresholds.similarity))
            .attr('y1', 0)
            .attr('x2', x(this.thresholds.similarity))
            .attr('y2', height)
            .attr('stroke', 'red')
            .attr('stroke-dasharray', '4')
            .attr('stroke-width', 1);
        
        svg.append('line')
            .attr('x1', 0)
            .attr('y1', y(this.thresholds.condProb))
            .attr('x2', width)
            .attr('y2', y(this.thresholds.condProb))
            .attr('stroke', 'red')
            .attr('stroke-dasharray', '4')
            .attr('stroke-width', 1);
        
        // Add dots
        const dots = svg.append('g')
            .selectAll('dot')
            .data(plotData)
            .enter()
            .append('circle')
            .attr('cx', d => x(d.x))
            .attr('cy', d => y(d.y))
            .attr('r', 5)
            .style('fill', d => color(d.mi))
            .style('opacity', 0.7)
            .style('stroke', 'white')
            .style('stroke-width', 0.5);
        
        // Add tooltip
        const tooltip = d3.select('#scatter-plot')
            .append('div')
            .style('position', 'absolute')
            .style('visibility', 'hidden')
            .style('background-color', 'white')
            .style('border', '1px solid #ddd')
            .style('border-radius', '4px')
            .style('padding', '8px')
            .style('font-size', '12px')
            .style('box-shadow', '0 0 10px rgba(0,0,0,0.1)')
            .style('z-index', 1000);
        
        dots.on('mouseover', function(event, d) {
            d3.select(this)
                .transition()
                .duration(200)
                .attr('r', 8)
                .style('opacity', 1);
            
            tooltip
                .style('visibility', 'visible')
                .html(`
                    <strong>Terms:</strong> ${d.term1} | ${d.term2}<br>
                    <strong>Parent:</strong> ${d.parent}<br>
                    <strong>Similarity:</strong> ${d.x.toFixed(2)}<br>
                    <strong>Cond. Prob.:</strong> ${d.y.toFixed(2)}<br>
                    <strong>Mutual Info:</strong> ${d.mi.toFixed(3)}
                `);
        })
        .on('mousemove', function(event) {
            tooltip
                .style('left', (event.pageX + 15) + 'px')
                .style('top', (event.pageY - 20) + 'px');
        })
        .on('mouseout', function() {
            d3.select(this)
                .transition()
                .duration(200)
                .attr('r', 5)
                .style('opacity', 0.7);
            
            tooltip.style('visibility', 'hidden');
        });
    }
    
    renderDuplicatesList() {
        const tableBody = document.getElementById('duplicates-table-body');
        if (!tableBody) return;
        
        if (this.filteredDuplicates.length === 0) {
            tableBody.innerHTML = `<tr><td colspan="8" class="text-center">No duplicate pairs found matching the current filters</td></tr>`;
            return;
        }
        
        let html = '';
        this.filteredDuplicates.forEach((duplicate, index) => {
            // Classify the similarity score
            let similarityClass = "low";
            if (duplicate.similarity >= 0.8) similarityClass = "high";
            else if (duplicate.similarity >= 0.7) similarityClass = "medium";
            
            // Classify the conditional probability
            const maxCondProb = Math.max(duplicate.cond_prob_1_given_2, duplicate.cond_prob_2_given_1);
            let condProbClass = "low";
            if (maxCondProb >= 0.5) condProbClass = "high";
            else if (maxCondProb >= 0.3) condProbClass = "medium";
            
            // Format the mutual information
            const mutualInfoFormatted = duplicate.mutual_information.toExponential(3);
            
            html += `
                <tr data-index="${index}">
                    <td>${duplicate.term1}</td>
                    <td>${duplicate.term2}</td>
                    <td>${duplicate.parent}</td>
                    <td><span class="metric-badge ${similarityClass}">${duplicate.similarity.toFixed(2)}</span></td>
                    <td>
                        <div>T1|T2: ${duplicate.cond_prob_1_given_2.toFixed(2)}</div>
                        <div>T2|T1: ${duplicate.cond_prob_2_given_1.toFixed(2)}</div>
                    </td>
                    <td>${mutualInfoFormatted}</td>
                    <td>${duplicate.co_occurrences}</td>
                    <td>
                        <button class="btn btn-sm btn-outline-primary view-details-btn" data-index="${index}">
                            <i class="bi bi-info-circle"></i>
                        </button>
                    </td>
                </tr>
            `;
        });
        
        tableBody.innerHTML = html;
        
        // Add event listeners for detail buttons
        document.querySelectorAll('.view-details-btn').forEach(button => {
            button.addEventListener('click', (event) => {
                const index = parseInt(event.currentTarget.getAttribute('data-index'));
                this.showDuplicateDetails(this.filteredDuplicates[index]);
                
                // Switch to the details tab
                document.querySelector('a[href="#duplicate-details-tab"]').click();
            });
        });
    }
    
    showDuplicateDetails(duplicate) {
        this.selectedDuplicate = duplicate;
        
        const detailsContainer = document.getElementById('duplicate-details-container');
        if (!detailsContainer) return;
        
        // Clone the template
        const template = document.getElementById('duplicate-pair-template');
        const detailsElement = template.content.cloneNode(true);
        
        // Populate the template with data
        detailsElement.querySelector('.term1').textContent = duplicate.term1;
        detailsElement.querySelector('.term2').textContent = duplicate.term2;
        detailsElement.querySelector('.parent').textContent = duplicate.parent;
        
        // Debug log to verify data
        console.log('Showing duplicate details:', JSON.stringify(duplicate));
        
        // Term abbreviations for probabilities
        const term1Short = duplicate.term1.length > 10 ? duplicate.term1.substring(0, 10) + "..." : duplicate.term1;
        const term2Short = duplicate.term2.length > 10 ? duplicate.term2.substring(0, 10) + "..." : duplicate.term2;
        
        // Update all term1-short elements
        const term1ShortElements = detailsElement.querySelectorAll('.term1-short');
        term1ShortElements.forEach(el => {
            el.textContent = term1Short;
        });
        
        // Update all term2-short elements
        const term2ShortElements = detailsElement.querySelectorAll('.term2-short');
        term2ShortElements.forEach(el => {
            el.textContent = term2Short;
        });
        
        // Similarity
        detailsElement.querySelector('.similarity').textContent = duplicate.similarity.toFixed(2);
        const similarityIndicator = detailsElement.querySelector('.similarity-indicator');
        similarityIndicator.style.setProperty('--percentage', `${duplicate.similarity * 100}%`);
        
        // Co-occurrences
        detailsElement.querySelector('.co-occurrences').textContent = duplicate.co_occurrences;
        
        // Mutual information
        const mutualInfoValue = detailsElement.querySelector('.mutual-info-value');
        mutualInfoValue.textContent = duplicate.mutual_information.toExponential(3);
        
        // Scale the mutual info bar to a max of 0.005 (common range for the data)
        const scaledMI = Math.min(duplicate.mutual_information / 0.005, 1) * 100;
        const mutualInfoFill = detailsElement.querySelector('.mutual-info-fill');
        mutualInfoFill.style.width = `${scaledMI}%`;
        
        // Conditional probabilities
        detailsElement.querySelector('.cond-prob-1-given-2').textContent = duplicate.cond_prob_1_given_2.toFixed(2);
        detailsElement.querySelector('.cond-prob-2-given-1').textContent = duplicate.cond_prob_2_given_1.toFixed(2);
        
        // Conditional probability bars
        const condProb1Bar = detailsElement.querySelector('.cond-prob-1-bar');
        condProb1Bar.style.width = `${duplicate.cond_prob_1_given_2 * 100}%`;
        
        const condProb2Bar = detailsElement.querySelector('.cond-prob-2-bar');
        condProb2Bar.style.width = `${duplicate.cond_prob_2_given_1 * 100}%`;
        
        // Co-occurrence matrix
        let a, b, c, d;
        
        // Use the contingency table from the backend if available
        if (duplicate.contingency_table && Object.keys(duplicate.contingency_table).length > 0) {
            console.log('Using backend contingency table');
            
            // Use the raw values from the backend
            a = parseInt(duplicate.contingency_table.a);
            b = parseInt(duplicate.contingency_table.b);
            c = parseInt(duplicate.contingency_table.c);
            d = parseInt(duplicate.contingency_table.d);
            
            // Verify data integrity
            console.log(`Contingency table - a: ${a}, b: ${b}, c: ${c}, d: ${d}`);
            console.log(`Total docs from backend: ${duplicate.contingency_table.total_docs}`);
        } else {
            console.log('No contingency table found, calculating from scratch');
            
            // Fallback: Calculate from co-occurrences and conditional probabilities
            a = duplicate.co_occurrences;
            
            // Calculate b and c based on conditional probabilities
            const term2Total = Math.max(1, Math.round(a / Math.max(0.0001, duplicate.cond_prob_1_given_2)));
            const term1Total = Math.max(1, Math.round(a / Math.max(0.0001, duplicate.cond_prob_2_given_1)));
            
            b = term1Total - a;
            c = term2Total - a;
            
            // Use a reasonable total_docs value (100 is a good minimum)
            const total_docs = Math.max(100, a + b + c + 10);
            d = total_docs - a - b - c;
        }
        
        // Ensure all values are valid numbers
        a = isNaN(a) ? 0 : a;
        b = isNaN(b) ? 0 : b;
        c = isNaN(c) ? 0 : c;
        d = isNaN(d) ? 0 : d;
        
        // Strictly calculate all totals to ensure mathematical consistency
        const term1Total = a + b;
        const term2Total = a + c;
        const term1AbsentTotal = c + d;
        const term2AbsentTotal = b + d;
        const grandTotal = a + b + c + d;
        
        // Validate that totals are consistent
        console.log(`Calculated - Row 1 total: ${term1Total}`);
        console.log(`Calculated - Row 2 total: ${term1AbsentTotal}`);
        console.log(`Calculated - Col 1 total: ${term2Total}`);
        console.log(`Calculated - Col 2 total: ${term2AbsentTotal}`);
        console.log(`Calculated - Grand total: ${grandTotal}`);
        console.log(`Sum of rows: ${term1Total + term1AbsentTotal}`);
        console.log(`Sum of columns: ${term2Total + term2AbsentTotal}`);
        
        // Update the DOM with the calculated values
        const cellElements = {
            a: detailsElement.querySelector('.both-present'),
            b: detailsElement.querySelector('.term1-only'),
            c: detailsElement.querySelector('.term2-only'),
            d: detailsElement.querySelector('.both-absent'),
            term1Total: detailsElement.querySelector('.term1-total'),
            term2Total: detailsElement.querySelector('.term2-total'),
            term1AbsentTotal: detailsElement.querySelector('.term1-absent-total'),
            term2AbsentTotal: detailsElement.querySelector('.term2-absent-total'),
            grandTotal: detailsElement.querySelector('.grand-total')
        };
        
        // Set all values 
        cellElements.a.textContent = a;
        cellElements.b.textContent = b;
        cellElements.c.textContent = c;
        cellElements.d.textContent = d;
        cellElements.term1Total.textContent = term1Total;
        cellElements.term2Total.textContent = term2Total;
        cellElements.term1AbsentTotal.textContent = term1AbsentTotal;
        cellElements.term2AbsentTotal.textContent = term2AbsentTotal;
        cellElements.grandTotal.textContent = grandTotal;
        
        // Recommendation
        const recommendationContainer = detailsElement.querySelector('.recommendation-container');
        let recommendation = '';
        const maxCondProb = Math.max(duplicate.cond_prob_1_given_2, duplicate.cond_prob_2_given_1);

        // Primary criteria: High similarity, LOW conditional probability, LOW mutual information -> Strong Duplicate signal (interchangeable use)
        if (duplicate.similarity >= 0.7 && maxCondProb <= 0.3 && duplicate.mutual_information <= 0.1) {
            recommendation = `
                <div class="alert alert-success">
                    <strong>Strong Duplicate Candidate:</strong> High similarity (${duplicate.similarity.toFixed(2)}) combined with low conditional probability (max: ${maxCondProb.toFixed(2)}) and low mutual information (${duplicate.mutual_information.toExponential(3)}) strongly suggests these terms are used interchangeably for the same concept across different contexts.
                    <hr>
                    <strong>Recommendation:</strong> Prioritize for review and consolidation. Identify the relationship (synonym, variation, etc.) and merge according to guidelines.
                </div>
            `;
        // High similarity BUT ALSO High conditional probability/co-occurrence -> Related, not duplicate
        } else if (duplicate.similarity >= 0.8 && (duplicate.co_occurrences > 1 || maxCondProb > 0.3)) { // Note: Adjusted maxCondProb threshold check
             recommendation = `
                <div class="alert alert-info">
                    <strong>Closely Related Terms:</strong> These terms have high semantic similarity (${duplicate.similarity.toFixed(2)}) but also appear together relatively often (Co-occurrences: ${duplicate.co_occurrences}, Max Cond Prob: ${maxCondProb.toFixed(2)}). They are likely related concepts used in conjunction, rather than duplicates.
                    <hr>
                    <strong>Recommendation:</strong> Verify their relationship. They likely represent distinct but associated ideas (e.g., field and sub-field). Consolidation is generally not needed.
                </div>
            `;
        // Moderate similarity or other cases not meeting the strong criteria
        } else if (duplicate.similarity >= 0.7) {
             recommendation = `
                <div class="alert alert-warning">
                    <strong>Possible Duplicate/Related:</strong> These terms show moderate-to-high similarity (${duplicate.similarity.toFixed(2)}) but don't fit the clear patterns for strong duplicates or closely related terms (Max Cond Prob: ${maxCondProb.toFixed(2)}, MI: ${duplicate.mutual_information.toExponential(3)}).
                    <hr>
                    <strong>Recommendation:</strong> Review context. Could be a weaker duplicate signal or a less common relationship. Lower priority than strong candidates.
                </div>
            `;
        // Low similarity cases
        } else {
            recommendation = `
                <div class="alert alert-secondary">
                    <strong>Unlikely Duplicate:</strong> Semantic similarity (${duplicate.similarity.toFixed(2)}) is below the typical threshold for consideration.
                    <hr>
                    <strong>Recommendation:</strong> Safe to keep separate unless domain knowledge suggests otherwise (e.g., very specific jargon variation).
                </div>
            `;
        }
        
        recommendationContainer.innerHTML = recommendation;
        
        // Clear previous content and add new details
        detailsContainer.innerHTML = '';
        detailsContainer.appendChild(detailsElement);
    }
    
    renderParentGroups() {
        const groupsContainer = document.getElementById('parent-groups-container');
        if (!groupsContainer) return;
        
        // Group the filtered duplicates by parent
        const filteredGroups = {};
        this.filteredDuplicates.forEach(duplicate => {
            const parent = duplicate.parent;
            if (!filteredGroups[parent]) {
                filteredGroups[parent] = [];
            }
            filteredGroups[parent].push(duplicate);
        });
        
        if (Object.keys(filteredGroups).length === 0) {
            groupsContainer.innerHTML = `<div class="alert alert-info">No parent groups found matching the current filters</div>`;
            return;
        }
        
        let html = '';
        
        // Sort parents by number of duplicates (descending)
        const sortedParents = Object.keys(filteredGroups).sort((a, b) => 
            filteredGroups[b].length - filteredGroups[a].length
        );
        
        sortedParents.forEach(parent => {
            const duplicates = filteredGroups[parent];
            
            html += `
                <div class="card mb-3">
                    <div class="card-header">
                        <h5 class="mb-0 d-flex justify-content-between align-items-center">
                            <span>${parent}</span>
                            <span class="badge bg-primary">${duplicates.length} pairs</span>
                        </h5>
                    </div>
                    <div class="card-body">
                    <div class="table-responsive">
                        <table class="table table-sm">
                            <thead>
                                <tr>
                                    <th>Term 1</th>
                                    <th>Term 2</th>
                                    <th>Similarity</th>
                                        <th>Co-occurrences</th>
                                        <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody>
            `;
            
            duplicates.forEach((duplicate, index) => {
                html += `
                    <tr>
                        <td>${duplicate.term1}</td>
                        <td>${duplicate.term2}</td>
                        <td><span class="metric-badge ${duplicate.similarity >= 0.8 ? 'high' : duplicate.similarity >= 0.7 ? 'medium' : 'low'}">${duplicate.similarity.toFixed(2)}</span></td>
                        <td>${duplicate.co_occurrences}</td>
                        <td>
                            <button class="btn btn-sm btn-outline-primary view-details-btn-group" data-parent="${parent}" data-index="${index}">
                                <i class="bi bi-info-circle"></i>
                            </button>
                        </td>
                    </tr>
                `;
            });
            
            html += `
                            </tbody>
                        </table>
                        </div>
                    </div>
                </div>
            `;
        });
        
        groupsContainer.innerHTML = html;
        
        // Add event listeners for detail buttons
        document.querySelectorAll('.view-details-btn-group').forEach(button => {
            button.addEventListener('click', (event) => {
                const parent = event.currentTarget.getAttribute('data-parent');
                const index = parseInt(event.currentTarget.getAttribute('data-index'));
                this.showDuplicateDetails(filteredGroups[parent][index]);
                
                // Switch to the details tab
                document.querySelector('a[href="#duplicate-details-tab"]').click();
            });
        });
    }
    
    populateParentSelector() {
        const selector = document.getElementById('parent-network-selector');
        if (!selector) return;
        
        // Group duplicates by parent
        const parentGroups = {};
        this.filteredDuplicates.forEach(duplicate => {
            if (!parentGroups[duplicate.parent]) {
                parentGroups[duplicate.parent] = [];
            }
            parentGroups[duplicate.parent].push(duplicate);
        });
        
        // Create options
        selector.innerHTML = '<option value="">Select a parent category</option>';
        
        Object.keys(parentGroups)
            .sort()
            // .filter(parent => parentGroups[parent].length >= 3) // Only show parents with at least 3 duplicates
            .forEach(parent => {
                const option = document.createElement('option');
                option.value = parent;
                option.textContent = `${parent} (${parentGroups[parent].length} duplicates)`;
                selector.appendChild(option);
            });
    }
    
    renderNetworkVisualization(parent) {
        const container = document.getElementById('network-visualization');
        if (!container || !parent) return;
        
        // Clear previous content
        container.innerHTML = '';
        
        // Find duplicates for this parent
        const duplicatesForParent = this.filteredDuplicates.filter(d => d.parent === parent);
        
        if (duplicatesForParent.length === 0) {
            container.innerHTML = '<div class="alert alert-info">No duplicates found for this parent</div>';
            return;
        }
        
        // Prepare network data
        const nodes = new Set();
        duplicatesForParent.forEach(d => {
            nodes.add(d.term1);
            nodes.add(d.term2);
        });
        
        const nodeData = Array.from(nodes).map(term => ({ 
            id: term,
            level: 2, // Most terms will be level 2, but this can be made dynamic
            similarity: Math.max(...duplicatesForParent
                .filter(d => d.term1 === term || d.term2 === term)
                .map(d => d.similarity))
        }));
        
        const linkData = duplicatesForParent.map(d => ({
            source: d.term1,
            target: d.term2,
            similarity: d.similarity,
            condProb: Math.max(d.cond_prob_1_given_2, d.cond_prob_2_given_1),
            mutualInfo: d.mutual_information,
            coOccurrences: d.co_occurrences
        }));
        
        // Set dimensions
        const width = container.clientWidth;
        const height = 500;
        
        // Create SVG with zoom capabilities
        const svg = d3.select('#network-visualization')
            .append('svg')
            .attr('width', width)
            .attr('height', height);
        
        // Add zoom and pan capabilities
        const zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on('zoom', (event) => {
                g.attr('transform', event.transform);
            });
            
        svg.call(zoom);
        
        // Create a group for all elements
        const g = svg.append('g');
        
        // Create force simulation with improved layout
        const simulation = d3.forceSimulation(nodeData)
            .force('link', d3.forceLink(linkData)
                .id(d => d.id)
                .distance(d => 150 * (1 - Math.min(0.9, d.similarity))))
            .force('charge', d3.forceManyBody().strength(-300))
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('collide', d3.forceCollide().radius(d => 40));
        
        // Create links with better styling
        const link = g.selectAll('.link')
            .data(linkData)
            .enter()
            .append('line')
            .attr('class', 'link')
            .attr('stroke-width', d => 1 + 2 * d.similarity) // Width based on similarity
            .attr('stroke', d => d.coOccurrences > 1 ? 
                  '#333' : '#999') // Darker for multiple co-occurrences
            .attr('stroke-opacity', d => 0.4 + 0.4 * d.similarity); // More opaque for higher similarity
            
        // Create node elements
        const nodeGroup = g.selectAll('.node')
            .data(nodeData)
            .enter()
            .append('g')
            .attr('class', 'node')
            .call(d3.drag()
                .on('start', dragstarted)
                .on('drag', dragged)
                .on('end', dragended));
        
        // Add circles to nodes with consistent styling
        nodeGroup.append('circle')
            .attr('r', d => {
                // Size based on connection count
                const connectionCount = linkData.filter(link => 
                    link.source.id === d.id || link.target.id === d.id
                ).length;
                return 10 + Math.min(10, connectionCount * 2);
            })
            .attr('class', d => `term-node`)
            .attr('fill', d => {
                // Color based on similarity (like the hierarchy visualizer)
                const similarity = d.similarity || 0.5;
                if (similarity >= 0.9) return '#EA4335'; // Red for high similarity
                if (similarity >= 0.8) return '#FBBC05'; // Yellow/Orange for medium
                if (similarity >= 0.7) return '#34A853'; // Green for lower
                return '#4285F4'; // Blue for lowest
            })
            .attr('stroke', '#fff')
            .attr('stroke-width', 1.5);
        
        // Add text labels using SVG text elements (more control than in the original)
        const textElements = nodeGroup.append('text')
            .attr('dy', d => {
                // Offset text based on node radius to place it below the node
                const connectionCount = linkData.filter(link => 
                    link.source.id === d.id || link.target.id === d.id
                ).length;
                const radius = 10 + Math.min(10, connectionCount * 2);
                return radius + 15;
            })
            .attr('text-anchor', 'middle')
            .attr('font-size', 12)
            .attr('font-family', 'Arial, sans-serif')
            .attr('pointer-events', 'none');
        
        // Handle multi-line text
        textElements.each(function(d) {
                const text = d3.select(this);
                const words = d.id.split(' ');
            
            // Use word wrapping
            let line = '';
            let lineNumber = 0;
            const lineHeight = 1.1; // ems
            const y = 0;
            const dy = 0;
            let tspan = text.append('tspan')
                        .attr('x', 0)
                .attr('y', y)
                .attr('dy', dy + 'em');
            
            for (let i = 0; i < words.length; i++) {
                const word = words[i];
                const testLine = line + word + ' ';
                // If adding this word would make the line too long (more than 15 chars)
                // New line with just this word
                if (testLine.length > 15) {
                    tspan.text(line);
                    line = word + ' ';
                    tspan = text.append('tspan')
                        .attr('x', 0)
                        .attr('y', y)
                        .attr('dy', ++lineNumber * lineHeight + dy + 'em')
                        .text(word);
                } else {
                    line = testLine;
                    tspan.text(line);
                }
            }
        });
        
        // Add tooltips to nodes
        nodeGroup.append('title')
            .text(d => {
                const connections = linkData.filter(link => 
                    link.source.id === d.id || link.target.id === d.id
                );
                
                let tooltip = d.id;
                tooltip += `\nConnections: ${connections.length}`;
                tooltip += `\nHighest similarity: ${d.similarity.toFixed(2)}`;
                return tooltip;
            });
            
        // Add hover effect to highlight connections
        nodeGroup
            .on('mouseover', function(event, d) {
                // Highlight connected links
                link
                    .style('stroke-opacity', l => 
                        (l.source.id === d.id || l.target.id === d.id) ? 0.8 : 0.1
                    )
                    .style('stroke-width', l => 
                        (l.source.id === d.id || l.target.id === d.id) ? 
                        (1 + 3 * l.similarity) : 1
                    );
                
                // Highlight connected nodes
                nodeGroup.selectAll('circle')
                    .style('opacity', nd => {
                        // Check if this node is connected to the hovered node
                        const isConnected = linkData.some(l => 
                            (l.source.id === d.id && l.target.id === nd.id) ||
                            (l.source.id === nd.id && l.target.id === d.id)
                        );
                        return isConnected || nd.id === d.id ? 1 : 0.3;
                    });
                
                // Make connected text more visible
                textElements
                    .style('font-weight', nd => {
                        // Check if this node is connected to the hovered node
                        const isConnected = linkData.some(l => 
                            (l.source.id === d.id && l.target.id === nd.id) ||
                            (l.source.id === nd.id && l.target.id === d.id)
                        );
                        return isConnected || nd.id === d.id ? 'bold' : 'normal';
                    });
            })
            .on('mouseout', function() {
                // Reset styles
                link
                    .style('stroke-opacity', d => 0.4 + 0.4 * d.similarity)
                    .style('stroke-width', d => 1 + 2 * d.similarity);
                
                nodeGroup.selectAll('circle').style('opacity', 1);
                textElements.style('font-weight', 'normal');
            });
        
        // Update positions on simulation tick
        simulation.on('tick', () => {
            // Keep nodes within bounds
            nodeData.forEach(d => {
                d.x = Math.max(50, Math.min(width - 50, d.x));
                d.y = Math.max(50, Math.min(height - 50, d.y));
            });
            
            link
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);
            
            nodeGroup.attr('transform', d => `translate(${d.x},${d.y})`);
        });
        
        // Drag functions
        function dragstarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }
        
        function dragged(event, d) {
            d.fx = event.x;
            d.fy = event.y;
        }
        
        function dragended(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }
        
        // Add network details with improved information
        const detailsContainer = document.getElementById('network-details');
        if (detailsContainer) {
            // Calculate statistics
            const avgSimilarity = linkData.reduce((sum, d) => sum + d.similarity, 0) / linkData.length;
            const highSimilarityLinks = linkData.filter(d => d.similarity >= 0.8).length;
            const multiCoOccurLinks = linkData.filter(d => d.coOccurrences > 1).length;
            
            // Get most connected terms
            const termConnections = {};
            nodeData.forEach(node => {
                termConnections[node.id] = linkData.filter(link => 
                    link.source.id === node.id || link.target.id === node.id
                ).length;
            });
            
            const mostConnected = Object.entries(termConnections)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 3);
            
            detailsContainer.innerHTML = `
                <div class="card">
                    <div class="card-header">
                        <h5>Network Statistics for "${parent}"</h5>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-6">
                                <p><strong>Terms:</strong> ${nodeData.length}</p>
                                <p><strong>Connections:</strong> ${linkData.length}</p>
                                <p><strong>Avg similarity:</strong> ${avgSimilarity.toFixed(2)}</p>
                                <p><strong>High similarity pairs:</strong> ${highSimilarityLinks} (${((highSimilarityLinks/linkData.length)*100).toFixed(1)}%)</p>
                            </div>
                            <div class="col-md-6">
                                <p><strong>Multiple co-occurrences:</strong> ${multiCoOccurLinks} pairs</p>
                                <p><strong>Most connected terms:</strong></p>
                                <ul>
                                    ${mostConnected.map(([term, count]) => 
                                        `<li>${term} (${count} connections)</li>`
                                    ).join('')}
                                </ul>
                            </div>
                        </div>
                        <div class="mt-3">
                            <button class="btn btn-sm btn-outline-secondary" id="reset-zoom-btn">
                                <i class="bi bi-arrows-fullscreen"></i> Reset Zoom
                            </button>
                        </div>
                    </div>
                </div>
            `;
            
            // Add reset zoom button functionality
            document.getElementById('reset-zoom-btn').addEventListener('click', () => {
                svg.transition().duration(750).call(
                    zoom.transform,
                    d3.zoomIdentity,
                    d3.zoomTransform(svg.node()).invert([width / 2, height / 2])
                );
            });
        }
    }
    
    updateOverviewCounts() {
        if (!this.duplicateData) return;
        
        const overviewTab = document.getElementById('overview-tab');
        if (overviewTab) {
            const filteredCount = this.filteredDuplicates.length;
            const totalCount = this.allDuplicates.length;
            
        const metricsContainer = document.getElementById('overview-metrics');
            if (metricsContainer) {
                const metricCards = metricsContainer.querySelectorAll('.metric-card');
                if (metricCards.length > 0) {
                    const firstCard = metricCards[0];
                    const valueElement = firstCard.querySelector('.value');
                    
                    if (valueElement) {
                        valueElement.innerHTML = `${filteredCount} <small class="text-muted">of ${totalCount}</small>`;
                    }
                }
            }
        }
    }
    
    exportToCSV() {
        if (!this.filteredDuplicates || this.filteredDuplicates.length === 0) {
            alert('No data to export');
            return;
        }
        
        // CSV header
        let csvContent = 'Term 1,Term 2,Parent,Similarity,Conditional Prob 1|2,Conditional Prob 2|1,Co-occurrences,Mutual Information\n';
        
        // Add rows
        this.filteredDuplicates.forEach(duplicate => {
            const row = [
                `"${duplicate.term1}"`,
                `"${duplicate.term2}"`,
                `"${duplicate.parent}"`,
                duplicate.similarity,
                duplicate.cond_prob_1_given_2,
                duplicate.cond_prob_2_given_1,
                duplicate.co_occurrences,
                duplicate.mutual_information
            ];
            csvContent += row.join(',') + '\n';
        });
        
        // Create download link
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.setAttribute('href', url);
        link.setAttribute('download', `duplicate_analysis_lv${this.currentLevel}.csv`);
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
    
    exportOverview() {
        if (!this.duplicateData) {
            alert('No data to export');
            return;
        }
        
        // Calculate statistics for export
        const totalDuplicates = this.allDuplicates.length;
        const uniqueParents = Object.keys(this.parentGroups).length;
        
        let avgSimilarity = 0;
        let highSimilarityCount = 0;
        let lowCondProbCount = 0;
        let zeroCoOccurrences = 0;
        let multipleCoOccurrences = 0;
        
        this.allDuplicates.forEach(duplicate => {
            avgSimilarity += duplicate.similarity;
            if (duplicate.similarity >= 0.8) highSimilarityCount++;
            if (Math.min(duplicate.cond_prob_1_given_2, duplicate.cond_prob_2_given_1) <= 0.1) {
                lowCondProbCount++;
            }
            if (duplicate.co_occurrences === 0) {
                zeroCoOccurrences++;
            }
            if (duplicate.co_occurrences > 1) {
                multipleCoOccurrences++;
            }
        });
        
        avgSimilarity = totalDuplicates > 0 ? avgSimilarity / totalDuplicates : 0;
        
        // Parent summary
        const parentSummary = Object.keys(this.parentGroups).map(parent => {
            return {
                parent,
                count: this.parentGroups[parent].length
            };
        }).sort((a, b) => b.count - a.count);
        
        // Create text report
        let report = `Duplicate Analysis Report - Level ${this.currentLevel}\n`;
        report += `=======================================\n\n`;
        report += `Generated: ${new Date().toLocaleString()}\n\n`;
        report += `Summary Statistics:\n`;
        report += `------------------\n`;
        report += `Total Potential Duplicates: ${totalDuplicates}\n`;
        report += `Unique Parent Categories: ${uniqueParents}\n`;
        report += `Average Similarity: ${avgSimilarity.toFixed(2)}\n`;
        report += `High Similarity Pairs (≥0.8): ${highSimilarityCount} (${((highSimilarityCount / totalDuplicates) * 100).toFixed(1)}%)\n`;
        report += `Low Conditional Probability Pairs (≤0.1): ${lowCondProbCount} (${((lowCondProbCount / totalDuplicates) * 100).toFixed(1)}%)\n`;
        report += `Zero Co-occurrence Pairs: ${zeroCoOccurrences} (${((zeroCoOccurrences / totalDuplicates) * 100).toFixed(1)}%)\n`;
        report += `Multiple Co-occurrence Pairs: ${multipleCoOccurrences} (${((multipleCoOccurrences / totalDuplicates) * 100).toFixed(1)}%)\n\n`;
        
        report += `Top 10 Parents by Duplicate Count:\n`;
        report += `------------------------------\n`;
        parentSummary.slice(0, 10).forEach((parent, index) => {
            report += `${index + 1}. ${parent.parent}: ${parent.count} pairs\n`;
        });
        
        // Create download link
        const blob = new Blob([report], { type: 'text/plain;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.setAttribute('href', url);
        link.setAttribute('download', `duplicate_analysis_summary_lv${this.currentLevel}.txt`);
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
    
    showErrorMessage(message) {
        // Display error message
        const containers = [
            document.getElementById('overview-metrics'),
            document.getElementById('duplicates-table-body'),
            document.getElementById('parent-groups-container'),
            document.getElementById('network-visualization')
        ];
        
        containers.forEach(container => {
            if (container) {
                container.innerHTML = `<div class="alert alert-danger">${message}</div>`;
            }
        });
    }
}

// Initialize the duplicate analysis module when the page loads
document.addEventListener('DOMContentLoaded', function() {
    const duplicateAnalysis = new DuplicateAnalysis();
    duplicateAnalysis.init();
}); 