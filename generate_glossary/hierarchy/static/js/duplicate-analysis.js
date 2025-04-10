// Duplicate Analysis Module
class DuplicateAnalysis {
    constructor() {
        this.duplicateData = null;
        this.filteredDuplicates = [];
        this.allDuplicates = [];
        this.currentLevel = 2; // Default to level 2
        this.parentGroups = {};
        this.dataLoaded = false;
        this.selectedDuplicate = null;
        
        // Filtering and sorting state
        this.sortBy = 'similarity';
        this.sortOrder = 'desc';
        this.filterParent = 'all';
        this.similarityRange = { min: 0, max: 1 };
        this.coOccurrencesRange = { min: 0, max: Number.MAX_SAFE_INTEGER };
        this.searchTerm = '';
        
        // Evaluation data
        this.evaluations = {};
        this.filteredEvaluations = [];
        this.evaluationSearchTerm = '';
        this.evaluationFilters = {
            decision: 'all',
            relationship: 'all',
            parent: 'all'
        };
    }
    
    init() {
        // Initialize event listeners
        this.initEventListeners();
        
        // Load duplicate analysis data
        this.loadDuplicateData();
        
        // Load saved evaluations
        this.loadSavedEvaluations();
    }
    
    initEventListeners() {
        // Level selector
        const levelSelector = document.getElementById('level-selector');
        if (levelSelector) {
            levelSelector.addEventListener('change', (event) => {
                this.currentLevel = parseInt(event.target.value);
                // Just load the duplicate data, which will also load the evaluations
                this.loadDuplicateData();
            });
        }
        
        // Refresh button
        const refreshBtn = document.getElementById('refresh-btn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => {
                // Just load the duplicate data, which will also load the evaluations
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
        
        // Evaluation controls
        document.addEventListener('click', (event) => {
            if (event.target.classList.contains('save-evaluation-btn')) {
                this.saveCurrentEvaluation();
            } else if (event.target.classList.contains('prev-duplicate-btn')) {
                this.navigateToDuplicate('prev');
            } else if (event.target.classList.contains('next-duplicate-btn')) {
                this.navigateToDuplicate('next');
            }
        });
        
        // Export evaluations
        const exportEvaluationsBtn = document.getElementById('export-evaluations-btn');
        if (exportEvaluationsBtn) {
            exportEvaluationsBtn.addEventListener('click', () => {
                this.exportEvaluations();
            });
        }
        
        // Import evaluations
        const importEvaluationsBtn = document.getElementById('import-evaluations-btn');
        const importFileInput = document.getElementById('import-evaluations-file');
        if (importEvaluationsBtn && importFileInput) {
            importEvaluationsBtn.addEventListener('click', () => {
                importFileInput.click();
            });
            
            importFileInput.addEventListener('change', (event) => {
                if (event.target.files.length > 0) {
                    this.importEvaluations(event.target.files[0]);
                }
            });
        }
        
        // Clear evaluations
        const clearEvaluationsBtn = document.getElementById('clear-evaluations-btn');
        if (clearEvaluationsBtn) {
            clearEvaluationsBtn.addEventListener('click', () => {
                if (confirm('Are you sure you want to clear all evaluations? This action cannot be undone.')) {
                    this.clearEvaluations();
                }
            });
        }
        
        // Evaluation search
        const evaluationSearch = document.getElementById('evaluation-search');
        if (evaluationSearch) {
            evaluationSearch.addEventListener('input', (event) => {
                this.evaluationSearchTerm = event.target.value;
                this.filterEvaluations();
            });
        }
        
        // Evaluation filter toggle
        const evaluationFilterBtn = document.getElementById('evaluation-filter-btn');
        if (evaluationFilterBtn) {
            evaluationFilterBtn.addEventListener('click', () => {
                const filterContainer = document.getElementById('evaluation-filters');
                if (filterContainer) {
                    filterContainer.style.display = filterContainer.style.display === 'none' ? 'block' : 'none';
                }
            });
        }
        
        // Apply evaluation filters
        const applyEvaluationFiltersBtn = document.getElementById('apply-evaluation-filters-btn');
        if (applyEvaluationFiltersBtn) {
            applyEvaluationFiltersBtn.addEventListener('click', () => {
                this.evaluationFilters.decision = document.getElementById('evaluation-decision-filter').value;
                this.evaluationFilters.relationship = document.getElementById('evaluation-relationship-filter').value;
                this.evaluationFilters.parent = document.getElementById('evaluation-parent-filter').value;
                this.filterEvaluations();
            });
        }
        
        // Reset evaluation filters
        const resetEvaluationFiltersBtn = document.getElementById('reset-evaluation-filters-btn');
        if (resetEvaluationFiltersBtn) {
            resetEvaluationFiltersBtn.addEventListener('click', () => {
                document.getElementById('evaluation-decision-filter').value = 'all';
                document.getElementById('evaluation-relationship-filter').value = 'all';
                document.getElementById('evaluation-parent-filter').value = 'all';
                
                this.evaluationFilters = {
                    decision: 'all',
                    relationship: 'all',
                    parent: 'all'
                };
                
                this.filterEvaluations();
            });
        }
        
        // Search filter
        const searchInput = document.getElementById('duplicate-search');
        if (searchInput) {
            searchInput.addEventListener('input', (event) => {
                this.searchTerm = event.target.value;
                this.applyFiltersAndSort();
            });
        }
        
        // Sort controls
        const sortBySelect = document.getElementById('sort-by');
        if (sortBySelect) {
            sortBySelect.addEventListener('change', (event) => {
                this.sortBy = event.target.value;
                this.applyFiltersAndSort();
            });
        }
        
        const sortOrderSelect = document.getElementById('sort-order');
        if (sortOrderSelect) {
            sortOrderSelect.addEventListener('change', (event) => {
                this.sortOrder = event.target.value;
                this.applyFiltersAndSort();
            });
        }
        
        // Parent filter
        const parentFilter = document.getElementById('parent-filter');
        if (parentFilter) {
            parentFilter.addEventListener('change', (event) => {
                this.filterParent = event.target.value;
                this.applyFiltersAndSort();
            });
        }
        
        // Range filters
        const similarityMinInput = document.getElementById('similarity-min');
        const similarityMaxInput = document.getElementById('similarity-max');
        if (similarityMinInput && similarityMaxInput) {
            similarityMinInput.addEventListener('change', (event) => {
                this.similarityRange.min = parseFloat(event.target.value) || 0;
            });
            
            similarityMaxInput.addEventListener('change', (event) => {
                this.similarityRange.max = parseFloat(event.target.value) || 1;
            });
        }
        
        const coOccurrencesMinInput = document.getElementById('co-occurrences-min');
        const coOccurrencesMaxInput = document.getElementById('co-occurrences-max');
        if (coOccurrencesMinInput && coOccurrencesMaxInput) {
            coOccurrencesMinInput.addEventListener('change', (event) => {
                this.coOccurrencesRange.min = parseInt(event.target.value) || 0;
            });
            
            coOccurrencesMaxInput.addEventListener('change', (event) => {
                const value = event.target.value.trim();
                this.coOccurrencesRange.max = value ? parseInt(value) : Number.MAX_SAFE_INTEGER;
            });
        }
        
        // Apply filters button
        const applyFiltersBtn = document.getElementById('apply-filters-btn');
        if (applyFiltersBtn) {
            applyFiltersBtn.addEventListener('click', () => {
                this.applyFiltersAndSort();
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
                const tabId = tab.getAttribute('href');
                
                if (tabId === '#network-tab') {
                    // Delay to allow the tab to become visible
                    setTimeout(() => {
                        if (parentNetworkSelector && parentNetworkSelector.value) {
                            this.renderNetworkVisualization(parentNetworkSelector.value);
                        }
                    }, 100);
                } else if (tabId === '#duplicates-list-tab') {
                    // Refresh the duplicate list with evaluation indicators when tab is clicked
                    setTimeout(() => {
                        this.refreshDuplicatesUI();
                    }, 100);
                } else if (tabId === '#evaluations-tab') {
                    // Refresh the evaluations list when tab is clicked
                    setTimeout(() => {
                        this.updateEvaluationUI();
                    }, 100);
                }
            });
        });
        
        // Bootstrap tab events for better tab switching detection
        const tabElements = document.querySelectorAll('a[data-bs-toggle="pill"]');
        tabElements.forEach(tabElement => {
            tabElement.addEventListener('shown.bs.tab', (event) => {
                const targetTabId = event.target.getAttribute('href');
                
                if (targetTabId === '#duplicates-list-tab') {
                    this.refreshDuplicatesUI();
                } else if (targetTabId === '#evaluations-tab') {
                    this.updateEvaluationUI();
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
                
                // Load saved evaluations for this level
                this.loadSavedEvaluations();
                
                // Render overview
                this.renderOverview();
                
                // Populate parent network selector
                this.populateParentSelector();
                
                // Apply filters and sort
                this.applyFiltersAndSort();
                
                // Render parent groups
                this.renderParentGroups();
                
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
        
        // Convert from object to array for easier processing
        this.allDuplicates = [];
        for (const key in this.duplicateData.parent_groups) {
            const duplicatesInGroup = this.duplicateData.parent_groups[key];
            this.allDuplicates = this.allDuplicates.concat(duplicatesInGroup);
        }
        
        // Show all duplicates without filtering initially
        this.filteredDuplicates = [...this.allDuplicates];
        
        // Populate parent filter with options
        this.populateParentFilter();
        
        // Set max value for co-occurrences filter
        const maxCoOccurrences = Math.max(...this.allDuplicates.map(d => d.co_occurrences));
        const coOccurrencesMaxInput = document.getElementById('co-occurrences-max');
        if (coOccurrencesMaxInput) {
            coOccurrencesMaxInput.value = maxCoOccurrences;
            this.coOccurrencesRange.max = maxCoOccurrences;
        }
    }
    
    populateParentFilter() {
        const parentFilter = document.getElementById('parent-filter');
        if (!parentFilter) return;
        
        // Save current selection
        const currentValue = parentFilter.value;
        
        // Clear existing options except the "All" option
        while (parentFilter.options.length > 1) {
            parentFilter.remove(1);
        }
        
        // Get unique parents
        const parents = [...new Set(this.allDuplicates.map(d => d.parent))].sort();
        
        // Add parent options
        parents.forEach(parent => {
            const option = document.createElement('option');
            option.value = parent;
            option.textContent = parent;
            parentFilter.appendChild(option);
        });
        
        // Restore selection if possible
        if (parents.includes(currentValue)) {
            parentFilter.value = currentValue;
        }
    }
    
    applyFiltersAndSort() {
        if (!this.allDuplicates || this.allDuplicates.length === 0) return;
        
        // Apply filters
        this.filteredDuplicates = this.allDuplicates.filter(duplicate => {
            // Search term filter
            const searchMatch = !this.searchTerm || 
                duplicate.term1.toLowerCase().includes(this.searchTerm.toLowerCase()) ||
                duplicate.term2.toLowerCase().includes(this.searchTerm.toLowerCase()) ||
                duplicate.parent.toLowerCase().includes(this.searchTerm.toLowerCase());
            
            // Parent filter
            const parentMatch = this.filterParent === 'all' || duplicate.parent === this.filterParent;
            
            // Similarity range filter
            const similarityMatch = 
                duplicate.similarity >= this.similarityRange.min && 
                duplicate.similarity <= this.similarityRange.max;
            
            // Co-occurrences range filter
            const coOccurrencesMatch = 
                duplicate.co_occurrences >= this.coOccurrencesRange.min && 
                duplicate.co_occurrences <= this.coOccurrencesRange.max;
            
            return searchMatch && parentMatch && similarityMatch && coOccurrencesMatch;
        });
        
        // Apply sorting
        this.filteredDuplicates.sort((a, b) => {
            let valueA, valueB;
            
            // Get the values to compare based on sort field
            switch (this.sortBy) {
                case 'similarity':
                    valueA = a.similarity;
                    valueB = b.similarity;
                    break;
                case 'cond_prob':
                    valueA = Math.min(a.cond_prob_1_given_2, a.cond_prob_2_given_1);
                    valueB = Math.min(b.cond_prob_1_given_2, b.cond_prob_2_given_1);
                    break;
                case 'mutual_info':
                    valueA = a.mutual_information;
                    valueB = b.mutual_information;
                    break;
                case 'co_occurrences':
                    valueA = a.co_occurrences;
                    valueB = b.co_occurrences;
                    break;
                case 'term1':
                    valueA = a.term1.toLowerCase();
                    valueB = b.term1.toLowerCase();
                    break;
                case 'term2':
                    valueA = a.term2.toLowerCase();
                    valueB = b.term2.toLowerCase();
                    break;
                case 'parent':
                    valueA = a.parent.toLowerCase();
                    valueB = b.parent.toLowerCase();
                    break;
                default:
                    valueA = a.similarity;
                    valueB = b.similarity;
            }
            
            // Handle string comparison separately from number comparison
            if (typeof valueA === 'string' && typeof valueB === 'string') {
                return this.sortOrder === 'asc' 
                    ? valueA.localeCompare(valueB) 
                    : valueB.localeCompare(valueA);
            } else {
                // For numbers, NaN should be handled specially
                if (isNaN(valueA)) valueA = this.sortOrder === 'asc' ? -Infinity : Infinity;
                if (isNaN(valueB)) valueB = this.sortOrder === 'asc' ? -Infinity : Infinity;
                
                return this.sortOrder === 'asc' ? valueA - valueB : valueB - valueA;
            }
        });
        
        // Update the UI
        this.renderDuplicatesList();
        
        // Apply evaluation indicators after rendering list
        this.updateEvaluationIndicators();
    }
    
    refreshDuplicatesUI() {
        // Check if duplicates list is visible
        const duplicatesTab = document.getElementById('duplicates-list-tab');
        if (duplicatesTab && duplicatesTab.classList.contains('active')) {
            // Apply current filters and sorting
            this.applyFiltersAndSort();
        } else {
            // Just update the evaluation indicators
            this.updateEvaluationIndicators();
        }
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
                    <div class="text-muted">Level ${this.currentLevel} - All entries shown</div>
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
                            which can be a strong indicator they are being used interchangeably as duplicate concepts in different contexts.</p>
                            
                            <p><strong>${multipleCoOccurrences}</strong> pairs have multiple co-occurrences, suggesting they might be related but distinct concepts that appear together in similar contexts.</p>
                            
                            <p class="mt-3"><strong>Recommendation:</strong> Focus on pairs with high semantic similarity (≥0.7) 
                            combined with low conditional probability (≤0.3) and low mutual information (≤0.1) as the strongest candidates for duplicate consolidation.</p>
                            
                            <p><strong>Note:</strong> All duplicate pairs are displayed. No filtering is applied.</p>
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
        
        // Populate the evaluation form with existing data if available
        this.populateEvaluationForm();
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
        // This method is simplified since no filtering is occurring
        if (!this.duplicateData) return;
        
        const overviewTab = document.getElementById('overview-tab');
        if (overviewTab) {
            const count = this.allDuplicates.length;
            
            const metricsContainer = document.getElementById('overview-metrics');
            if (metricsContainer) {
                const metricCards = metricsContainer.querySelectorAll('.metric-card');
                if (metricCards.length > 0) {
                    const firstCard = metricCards[0];
                    const valueElement = firstCard.querySelector('.value');
                    
                    if (valueElement) {
                        valueElement.innerHTML = `${count} <small class="text-muted">All entries shown</small>`;
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
    
    loadSavedEvaluations() {
        // First try to load from localStorage
        const savedEvaluations = localStorage.getItem(`duplicate_evaluations_lv${this.currentLevel}`);
        if (savedEvaluations) {
            try {
                this.evaluations = JSON.parse(savedEvaluations);
                console.log(`Loaded ${Object.keys(this.evaluations).length} saved evaluations for level ${this.currentLevel}`);
                
                // Update UI elements with evaluation data
                this.updateEvaluationUI();
                
                // Make sure the duplicate pairs list shows evaluation indicators
                this.refreshDuplicatesUI();
                return; // Successfully loaded from localStorage
            } catch (error) {
                console.error('Error loading saved evaluations from localStorage:', error);
                this.evaluations = {};
            }
        } else {
            console.log(`No saved evaluations found in localStorage for level ${this.currentLevel}`);
            this.evaluations = {};
        }
        
        // If we reach here, no evaluations were loaded from localStorage
        // Try to find and load the latest evaluation file
        this.findAndLoadLatestEvaluationFile();
    }
    
    findAndLoadLatestEvaluationFile() {
        console.log('Attempting to find latest evaluation file...');
        
        // Make a fetch request to find latest evaluation file
        fetch(`/api/find_latest_evaluation?level=${this.currentLevel}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Server returned status ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                if (data.success && data.filePath) {
                    console.log(`Found latest evaluation file: ${data.filePath}`);
                    
                    // Now fetch the file content
                    return fetch(data.filePath)
                        .then(response => {
                            if (!response.ok) {
                                throw new Error(`Server returned status ${response.status}`);
                            }
                            
                            // Show a notification that we found a file
                            const fileType = data.filePath.toLowerCase().endsWith('.json') ? 'JSON' : 'CSV';
                            this.showNotification(`Loading ${fileType} evaluation file from ${new Date(data.timestamp).toLocaleDateString()}`);
                            
                            return response;
                        });
                } else {
                    console.log('No evaluation files found: ', data.message || 'Unknown error');
                    // Just show UI with empty evaluations - no error notification needed
                    this.updateEvaluationUI();
                    throw new Error('No evaluation files found');
                }
            })
            .then(response => {
                // Determine file type based on extension
                const isJson = response.url.toLowerCase().endsWith('.json');
                
                if (isJson) {
                    return response.json().then(jsonData => {
                        this.importFromJSON(JSON.stringify(jsonData), false); // false = don't show confirmation
                        console.log(`Successfully loaded evaluation data from ${response.url}`);
                        
                        // Show success notification
                        const evaluationCount = Object.keys(jsonData).length;
                        this.showNotification(`Successfully loaded ${evaluationCount} evaluations`);
                    });
                } else {
                    return response.text().then(csvData => {
                        this.importFromCSV(csvData, false); // false = don't show confirmation
                        console.log(`Successfully loaded evaluation data from ${response.url}`);
                        
                        // Show success notification based on rows loaded
                        const rowCount = csvData.split('\n').length - 1; // subtract header row
                        this.showNotification(`Successfully loaded ${rowCount} evaluations from CSV`);
                    });
                }
            })
            .catch(error => {
                console.error('Error finding or loading latest evaluation file:', error);
                // Make sure UI is updated even if we failed to load
                this.updateEvaluationUI();
            });
    }
    
    showNotification(message, type = 'info', duration = 3000) {
        const container = document.createElement('div');
        container.className = `alert alert-${type} notification-toast`;
        container.style.position = 'fixed';
        container.style.top = '20px';
        container.style.right = '20px';
        container.style.zIndex = '1050';
        container.style.minWidth = '300px';
        container.style.boxShadow = '0 4px 8px rgba(0,0,0,0.2)';
        container.style.transition = 'opacity 0.5s';
        
        container.innerHTML = `
            <div class="d-flex align-items-center">
                <i class="bi bi-info-circle me-2"></i>
                <span>${message}</span>
            </div>
        `;
        
        document.body.appendChild(container);
        
        setTimeout(() => {
            container.style.opacity = '0';
            setTimeout(() => {
                document.body.removeChild(container);
            }, 500);
        }, duration);
    }
    
    saveEvaluations() {
        localStorage.setItem(`duplicate_evaluations_lv${this.currentLevel}`, JSON.stringify(this.evaluations));
        console.log(`Saved ${Object.keys(this.evaluations).length} evaluations to localStorage`);
    }
    
    // Evaluation methods
    
    saveCurrentEvaluation() {
        if (!this.selectedDuplicate) {
            alert('No duplicate pair selected');
            return;
        }
        
        // Generate a unique key for this duplicate pair
        const key = `${this.selectedDuplicate.term1}___${this.selectedDuplicate.term2}`;
        
        // Get form values
        const decision = document.querySelector('input[name="decision"]:checked')?.value;
        if (!decision) {
            alert('Please select an evaluation decision');
            return;
        }
        
        const relationship = document.getElementById('relationship-type').value;
        const canonical = document.querySelector('input[name="canonical"]:checked')?.value;
        const notes = document.getElementById('evaluation-notes').value;
        
        // Check if this is an update to an existing evaluation
        const isUpdate = key in this.evaluations;
        const previousEvaluation = isUpdate ? {...this.evaluations[key]} : null;
        
        // Create evaluation object
        const evaluation = {
            term1: this.selectedDuplicate.term1,
            term2: this.selectedDuplicate.term2,
            parent: this.selectedDuplicate.parent,
            similarity: this.selectedDuplicate.similarity,
            cond_prob_1_given_2: this.selectedDuplicate.cond_prob_1_given_2,
            cond_prob_2_given_1: this.selectedDuplicate.cond_prob_2_given_1,
            mutual_information: this.selectedDuplicate.mutual_information,
            co_occurrences: this.selectedDuplicate.co_occurrences,
            decision: decision,
            relationship: relationship,
            canonical: canonical,
            notes: notes,
            timestamp: new Date().toISOString()
        };
        
        // For updates, keep track of the history
        if (isUpdate) {
            evaluation.updated = true;
            evaluation.original_timestamp = previousEvaluation.timestamp;
            
            if (!previousEvaluation.updates) {
                evaluation.updates = [previousEvaluation];
            } else {
                evaluation.updates = [...previousEvaluation.updates, previousEvaluation];
            }
        }
        
        // Save to evaluations object
        this.evaluations[key] = evaluation;
        
        // Save to localStorage
        this.saveEvaluations();
        
        // Provide visual feedback
        const saveBtn = document.querySelector('.save-evaluation-btn');
        if (saveBtn) {
            const originalText = saveBtn.textContent;
            
            if (isUpdate) {
                saveBtn.textContent = 'Updated!';
                saveBtn.classList.add('btn-success');
                saveBtn.classList.remove('btn-warning', 'btn-primary');
                
                // Update the note to show success message
                const existingNote = document.querySelector('.evaluation-exists-note');
                if (existingNote) {
                    existingNote.innerHTML = '<i class="bi bi-check-circle"></i> Evaluation successfully updated.';
                    existingNote.className = 'evaluation-exists-note alert alert-success mb-3';
                    
                    // Change back to warning after some time
                    setTimeout(() => {
                        existingNote.innerHTML = '<i class="bi bi-exclamation-triangle"></i> You are editing an existing evaluation. Save to update your decision.';
                        existingNote.className = 'evaluation-exists-note alert alert-warning mb-3';
                    }, 2000);
                }
            } else {
                saveBtn.textContent = 'Saved!';
                saveBtn.classList.add('btn-success');
                saveBtn.classList.remove('btn-primary');
            }
            
            setTimeout(() => {
                if (isUpdate) {
                    saveBtn.textContent = 'Update Evaluation';
                    saveBtn.classList.remove('btn-success');
                    saveBtn.classList.add('btn-warning');
                } else {
                    saveBtn.textContent = originalText;
                    saveBtn.classList.remove('btn-success');
                    saveBtn.classList.add('btn-primary');
                }
            }, 1500);
        }
        
        console.log(`Saved evaluation for ${key} (${isUpdate ? 'updated' : 'new'})`);
        
        // Update evaluations tab
        this.updateEvaluationUI();
        
        // Update evaluation indicators in the duplicates list
        this.updateEvaluationIndicators();
        
        // If we're in the Evaluations tab, we need to refresh the table
        const evaluationsTab = document.getElementById('evaluations-tab');
        if (evaluationsTab && evaluationsTab.classList.contains('active')) {
            this.filterEvaluations();
        }
    }
    
    navigateToDuplicate(direction) {
        if (!this.selectedDuplicate || !this.filteredDuplicates.length) return;
        
        // Find current index
        const currentIndex = this.filteredDuplicates.findIndex(
            d => d.term1 === this.selectedDuplicate.term1 && d.term2 === this.selectedDuplicate.term2
        );
        
        if (currentIndex === -1) return;
        
        let newIndex;
        if (direction === 'prev') {
            newIndex = currentIndex - 1;
            if (newIndex < 0) newIndex = this.filteredDuplicates.length - 1;
        } else {
            newIndex = currentIndex + 1;
            if (newIndex >= this.filteredDuplicates.length) newIndex = 0;
        }
        
        // Show the next/previous duplicate
        this.showDuplicateDetails(this.filteredDuplicates[newIndex]);
    }
    
    updateEvaluationUI() {
        // Update summary statistics
        const evaluations = Object.values(this.evaluations);
        
        // Filter evaluations based on current level
        const currentLevelEvaluations = evaluations.filter(e => {
            // Find the duplicate in allDuplicates to check if it's in the current level
            return this.allDuplicates.some(d => 
                d.term1 === e.term1 && d.term2 === e.term2
            );
        });
        
        console.log(`Found ${currentLevelEvaluations.length} evaluations for level ${this.currentLevel} out of ${evaluations.length} total evaluations`);
        
        // Check if we have any evaluations
        const hasEvaluations = currentLevelEvaluations.length > 0;
        
        // Show/hide appropriate elements
        const summaryEmptyEl = document.getElementById('evaluation-summary-empty');
        const summaryEl = document.getElementById('evaluation-summary');
        
        if (summaryEmptyEl) {
            summaryEmptyEl.style.display = hasEvaluations ? 'none' : 'block';
        }
        
        if (summaryEl) {
            summaryEl.style.display = hasEvaluations ? 'block' : 'none';
        }
        
        if (!hasEvaluations) return;
        
        // Count decisions by type
        const decisionCounts = {
            true_duplicate: 0,
            related_terms: 0,
            hierarchical: 0,
            unrelated: 0,
            unsure: 0
        };
        
        currentLevelEvaluations.forEach(e => {
            if (e.decision && decisionCounts[e.decision] !== undefined) {
                decisionCounts[e.decision]++;
            }
        });
        
        // Update statistics
        document.getElementById('total-evaluated').textContent = currentLevelEvaluations.length;
        document.getElementById('percent-evaluated').textContent = 
            `${((currentLevelEvaluations.length / this.allDuplicates.length) * 100).toFixed(1)}%`;
        
        document.getElementById('true-duplicate-count').textContent = decisionCounts.true_duplicate;
        document.getElementById('related-terms-count').textContent = decisionCounts.related_terms;
        document.getElementById('hierarchical-count').textContent = decisionCounts.hierarchical;
        document.getElementById('unrelated-count').textContent = decisionCounts.unrelated;
        document.getElementById('unsure-count').textContent = decisionCounts.unsure;
        
        // Populate evaluation parent filter with parents from current level only
        this.populateEvaluationParentFilter();
        
        // Count relationships
        const relationshipCounts = {};
        currentLevelEvaluations.forEach(e => {
            if (e.relationship) {
                if (!relationshipCounts[e.relationship]) {
                    relationshipCounts[e.relationship] = 0;
                }
                relationshipCounts[e.relationship]++;
            }
        });
        
        // Display relationship stats
        const relationshipStatsEl = document.getElementById('relationship-stats');
        if (relationshipStatsEl) {
            let html = '<ul class="list-group">';
            
            const relationships = Object.entries(relationshipCounts)
                .sort((a, b) => b[1] - a[1]);
            
            relationships.forEach(([type, count]) => {
                const typeName = this.getRelationshipTypeName(type);
                html += `
                    <li class="list-group-item d-flex justify-content-between align-items-center">
                        ${typeName}
                        <span class="badge bg-primary rounded-pill">${count}</span>
                    </li>
                `;
            });
            
            html += '</ul>';
            relationshipStatsEl.innerHTML = html;
        }
        
        // Draw chart
        this.renderEvaluationChart(decisionCounts);
        
        // Filter and render evaluations table
        this.filterEvaluations();
        
        // Update evaluation indicators
        this.updateEvaluationIndicators();
    }
    
    populateEvaluationParentFilter() {
        const parentFilter = document.getElementById('evaluation-parent-filter');
        if (!parentFilter) return;
        
        // Save current selection
        const currentValue = parentFilter.value;
        
        // Clear existing options except the "All" option
        while (parentFilter.options.length > 1) {
            parentFilter.remove(1);
        }
        
        // Get unique parents from evaluations in the current level
        const parents = [];
        Object.values(this.evaluations).forEach(e => {
            // Check if this evaluation is for a term in the current level
            const isCurrentLevel = this.allDuplicates.some(d => 
                d.term1 === e.term1 && d.term2 === e.term2
            );
            
            if (isCurrentLevel && e.parent && !parents.includes(e.parent)) {
                parents.push(e.parent);
            }
        });
        
        // Sort parents alphabetically
        parents.sort();
        
        // Add parent options
        parents.forEach(parent => {
            const option = document.createElement('option');
            option.value = parent;
            option.textContent = parent;
            parentFilter.appendChild(option);
        });
        
        // Restore selection if possible
        if (parents.includes(currentValue)) {
            parentFilter.value = currentValue;
        } else {
            parentFilter.value = 'all';
        }
    }
    
    getRelationshipTypeName(type) {
        const relationshipTypes = {
            "exact_synonym": "Exact Synonym",
            "spelling_variation": "Spelling Variation",
            "abbreviation": "Abbreviation/Full Form",
            "term_variation": "Terminological Variation",
            "broader_term": "Broader Term",
            "narrower_term": "Narrower Term",
            "related_concept": "Related Concept",
            "other": "Other"
        };
        
        return relationshipTypes[type] || type;
    }
    
    getDecisionName(decision) {
        const decisionTypes = {
            "true_duplicate": "True Duplicate",
            "related_terms": "Related Terms",
            "hierarchical": "Hierarchical Relationship",
            "unrelated": "Unrelated Terms",
            "unsure": "Need More Information"
        };
        
        return decisionTypes[decision] || decision;
    }
    
    renderEvaluationChart(decisionCounts) {
        const chartElement = document.getElementById('evaluation-chart');
        if (!chartElement) return;
        
        // Clear existing chart
        chartElement.innerHTML = '';
        
        // Set up dimensions
        const width = chartElement.clientWidth;
        const height = chartElement.clientHeight;
        const margin = { top: 10, right: 20, bottom: 30, left: 40 };
        const chartWidth = width - margin.left - margin.right;
        const chartHeight = height - margin.top - margin.bottom;
        
        // Create SVG
        const svg = d3.select('#evaluation-chart')
            .append('svg')
            .attr('width', width)
            .attr('height', height)
            .append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);
        
        // Prepare data
        const data = [
            { decision: 'True Duplicate', count: decisionCounts.true_duplicate, color: '#28a745', key: 'TD' },
            { decision: 'Related Terms', count: decisionCounts.related_terms, color: '#6c757d', key: 'RT' },
            { decision: 'Hierarchical', count: decisionCounts.hierarchical, color: '#17a2b8', key: 'HR' },
            { decision: 'Unrelated', count: decisionCounts.unrelated, color: '#dc3545', key: 'UR' },
            { decision: 'Need Info', count: decisionCounts.unsure, color: '#ffc107', key: 'NI' }
        ];
        
        // X scale - use keys instead of full labels for the axis
        const x = d3.scaleBand()
            .domain(data.map(d => d.key))
            .range([0, chartWidth])
            .padding(0.3);
        
        // Y scale
        const y = d3.scaleLinear()
            .domain([0, d3.max(data, d => d.count) || 1])
            .range([chartHeight, 0]);
        
        // Add X axis with shortened labels
        svg.append('g')
            .attr('transform', `translate(0,${chartHeight})`)
            .call(d3.axisBottom(x))
            .selectAll("text")
            .style("text-anchor", "middle");
        
        // Add Y axis
        svg.append('g')
            .call(d3.axisLeft(y).ticks(Math.min(5, d3.max(data, d => d.count))));
        
        // Bars
        svg.selectAll('.bar')
            .data(data)
            .enter()
            .append('rect')
            .attr('class', 'bar')
            .attr('x', d => x(d.key))
            .attr('y', d => y(d.count))
            .attr('width', x.bandwidth())
            .attr('height', d => chartHeight - y(d.count))
            .attr('fill', d => d.color);
        
        // Bar labels
        svg.selectAll('.label')
            .data(data)
            .enter()
            .append('text')
            .attr('class', 'label')
            .attr('x', d => x(d.key) + x.bandwidth() / 2)
            .attr('y', d => Math.max(y(d.count) - 5, 15))
            .attr('text-anchor', 'middle')
            .text(d => d.count > 0 ? d.count : '');
        
        // Create a legend below the chart
        const legendContainer = d3.select('#evaluation-chart')
            .append('div')
            .attr('class', 'chart-legend');
        
        // Add legend items
        data.forEach(d => {
            const legendItem = legendContainer.append('span')
                .style('margin-right', '10px');
                
            legendItem.append('span')
                .attr('class', 'legend-color')
                .style('background-color', d.color);
                
            legendItem.append('span')
                .text(d.decision);
        });
    }
    
    filterEvaluations() {
        // Convert evaluations object to array
        const evaluationsArray = Object.values(this.evaluations);
        
        // First, filter by current level
        const currentLevelEvaluations = evaluationsArray.filter(e => {
            return this.allDuplicates.some(d => 
                d.term1 === e.term1 && d.term2 === e.term2
            );
        });
        
        // Apply search term filter
        this.filteredEvaluations = currentLevelEvaluations.filter(e => {
            const searchMatch = !this.evaluationSearchTerm || 
                e.term1.toLowerCase().includes(this.evaluationSearchTerm.toLowerCase()) ||
                e.term2.toLowerCase().includes(this.evaluationSearchTerm.toLowerCase()) ||
                e.parent.toLowerCase().includes(this.evaluationSearchTerm.toLowerCase());
            
            // Apply filters
            const decisionMatch = this.evaluationFilters.decision === 'all' || 
                                 e.decision === this.evaluationFilters.decision;
            
            const relationshipMatch = this.evaluationFilters.relationship === 'all' || 
                                     e.relationship === this.evaluationFilters.relationship;
            
            const parentMatch = this.evaluationFilters.parent === 'all' || 
                               e.parent === this.evaluationFilters.parent;
            
            return searchMatch && decisionMatch && relationshipMatch && parentMatch;
        });
        
        console.log(`Filtered to ${this.filteredEvaluations.length} evaluations out of ${currentLevelEvaluations.length} for level ${this.currentLevel}`);
        
        // Render filtered evaluations
        this.renderEvaluationsTable();
    }
    
    renderEvaluationsTable() {
        const tableBody = document.getElementById('evaluations-table-body');
        if (!tableBody) return;
        
        if (this.filteredEvaluations.length === 0) {
            tableBody.innerHTML = `
                <tr>
                    <td colspan="7" class="text-center">No evaluations found matching the current filters</td>
                </tr>
            `;
            return;
        }
        
        // Sort evaluations by timestamp (most recent first)
        const sortedEvaluations = [...this.filteredEvaluations].sort((a, b) => {
            return new Date(b.timestamp) - new Date(a.timestamp);
        });
        
        let html = '';
        sortedEvaluations.forEach(evaluation => {
            const decisionName = this.getDecisionName(evaluation.decision);
            const relationshipName = evaluation.relationship ? this.getRelationshipTypeName(evaluation.relationship) : '';
            
            let canonicalTerm = '';
            if (evaluation.canonical === 'term1') {
                canonicalTerm = evaluation.term1;
            } else if (evaluation.canonical === 'term2') {
                canonicalTerm = evaluation.term2;
            } else if (evaluation.canonical === 'neither') {
                canonicalTerm = '(Keep Both)';
            }
            
            // Determine badge class for decision
            let badgeClass = 'bg-secondary';
            if (evaluation.decision === 'true_duplicate') badgeClass = 'bg-success';
            else if (evaluation.decision === 'hierarchical') badgeClass = 'bg-info';
            else if (evaluation.decision === 'unrelated') badgeClass = 'bg-danger';
            else if (evaluation.decision === 'unsure') badgeClass = 'bg-warning';
            
            html += `
                <tr data-key="${evaluation.term1}___${evaluation.term2}">
                    <td>${evaluation.term1}</td>
                    <td>${evaluation.term2}</td>
                    <td>${evaluation.parent}</td>
                    <td><span class="badge ${badgeClass}">${decisionName}</span></td>
                    <td>${relationshipName}</td>
                    <td>${canonicalTerm}</td>
                    <td>
                        <div class="btn-group">
                            <button class="btn btn-sm btn-outline-primary view-evaluated-btn" 
                                data-term1="${evaluation.term1}" data-term2="${evaluation.term2}" title="View">
                                <i class="bi bi-eye"></i>
                            </button>
                            <button class="btn btn-sm btn-warning edit-evaluation-btn" 
                                data-term1="${evaluation.term1}" data-term2="${evaluation.term2}" title="Edit">
                                <i class="bi bi-pencil"></i>
                            </button>
                            <button class="btn btn-sm btn-outline-danger delete-evaluation-btn" 
                                data-key="${evaluation.term1}___${evaluation.term2}" title="Delete">
                                <i class="bi bi-trash"></i>
                            </button>
                        </div>
                    </td>
                </tr>
            `;
        });
        
        tableBody.innerHTML = html;
        
        // Add event listeners to buttons
        document.querySelectorAll('.view-evaluated-btn').forEach(button => {
            button.addEventListener('click', (event) => {
                const term1 = event.currentTarget.getAttribute('data-term1');
                const term2 = event.currentTarget.getAttribute('data-term2');
                
                // Find the duplicate in the current list
                const duplicate = this.allDuplicates.find(d => 
                    d.term1 === term1 && d.term2 === term2
                );
                
                if (duplicate) {
                    this.showDuplicateDetails(duplicate);
                    document.querySelector('a[href="#duplicate-details-tab"]').click();
                } else {
                    alert('This duplicate pair is no longer available in the current data');
                }
            });
        });
        
        document.querySelectorAll('.edit-evaluation-btn').forEach(button => {
            button.addEventListener('click', (event) => {
                const term1 = event.currentTarget.getAttribute('data-term1');
                const term2 = event.currentTarget.getAttribute('data-term2');
                
                // Find the duplicate in the current list
                const duplicate = this.allDuplicates.find(d => 
                    d.term1 === term1 && d.term2 === term2
                );
                
                if (duplicate) {
                    this.showDuplicateDetails(duplicate);
                    document.querySelector('a[href="#duplicate-details-tab"]').click();
                    
                    // Highlight the evaluation section
                    const evaluationSection = document.querySelector('.evaluation-section');
                    if (evaluationSection) {
                        evaluationSection.classList.add('highlight-section');
                        setTimeout(() => {
                            evaluationSection.classList.remove('highlight-section');
                        }, 2000);
                    }
                } else {
                    alert('This duplicate pair is no longer available in the current data');
                }
            });
        });
        
        document.querySelectorAll('.delete-evaluation-btn').forEach(button => {
            button.addEventListener('click', (event) => {
                if (confirm('Are you sure you want to delete this evaluation?')) {
                    const key = event.currentTarget.getAttribute('data-key');
                    delete this.evaluations[key];
                    this.saveEvaluations();
                    this.updateEvaluationUI();
                }
            });
        });
    }
    
    updateEvaluationIndicators() {
        // Update the duplicate list table to indicate which pairs have been evaluated
        const tableBody = document.getElementById('duplicates-table-body');
        if (!tableBody) return;
        
        // First, remove any existing evaluation badges and buttons from all rows
        tableBody.querySelectorAll('.evaluated-badge, .change-evaluation-btn').forEach(el => {
            el.remove();
        });
        
        tableBody.querySelectorAll('tr').forEach(row => {
            row.classList.remove('evaluated-row');
        });
        
        // Filter evaluation keys to only those that match duplicates in the current level
        const currentLevelEvalKeys = Object.keys(this.evaluations).filter(key => {
            const [term1, term2] = key.split('___');
            return this.allDuplicates.some(d => 
                d.term1 === term1 && d.term2 === term2
            );
        });
        
        // Now add indicators for evaluations that match the current level
        tableBody.querySelectorAll('tr').forEach(row => {
            const index = row.getAttribute('data-index');
            if (index === null) return;
            
            const duplicate = this.filteredDuplicates[parseInt(index)];
            if (!duplicate) return;
            
            const key = `${duplicate.term1}___${duplicate.term2}`;
            const isEvaluated = currentLevelEvalKeys.includes(key);
            
            // Add or remove evaluated class
            if (isEvaluated) {
                row.classList.add('evaluated-row');
                // Optional: Add a small badge/icon to indicate it's evaluated
                const actionCell = row.querySelector('td:last-child');
                if (actionCell) {
                    const evaluation = this.evaluations[key];
                    let badgeClass = 'bg-secondary';
                    if (evaluation.decision === 'true_duplicate') badgeClass = 'bg-success';
                    else if (evaluation.decision === 'hierarchical') badgeClass = 'bg-info';
                    else if (evaluation.decision === 'unrelated') badgeClass = 'bg-danger';
                    else if (evaluation.decision === 'unsure') badgeClass = 'bg-warning';
                    
                    // Check if the badge already exists
                    let badge = actionCell.querySelector('.evaluated-badge');
                    if (!badge) {
                        badge = document.createElement('span');
                        badge.className = `evaluated-badge badge ${badgeClass} ms-2`;
                        badge.textContent = '✓';
                        badge.title = this.getDecisionName(evaluation.decision);
                        actionCell.prepend(badge);
                    } else {
                        // Update existing badge
                        badge.className = `evaluated-badge badge ${badgeClass} ms-2`;
                        badge.title = this.getDecisionName(evaluation.decision);
                    }
                    
                    // Add "Change" button if it doesn't exist
                    let changeButton = actionCell.querySelector('.change-evaluation-btn');
                    if (!changeButton) {
                        const viewButton = actionCell.querySelector('.view-details-btn');
                        if (viewButton) {
                            changeButton = document.createElement('button');
                            changeButton.className = 'btn btn-sm btn-warning change-evaluation-btn ms-2';
                            changeButton.innerHTML = '<i class="bi bi-pencil"></i> Change';
                            changeButton.title = 'Change Evaluation';
                            changeButton.setAttribute('data-index', index);
                            
                            // Insert after view details button
                            viewButton.parentNode.insertBefore(changeButton, viewButton.nextSibling);
                            
                            // Add event listener
                            changeButton.addEventListener('click', (event) => {
                                const idx = parseInt(event.currentTarget.getAttribute('data-index'));
                                if (idx !== null && idx >= 0 && idx < this.filteredDuplicates.length) {
                                    this.showDuplicateDetails(this.filteredDuplicates[idx]);
                                    document.querySelector('a[href="#duplicate-details-tab"]').click();
                                }
                            });
                        }
                    }
                }
            }
        });
        
        // Log that indicators were updated
        console.log(`Updated evaluation indicators for level ${this.currentLevel} with ${currentLevelEvalKeys.length} evaluations`);
    }
    
    populateEvaluationForm() {
        if (!this.selectedDuplicate) return;
        
        // Generate key for this duplicate pair
        const key = `${this.selectedDuplicate.term1}___${this.selectedDuplicate.term2}`;
        
        // Check if we have an evaluation
        const evaluation = this.evaluations[key];
        
        // Update term labels in the form
        document.querySelectorAll('.term1-label').forEach(el => {
            el.textContent = this.selectedDuplicate.term1;
        });
        
        document.querySelectorAll('.term2-label').forEach(el => {
            el.textContent = this.selectedDuplicate.term2;
        });
        
        // Clear the form
        document.querySelectorAll('input[name="decision"]').forEach(radio => {
            radio.checked = false;
        });
        
        document.querySelectorAll('input[name="canonical"]').forEach(radio => {
            radio.checked = false;
        });
        
        document.getElementById('relationship-type').value = '';
        document.getElementById('evaluation-notes').value = '';
        
        // If we have an evaluation, populate the form
        const saveBtn = document.querySelector('.save-evaluation-btn');
        if (evaluation) {
            // Set decision
            const decisionRadio = document.querySelector(`input[name="decision"][value="${evaluation.decision}"]`);
            if (decisionRadio) decisionRadio.checked = true;
            
            // Set relationship
            document.getElementById('relationship-type').value = evaluation.relationship || '';
            
            // Set canonical term
            const canonicalRadio = document.querySelector(`input[name="canonical"][value="${evaluation.canonical}"]`);
            if (canonicalRadio) canonicalRadio.checked = true;
            
            // Set notes
            document.getElementById('evaluation-notes').value = evaluation.notes || '';
            
            // Update save button to indicate edit mode
            if (saveBtn) {
                saveBtn.textContent = 'Update Evaluation';
                saveBtn.classList.add('btn-warning');
                saveBtn.classList.remove('btn-primary');
                
                // Add a note indicating existing evaluation
                const evaluationSection = document.querySelector('.evaluation-section');
                if (evaluationSection) {
                    let existingNote = evaluationSection.querySelector('.evaluation-exists-note');
                    if (!existingNote) {
                        existingNote = document.createElement('div');
                        existingNote.className = 'evaluation-exists-note alert alert-warning mb-3';
                        existingNote.innerHTML = '<i class="bi bi-exclamation-triangle"></i> You are editing an existing evaluation. Save to update your decision.';
                        evaluationSection.insertBefore(existingNote, evaluationSection.firstChild.nextSibling);
                    }
                }
            }
        } else {
            // Reset save button to normal state
            if (saveBtn) {
                saveBtn.textContent = 'Save Evaluation';
                saveBtn.classList.remove('btn-warning');
                saveBtn.classList.add('btn-primary');
                
                // Remove edit note if it exists
                const existingNote = document.querySelector('.evaluation-exists-note');
                if (existingNote) {
                    existingNote.remove();
                }
            }
        }
        
        // Scroll to evaluation section
        const evaluationSection = document.querySelector('.evaluation-section');
        if (evaluationSection) {
            setTimeout(() => {
                evaluationSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }, 300);
        }
    }
    
    exportEvaluations() {
        if (Object.keys(this.evaluations).length === 0) {
            alert('No evaluations to export');
            return;
        }
        
        // Get the current date and time for filenames
        const currentDate = new Date().toISOString().slice(0,10);
        const fileName = `duplicate_evaluations_lv${this.currentLevel}_${currentDate}`;
        
        // Convert evaluations to CSV
        const headers = [
            'Term 1',
            'Term 2',
            'Parent',
            'Similarity',
            'Cond Prob 1|2',
            'Cond Prob 2|1',
            'Mutual Info',
            'Co-occurrences',
            'Decision',
            'Relationship Type',
            'Canonical Term',
            'Notes',
            'Timestamp'
        ];
        
        let csv = headers.join(',') + '\n';
        
        // Filter evaluations for current level
        const currentLevelEvaluations = Object.values(this.evaluations).filter(e => {
            return this.allDuplicates.some(d => 
                d.term1 === e.term1 && d.term2 === e.term2
            );
        });
        
        currentLevelEvaluations.forEach(e => {
            const row = [
                `"${e.term1}"`,
                `"${e.term2}"`,
                `"${e.parent}"`,
                e.similarity,
                e.cond_prob_1_given_2,
                e.cond_prob_2_given_1,
                e.mutual_information,
                e.co_occurrences,
                `"${this.getDecisionName(e.decision)}"`,
                `"${e.relationship ? this.getRelationshipTypeName(e.relationship) : ''}"`,
                `"${e.canonical === 'term1' ? e.term1 : e.canonical === 'term2' ? e.term2 : e.canonical === 'neither' ? 'Keep Both' : ''}"`,
                `"${e.notes ? e.notes.replace(/"/g, '""') : ''}"`,
                e.timestamp
            ];
            
            csv += row.join(',') + '\n';
        });
        
        // Create and download CSV file
        const csvBlob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
        const csvUrl = URL.createObjectURL(csvBlob);
        const csvLink = document.createElement('a');
        csvLink.setAttribute('href', csvUrl);
        csvLink.setAttribute('download', `${fileName}.csv`);
        csvLink.style.visibility = 'hidden';
        document.body.appendChild(csvLink);
        csvLink.click();
        document.body.removeChild(csvLink);
        
        // Create JSON export
        const jsonData = JSON.stringify(this.evaluations, null, 2);
        const jsonBlob = new Blob([jsonData], { type: 'application/json;charset=utf-8;' });
        const jsonUrl = URL.createObjectURL(jsonBlob);
        const jsonLink = document.createElement('a');
        jsonLink.setAttribute('href', jsonUrl);
        jsonLink.setAttribute('download', `${fileName}.json`);
        
        // Ask if they want JSON export as well
        if (confirm('CSV file has been downloaded. Would you like to also download a JSON file with complete evaluation data (recommended for reimporting)?')) {
            jsonLink.click();
        }
        
        URL.revokeObjectURL(jsonUrl);
        document.body.appendChild(jsonLink);
        jsonLink.click();
        document.body.removeChild(jsonLink);
        
        // Also save copies to the server
        this.saveExportToServer(fileName, csvBlob, jsonBlob);
    }
    
    saveExportToServer(fileName, csvBlob, jsonBlob) {
        // Save CSV to server
        const csvFormData = new FormData();
        csvFormData.append('file', csvBlob, `${fileName}.csv`);
        
        fetch('/api/save_evaluation_export', {
            method: 'POST',
            body: csvFormData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                console.log(`CSV export saved to server: ${data.filePath}`);
            } else {
                console.error('Failed to save CSV export to server:', data.message);
            }
        })
        .catch(error => {
            console.error('Error saving CSV export to server:', error);
        });
        
        // Save JSON to server
        const jsonFormData = new FormData();
        jsonFormData.append('file', jsonBlob, `${fileName}.json`);
        
        fetch('/api/save_evaluation_export', {
            method: 'POST',
            body: jsonFormData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                console.log(`JSON export saved to server: ${data.filePath}`);
            } else {
                console.error('Failed to save JSON export to server:', data.message);
            }
        })
        .catch(error => {
            console.error('Error saving JSON export to server:', error);
        });
    }
    
    importEvaluations(file) {
        if (!file) return;
        
        const reader = new FileReader();
        const fileExtension = file.name.split('.').pop().toLowerCase();
        
        reader.onload = (event) => {
            try {
                if (fileExtension === 'json') {
                    // Handle JSON import
                    this.importFromJSON(event.target.result);
                } else if (fileExtension === 'csv') {
                    // Handle CSV import
                    this.importFromCSV(event.target.result);
                } else {
                    alert('Unsupported file format. Please upload a .csv or .json file.');
                }
            } catch (error) {
                console.error('Error importing evaluations:', error);
                alert(`Error importing evaluations: ${error.message}`);
            }
        };
        
        if (fileExtension === 'json') {
            reader.readAsText(file);
        } else if (fileExtension === 'csv') {
            reader.readAsText(file);
        } else {
            alert('Unsupported file format. Please upload a .csv or .json file.');
        }
    }
    
    importFromJSON(jsonString, showConfirmation = true) {
        try {
            const importedEvaluations = JSON.parse(jsonString);
            const importCount = Object.keys(importedEvaluations).length;
            
            if (importCount === 0) {
                if (showConfirmation) {
                    alert('No evaluations found in the imported file.');
                }
                return;
            }
            
            // Confirm before overwriting
            if (Object.keys(this.evaluations).length > 0) {
                let shouldMerge = true;
                if (showConfirmation) {
                    shouldMerge = confirm(`You already have evaluations saved. Do you want to merge the imported ${importCount} evaluations with your existing ones? Click Cancel to replace all existing evaluations.`);
                }
                
                if (!shouldMerge) {
                    // Replace all evaluations
                    this.evaluations = importedEvaluations;
                    this.saveEvaluations();
                    if (showConfirmation) {
                        alert(`Replaced all evaluations with ${importCount} imported evaluations.`);
                    }
                    this.updateEvaluationUI();
                    this.refreshDuplicatesUI();
                    return;
                }
                
                // Merge evaluations (imported overwrite existing for same keys)
                const mergedCount = this.mergeEvaluations(importedEvaluations);
                if (showConfirmation) {
                    alert(`Successfully merged ${mergedCount} imported evaluations.`);
                }
            } else {
                // No existing evaluations, just use imported ones
                this.evaluations = importedEvaluations;
                this.saveEvaluations();
                if (showConfirmation) {
                    alert(`Successfully imported ${importCount} evaluations.`);
                }
            }
            
            // Update UI to reflect changes
            this.updateEvaluationUI();
            this.refreshDuplicatesUI();
        } catch (error) {
            console.error('Error parsing JSON:', error);
            if (showConfirmation) {
                alert('Failed to parse the JSON file. Please make sure it is a valid JSON file exported from this tool.');
            }
        }
    }
    
    importFromCSV(csvString, showConfirmation = true) {
        try {
            // Parse CSV
            const lines = csvString.split('\n');
            const headers = lines[0].split(',');
            
            // Expected headers (we need to map these to our internal property names)
            const headerMap = {
                'Term 1': 'term1',
                'Term 2': 'term2',
                'Parent': 'parent',
                'Similarity': 'similarity',
                'Cond Prob 1|2': 'cond_prob_1_given_2',
                'Cond Prob 2|1': 'cond_prob_2_given_1',
                'Mutual Info': 'mutual_information',
                'Co-occurrences': 'co_occurrences',
                'Decision': 'decision',
                'Relationship Type': 'relationship',
                'Canonical Term': 'canonical',
                'Notes': 'notes',
                'Timestamp': 'timestamp'
            };
            
            // Validate headers
            const requiredHeaders = ['Term 1', 'Term 2', 'Decision'];
            for (const header of requiredHeaders) {
                if (!headers.includes(header)) {
                    if (showConfirmation) {
                        alert(`Missing required column: ${header}. Please use a CSV file exported from this tool.`);
                    }
                    return;
                }
            }
            
            // Parse the CSV data
            const importedEvaluations = {};
            let importCount = 0;
            
            for (let i = 1; i < lines.length; i++) {
                if (!lines[i].trim()) continue; // Skip empty lines
                
                // Handle quoted values with commas inside them
                const values = this.parseCSVLine(lines[i]);
                if (values.length !== headers.length) {
                    console.warn(`Skipping line ${i+1}: incorrect number of columns`);
                    continue;
                }
                
                const evaluation = {};
                let term1, term2;
                
                // Map CSV columns to evaluation properties
                for (let j = 0; j < headers.length; j++) {
                    const header = headers[j];
                    const value = values[j];
                    
                    if (header === 'Term 1') {
                        term1 = this.cleanCSVValue(value);
                        evaluation.term1 = term1;
                    } else if (header === 'Term 2') {
                        term2 = this.cleanCSVValue(value);
                        evaluation.term2 = term2;
                    } else if (header === 'Decision') {
                        // Map human-readable decision name back to internal value
                        const decisionName = this.cleanCSVValue(value);
                        evaluation.decision = this.getDecisionValue(decisionName);
                    } else if (header === 'Relationship Type') {
                        // Map human-readable relationship type back to internal value
                        const relationshipName = this.cleanCSVValue(value);
                        evaluation.relationship = this.getRelationshipValue(relationshipName);
                    } else if (header === 'Canonical Term') {
                        // Handle canonical term logic
                        const canonicalTerm = this.cleanCSVValue(value);
                        if (canonicalTerm === 'Keep Both') {
                            evaluation.canonical = 'neither';
                        } else if (canonicalTerm === term1) {
                            evaluation.canonical = 'term1';
                        } else if (canonicalTerm === term2) {
                            evaluation.canonical = 'term2';
                        } else {
                            evaluation.canonical = '';
                        }
                    } else if (headerMap[header]) {
                        // Map numeric values correctly
                        const propName = headerMap[header];
                        if (['similarity', 'cond_prob_1_given_2', 'cond_prob_2_given_1', 'mutual_information'].includes(propName)) {
                            evaluation[propName] = parseFloat(value) || 0;
                        } else if (propName === 'co_occurrences') {
                            evaluation[propName] = parseInt(value) || 0;
                        } else {
                            evaluation[propName] = this.cleanCSVValue(value);
                        }
                    }
                }
                
                // Set timestamp if missing
                if (!evaluation.timestamp) {
                    evaluation.timestamp = new Date().toISOString();
                }
                
                // Only add if we have the minimum required fields
                if (evaluation.term1 && evaluation.term2 && evaluation.decision) {
                    const key = `${evaluation.term1}___${evaluation.term2}`;
                    importedEvaluations[key] = evaluation;
                    importCount++;
                }
            }
            
            if (importCount === 0) {
                if (showConfirmation) {
                    alert('No valid evaluations found in the CSV file.');
                }
                return;
            }
            
            // Handle merging or replacing existing evaluations
            if (Object.keys(this.evaluations).length > 0) {
                let shouldMerge = true;
                if (showConfirmation) {
                    shouldMerge = confirm(`You already have evaluations saved. Do you want to merge the imported ${importCount} evaluations with your existing ones? Click Cancel to replace all existing evaluations.`);
                }
                
                if (!shouldMerge) {
                    // Replace all evaluations
                    this.evaluations = importedEvaluations;
                    this.saveEvaluations();
                    if (showConfirmation) {
                        alert(`Replaced all evaluations with ${importCount} imported evaluations.`);
                    }
                    this.updateEvaluationUI();
                    this.refreshDuplicatesUI();
                    return;
                }
                
                // Merge evaluations
                const mergedCount = this.mergeEvaluations(importedEvaluations);
                if (showConfirmation) {
                    alert(`Successfully merged ${mergedCount} imported evaluations.`);
                }
            } else {
                // No existing evaluations, just use imported ones
                this.evaluations = importedEvaluations;
                this.saveEvaluations();
                if (showConfirmation) {
                    alert(`Successfully imported ${importCount} evaluations.`);
                }
            }
            
            // Update UI to reflect changes
            this.updateEvaluationUI();
            this.refreshDuplicatesUI();
        } catch (error) {
            console.error('Error parsing CSV:', error);
            if (showConfirmation) {
                alert(`Failed to parse the CSV file: ${error.message}`);
            }
        }
    }
    
    parseCSVLine(line) {
        const result = [];
        let currentValue = '';
        let withinQuotes = false;
        
        for (let i = 0; i < line.length; i++) {
            const char = line[i];
            
            if (char === '"') {
                // Check if this is an escaped quote (""")
                if (i + 1 < line.length && line[i + 1] === '"') {
                    currentValue += '"';
                    i++; // Skip the next quote
                } else {
                    // Toggle quote state
                    withinQuotes = !withinQuotes;
                }
            } else if (char === ',' && !withinQuotes) {
                // End of field
                result.push(currentValue);
                currentValue = '';
            } else {
                currentValue += char;
            }
        }
        
        // Add the last field
        result.push(currentValue);
        return result;
    }
    
    cleanCSVValue(value) {
        // Remove surrounding quotes and handle escaped quotes
        if (value.startsWith('"') && value.endsWith('"')) {
            return value.substring(1, value.length - 1).replace(/""/g, '"');
        }
        return value;
    }
    
    getDecisionValue(decisionName) {
        // Map human-readable decision name back to internal value
        const decisionMap = {
            'True Duplicate': 'true_duplicate',
            'Related Terms': 'related_terms',
            'Hierarchical Relationship': 'hierarchical',
            'Unrelated Terms': 'unrelated',
            'Need More Information': 'unsure'
        };
        
        return decisionMap[decisionName] || 'unsure';
    }
    
    getRelationshipValue(relationshipName) {
        // Map human-readable relationship type back to internal value
        const relationshipMap = {
            'Exact Synonym': 'exact_synonym',
            'Spelling Variation': 'spelling_variation',
            'Abbreviation/Full Form': 'abbreviation',
            'Terminological Variation': 'term_variation',
            'Broader Term': 'broader_term',
            'Narrower Term': 'narrower_term',
            'Related Concept': 'related_concept',
            'Other': 'other'
        };
        
        return relationshipMap[relationshipName] || '';
    }
    
    mergeEvaluations(importedEvaluations) {
        let mergedCount = 0;
        
        // Merge imported evaluations with existing ones
        for (const [key, evaluation] of Object.entries(importedEvaluations)) {
            if (key in this.evaluations) {
                // This is an update to an existing evaluation
                const existingEval = { ...this.evaluations[key] };
                
                // Keep history of previous evaluations if not already in the imported data
                if (!evaluation.updates) {
                    evaluation.updated = true;
                    evaluation.original_timestamp = existingEval.timestamp;
                    
                    if (!existingEval.updates) {
                        evaluation.updates = [existingEval];
                    } else {
                        evaluation.updates = [...existingEval.updates, existingEval];
                    }
                }
            }
            
            // Add/update the evaluation
            this.evaluations[key] = evaluation;
            mergedCount++;
        }
        
        // Save to localStorage
        this.saveEvaluations();
        
        return mergedCount;
    }
    
    clearEvaluations() {
        // Create a new object with only evaluations from other levels
        const newEvaluations = {};
        
        Object.entries(this.evaluations).forEach(([key, evaluation]) => {
            // Check if this evaluation is for the current level
            const isCurrentLevel = this.allDuplicates.some(d => 
                d.term1 === evaluation.term1 && d.term2 === evaluation.term2
            );
            
            // Keep evaluations from other levels
            if (!isCurrentLevel) {
                newEvaluations[key] = evaluation;
            }
        });
        
        this.evaluations = newEvaluations;
        this.saveEvaluations();
        this.updateEvaluationUI();
        this.updateEvaluationIndicators();
    }
}

// Initialize the duplicate analysis module when the page loads
document.addEventListener('DOMContentLoaded', function() {
    const duplicateAnalysis = new DuplicateAnalysis();
    duplicateAnalysis.init();
}); 