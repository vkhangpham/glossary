document.addEventListener('DOMContentLoaded', function() {
    const statsContainer = document.getElementById('stats-container');
    const evaluatedTermsList = document.getElementById('evaluated-terms-list');
    const evaluationNotesBody = document.getElementById('evaluation-notes-body');
    const problematicTermsList = document.getElementById('problematic-terms-list');
    const applyFiltersBtn = document.getElementById('apply-filters-btn');
    const filterCheckboxes = document.querySelectorAll('.filter-checkbox');

    let fullEvaluationsData = {}; // Store the full data for filtering

    const importanceLabels = {
        'field': 'Field of Research',
        'subject': 'Subject of Research',
        'general': 'General Term',
        'not_academic': 'Not Academic / Noise'
    };

    const levelLabels = {
        'Correct': 'Correct',
        'Higher': 'Should be higher level',
        'Lower': 'Should be lower level'
    };

    // Color scheme for charts
    const chartColors = [
        '#4285F4', // Google Blue
        '#DB4437', // Google Red
        '#F4B400', // Google Yellow
        '#0F9D58', // Google Green
        '#AB47BC', // Purple
        '#00ACC1', // Cyan
        '#FF7043', // Orange
        '#78909C'  // Blue Grey
    ];

    async function fetchAndDisplayStats() {
        try {
            const response = await fetch('/api/evaluation_stats');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const stats = await response.json();
            displayStats(stats);

            // Fetch and display the list of evaluated terms separately
            fetchAndDisplayEvaluationDetails();

        } catch (error) {
            console.error('Error fetching evaluation stats:', error);
            statsContainer.innerHTML = '<div class="alert alert-danger">Failed to load statistics. Please try again later.</div>';
        }
    }

    async function fetchAndDisplayEvaluationDetails() {
        try {
            const response = await fetch('/api/all_evaluations');
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const evaluations = await response.json(); // This is now the full dictionary
            const terms = Object.keys(evaluations).sort((a, b) => a.toLowerCase().localeCompare(b.toLowerCase()));

            fullEvaluationsData = evaluations; // Store for filtering

            evaluatedTermsList.innerHTML = ''; // Clear loading message
            evaluationNotesBody.innerHTML = ''; // Clear loading message for notes

            if (terms.length === 0) {
                evaluatedTermsList.innerHTML = '<li class="list-group-item text-muted">No terms have been evaluated yet.</li>';
                evaluationNotesBody.innerHTML = '<p class="text-muted">No evaluations with notes found.</p>';
                problematicTermsList.innerHTML = '<li class="list-group-item text-muted">No evaluations to filter.</li>';
                return;
            }

            let notesFound = false;

            terms.forEach(term => {
                const evaluation = evaluations[term];

                // --- Populate Evaluated Terms List (as before) ---
                const li = document.createElement('li');
                li.className = 'list-group-item';
                // Create a link - Note: This just goes to the page; pre-selecting requires more logic later if needed.
                const link = document.createElement('a');
                link.href = `/manual_evaluation?term=${encodeURIComponent(term)}`; // Add term as query param
                link.textContent = term;
                link.target = "_blank"; // Open in new tab for convenience
                li.appendChild(link);
                evaluatedTermsList.appendChild(li);

                // --- Populate Notes Section ---
                let termNotesHTML = '';
                const notesMap = {
                    'academic_importance_notes': 'Importance',
                    'level_correctness_notes': 'Level',
                    'variation_correctness_notes': 'Variations',
                    'parent_relationship_notes': 'Parents',
                    'related_concepts_notes': 'Related Concepts'
                };

                for (const noteKey in notesMap) {
                    if (evaluation[noteKey] && evaluation[noteKey].trim() !== '') {
                        const noteLabel = notesMap[noteKey];
                        // Simple escaping for display
                        const escapedNote = evaluation[noteKey]
                             .replace(/&/g, "&amp;")
                             .replace(/</g, "&lt;")
                             .replace(/>/g, "&gt;")
                             .replace(/"/g, "&quot;")
                             .replace(/'/g, "&#039;");
                        termNotesHTML += `<dt>${noteLabel} Note:</dt><dd><pre>${escapedNote}</pre></dd>`;
                        notesFound = true;
                    }
                }

                if (termNotesHTML) {
                     const termDiv = document.createElement('div');
                     termDiv.className = 'mb-3';
                     termDiv.innerHTML = `
                        <h5>${term}</h5>
                        <dl class="dl-horizontal">
                            ${termNotesHTML}
                        </dl>
                        <hr>
                    `;
                    evaluationNotesBody.appendChild(termDiv);
                }
            });

            if (!notesFound) {
                evaluationNotesBody.innerHTML = '<p class="text-muted">No notes found in the recorded evaluations.</p>';
            }

        } catch (error) {
            console.error('Error fetching evaluated terms list:', error);
            evaluatedTermsList.innerHTML = '<li class="list-group-item text-danger">Failed to load evaluated terms list.</li>';
            evaluationNotesBody.innerHTML = '<p class="text-danger">Failed to load evaluation notes.</p>';
            problematicTermsList.innerHTML = '<li class="list-group-item text-danger">Failed to load data for filtering.</li>';
        }
    }

    function displayStats(stats) {
        if (stats.total_evaluated === 0) {
            statsContainer.innerHTML = '<div class="alert alert-info">No evaluations have been recorded yet.</div>';
            return;
        }

        // Generate the HTML for the single main card
        statsContainer.innerHTML = `
            <div class="card stat-card">
                 <div class="card-header">Evaluation Summary</div>
                 <div class="card-body">
                    <h5 class="card-title mb-4">${stats.total_evaluated} Terms Evaluated</h5>

                    <div class="row chart-area">
                        <div class="col-md-6">
                            <h6 class="text-center mb-3">Academic Importance</h6>
                            <div id="importance-chart"></div> <!-- Plotly target -->
                        </div>
                         <div class="col-md-6">
                            <h6 class="text-center mb-3">Level Correctness</h6>
                            <div id="level-chart"></div> <!-- Plotly target -->
                        </div>
                    </div>

                    <hr>

                     <div class="row progress-area">
                        <div class="col-md-6">
                             <h6>Variation Correctness</h6>
                            <p class="card-text small text-muted">Based on ${stats.correct_variations_count} correct variations out of ${stats.total_variations_rated} rated variations.</p>
                            <div class="progress mb-3">
                                <div class="progress-bar bg-success" role="progressbar" style="width: ${stats.variation_correctness_avg}%;" aria-valuenow="${stats.variation_correctness_avg}" aria-valuemin="0" aria-valuemax="100">${stats.variation_correctness_avg}% Avg Correct</div>
                            </div>
                        </div>
                         <div class="col-md-6">
                             <h6>Parent Relationship Correctness</h6>
                            <p class="card-text small text-muted">Based on ${stats.correct_parents_count} correct relationships out of ${stats.total_parents_rated} rated relationships.</p>
                             <div class="progress mb-3">
                                <div class="progress-bar bg-primary" role="progressbar" style="width: ${stats.parent_correctness_avg}%;" aria-valuenow="${stats.parent_correctness_avg}" aria-valuemin="0" aria-valuemax="100">${stats.parent_correctness_avg}% Avg Correct</div>
                            </div>
                        </div>
                    </div>
                    
                    <hr>
                    
                    <div class="row progress-area">
                        <div class="col-md-6">
                            <h6>Potential Missing Parents (from Related Concepts)</h6>
                            <p class="card-text small text-muted">Based on ${stats.potential_parents?.selected || 0} selected out of ${stats.potential_parents?.total || 0} potential parent relationships.</p>
                            <div class="progress mb-3">
                                <div class="progress-bar bg-info" role="progressbar" style="width: ${stats.potential_parents?.percent || 0}%;" aria-valuenow="${stats.potential_parents?.percent || 0}" aria-valuemin="0" aria-valuemax="100">${stats.potential_parents?.percent || 0}% Selected</div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <h6>Potential Missing Variations (from Related Concepts)</h6>
                            <p class="card-text small text-muted">Based on ${stats.potential_variations?.selected || 0} selected out of ${stats.potential_variations?.total || 0} potential variations.</p>
                            <div class="progress mb-3">
                                <div class="progress-bar bg-warning" role="progressbar" style="width: ${stats.potential_variations?.percent || 0}%;" aria-valuenow="${stats.potential_variations?.percent || 0}" aria-valuemin="0" aria-valuemax="100">${stats.potential_variations?.percent || 0}% Selected</div>
                            </div>
                        </div>
                    </div>
                    
                    <hr>
                    
                    <div class="row">
                        <div class="col-md-6">
                            <div class="card mb-3">
                                <div class="card-header bg-light">
                                    <a class="text-decoration-none" data-bs-toggle="collapse" href="#orphan-terms-collapse" role="button" aria-expanded="false" aria-controls="orphan-terms-collapse">
                                        Orphan Terms <span class="badge bg-danger">${stats.orphan_terms?.length || 0}</span> <i class="bi bi-chevron-down small"></i>
                                    </a>
                                </div>
                                <div class="collapse" id="orphan-terms-collapse">
                                    <div class="card-body">
                                        <p class="card-text small text-muted">Terms without any parent terms.</p>
                                        <div class="orphan-terms-list">
                                            ${renderOrphanTermsList(stats.orphan_terms || [])}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="col-md-6">
                            <div class="card mb-3">
                                <div class="card-header bg-light">
                                    <a class="text-decoration-none" data-bs-toggle="collapse" href="#leaf-terms-collapse" role="button" aria-expanded="false" aria-controls="leaf-terms-collapse">
                                        Leaf Terms Not at Level 3 <span class="badge bg-warning">${stats.leaf_terms_not_level_3?.length || 0}</span> <i class="bi bi-chevron-down small"></i>
                                    </a>
                                </div>
                                <div class="collapse" id="leaf-terms-collapse">
                                    <div class="card-body">
                                        <p class="card-text small text-muted">Terms without children that are not at the lowest level (level 3).</p>
                                        <div class="leaf-terms-list">
                                            ${renderLeafTermsList(stats.leaf_terms_not_level_3 || [])}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                 </div>
            </div>
        `;

        // Populate charts (lists are no longer populated here)
        populateListAndChart('importance', stats.importance_counts, importanceLabels);
        populateListAndChart('level', stats.level_correctness_counts, levelLabels);

        // Add event listener for filter button after Plotly setup (or ensure it's added only once)
        if (applyFiltersBtn && !applyFiltersBtn.dataset.listenerAttached) {
            applyFiltersBtn.addEventListener('click', () => filterAndDisplayProblematicTerms());
            applyFiltersBtn.dataset.listenerAttached = 'true'; // Prevent multiple listeners
        }
    }

    function populateListAndChart(type, counts, labels) {
        // const listElement = document.getElementById(`${type}-list`); // List no longer exists
        const chartDivId = `${type}-chart`; // Target div ID for Plotly
        // listElement.innerHTML = ''; // Clear placeholder

        const plotlyLabels = [];
        const plotlyValues = [];
        // Plotly can use the same color array, but might cycle differently
        // const plotlyColors = [];

        // let colorIndex = 0;
        const sortedKeys = Object.keys(counts).sort((a, b) => counts[b] - counts[a]); // Sort by count desc

        sortedKeys.forEach(key => {
            const count = counts[key];
            // const percentage = ((count / total) * 100).toFixed(1); // Total no longer needed just for chart
            const label = labels[key] || key; // Use defined label or key itself

            plotlyLabels.push(label);
            plotlyValues.push(count);
            // plotlyColors.push(chartColors[colorIndex % chartColors.length]);
            // colorIndex++;

            /* List item creation removed
            const listItem = document.createElement('li');
            listItem.className = 'list-group-item';
            listItem.innerHTML = `
                ${label}
                <span class="badge bg-secondary rounded-pill">${count} (${percentage}%)</span>
            `;
            listElement.appendChild(listItem);
            */
        });

        // Create Plotly Doughnut chart
        const plotData = [{
            values: plotlyValues,
            labels: plotlyLabels,
            type: 'pie',
            hole: .4, // Makes it a doughnut chart
            marker: {
                colors: chartColors // Use the predefined color scheme
            },
            hoverinfo: 'label+percent',
            textinfo: 'value+percent', // Show percentage on slices
            insidetextorientation: 'radial',
            textfont: { // Explicitly set font for visibility
                size: 11, 
                color: '#ffffff' // White color for contrast
            }
        }];

        const layout = {
            // title: `Distribution by ${type}`, // Title removed, using h6 instead
            height: 260, // Reduced height for side-by-side layout
            showlegend: true,
            legend: {
                x: 0.5,
                xanchor: 'center',
                y: -0.1, // Position legend below the chart
                orientation: 'h' // Horizontal legend
            },
            margin: { l: 20, r: 20, t: 20, b: 40 }, // Adjust margins (reduced top)
            paper_bgcolor: 'rgba(0,0,0,0)', // Transparent background
            plot_bgcolor: 'rgba(0,0,0,0)'
        };

        Plotly.newPlot(chartDivId, plotData, layout, {responsive: true});
    }

    function filterAndDisplayProblematicTerms() {
        // Get selected filter criteria
        const selectedFilters = Array.from(filterCheckboxes)
            .filter(checkbox => checkbox.checked)
            .map(checkbox => checkbox.value);

        if (selectedFilters.length === 0) {
            problematicTermsList.innerHTML = '<li class="list-group-item text-muted">No filter criteria selected.</li>';
            return;
        }

        if (!fullEvaluationsData || Object.keys(fullEvaluationsData).length === 0) {
            problematicTermsList.innerHTML = '<li class="list-group-item text-muted">No evaluation data available to filter.</li>';
            return;
        }

        // Filter terms that meet any of the selected criteria
        const problematicTerms = [];
        
        for (const [term, evaluation] of Object.entries(fullEvaluationsData)) {
            const isProblematic = {
                importance: false,
                level: false,
                variations: false, 
                parents: false,
                potential_parents: false,
                potential_variations: false
            };
            
            // Check importance criteria
            if (evaluation.academic_importance && ['general', 'not_academic'].includes(evaluation.academic_importance)) {
                isProblematic.importance = true;
            }
            
            // Check level criteria (if level_correctness is not 'correct')
            if (evaluation.level_correctness && evaluation.level_correctness !== 'correct') {
                isProblematic.level = true;
            }
            
            // Check variations criteria (if any variation is marked as incorrect)
            if (evaluation.variation_correctness && typeof evaluation.variation_correctness === 'object') {
                const hasIncorrectVariation = Object.values(evaluation.variation_correctness).some(isCorrect => !isCorrect);
                if (hasIncorrectVariation) {
                    isProblematic.variations = true;
                }
            }
            
            // Check parent relationships criteria (if any parent is marked as incorrect)
            if (evaluation.parent_relationships && typeof evaluation.parent_relationships === 'object') {
                const hasIncorrectParent = Object.values(evaluation.parent_relationships).some(isCorrect => !isCorrect);
                if (hasIncorrectParent) {
                    isProblematic.parents = true;
                }
            }
            
            // Check potential parents criteria (if any potential parent is selected)
            if (evaluation.potential_parents && typeof evaluation.potential_parents === 'object') {
                const hasSelectedPotentialParent = Object.values(evaluation.potential_parents).some(isSelected => isSelected);
                if (hasSelectedPotentialParent) {
                    isProblematic.potential_parents = true;
                }
            }
            
            // Check potential variations criteria (if any potential variation is selected)
            if (evaluation.potential_variations && typeof evaluation.potential_variations === 'object') {
                const hasSelectedPotentialVariation = Object.values(evaluation.potential_variations).some(isSelected => isSelected);
                if (hasSelectedPotentialVariation) {
                    isProblematic.potential_variations = true;
                }
            }
            
            // Check if any of the selected filters match this term's issues
            const hasSelectedIssue = selectedFilters.some(filter => isProblematic[filter]);
            
            if (hasSelectedIssue) {
                // Collect the problematic issues for this term
                const issues = [];
                if (isProblematic.importance && selectedFilters.includes('importance')) {
                    issues.push(`Academic importance: ${importanceLabels[evaluation.academic_importance] || evaluation.academic_importance || 'Not set'}`);
                }
                if (isProblematic.level && selectedFilters.includes('level')) {
                    issues.push(`Level suggestion: Should be ${evaluation.level_correctness}`);
                }
                if (isProblematic.variations && selectedFilters.includes('variations')) {
                    const incorrectCount = Object.values(evaluation.variation_correctness).filter(isCorrect => !isCorrect).length;
                    issues.push(`${incorrectCount} incorrect variation(s)`);
                }
                if (isProblematic.parents && selectedFilters.includes('parents')) {
                    const incorrectCount = Object.values(evaluation.parent_relationships).filter(isCorrect => !isCorrect).length;
                    issues.push(`${incorrectCount} incorrect parent(s)`);
                }
                if (isProblematic.potential_parents && selectedFilters.includes('potential_parents')) {
                    const selectedCount = Object.values(evaluation.potential_parents).filter(isSelected => isSelected).length;
                    issues.push(`${selectedCount} potential missing parent(s)`);
                }
                if (isProblematic.potential_variations && selectedFilters.includes('potential_variations')) {
                    const selectedCount = Object.values(evaluation.potential_variations).filter(isSelected => isSelected).length;
                    issues.push(`${selectedCount} potential missing variation(s)`);
                }
                
                problematicTerms.push({
                    term,
                    issues,
                    timestamp: evaluation.timestamp || 'Unknown'
                });
            }
        }
        
        // Sort problematic terms by timestamp (most recent first)
        problematicTerms.sort((a, b) => {
            if (a.timestamp === 'Unknown') return 1;
            if (b.timestamp === 'Unknown') return -1;
            return new Date(b.timestamp) - new Date(a.timestamp);
        });
        
        // Display the filtered terms
        problematicTermsList.innerHTML = '';
        
        if (problematicTerms.length === 0) {
            problematicTermsList.innerHTML = '<li class="list-group-item text-muted">No terms match the selected filter criteria.</li>';
            return;
        }
        
        problematicTerms.forEach(item => {
            const li = document.createElement('li');
            li.className = 'list-group-item';
            
            const link = document.createElement('a');
            link.href = `/manual_evaluation?term=${encodeURIComponent(item.term)}`;
            link.textContent = item.term;
            link.target = "_blank";
            li.appendChild(link);
            
            const issuesBadges = document.createElement('div');
            issuesBadges.className = 'mt-1';
            item.issues.forEach(issue => {
                const badge = document.createElement('span');
                badge.className = 'badge bg-warning text-dark me-1 mb-1';
                badge.textContent = issue;
                issuesBadges.appendChild(badge);
            });
            li.appendChild(issuesBadges);
            
            const timestamp = document.createElement('small');
            timestamp.className = 'text-muted d-block mt-1';
            timestamp.textContent = `Evaluated: ${new Date(item.timestamp).toLocaleDateString()} ${new Date(item.timestamp).toLocaleTimeString()}`;
            li.appendChild(timestamp);
            
            problematicTermsList.appendChild(li);
        });
    }

    // Helper function to render orphan terms list
    function renderOrphanTermsList(orphanTerms) {
        if (!orphanTerms || orphanTerms.length === 0) {
            return '<p class="text-muted">No orphan terms found.</p>';
        }
        
        // Group terms by level
        const termsByLevel = {};
        orphanTerms.forEach(term => {
            const level = term.level !== undefined ? term.level : 'Unknown';
            if (!termsByLevel[level]) {
                termsByLevel[level] = [];
            }
            termsByLevel[level].push(term);
        });
        
        // Sort levels and generate HTML
        let html = '<div class="list-group">';
        Object.keys(termsByLevel)
            .sort((a, b) => {
                // Convert to number for numeric sorting if possible
                const numA = isNaN(a) ? 999 : Number(a);
                const numB = isNaN(b) ? 999 : Number(b);
                return numA - numB;
            })
            .forEach(level => {
                const terms = termsByLevel[level];
                html += `<div class="list-group-item">
                    <h6>Level ${level}</h6>
                    <div class="d-flex flex-wrap">
                        ${terms.map(term => `
                            <a href="/manual_evaluation?term=${encodeURIComponent(term.term)}" 
                               class="badge bg-secondary me-1 mb-1 text-decoration-none" 
                               target="_blank">${term.term}</a>
                        `).join('')}
                    </div>
                </div>`;
            });
        html += '</div>';
        
        return html;
    }
    
    // Helper function to render leaf terms list
    function renderLeafTermsList(leafTerms) {
        if (!leafTerms || leafTerms.length === 0) {
            return '<p class="text-muted">No leaf terms not at level 3 found.</p>';
        }
        
        // Group terms by level
        const termsByLevel = {};
        leafTerms.forEach(term => {
            const level = term.level !== undefined ? term.level : 'Unknown';
            if (!termsByLevel[level]) {
                termsByLevel[level] = [];
            }
            termsByLevel[level].push(term);
        });
        
        // Sort levels and generate HTML
        let html = '<div class="list-group">';
        Object.keys(termsByLevel)
            .sort((a, b) => {
                // Convert to number for numeric sorting if possible
                const numA = isNaN(a) ? 999 : Number(a);
                const numB = isNaN(b) ? 999 : Number(b);
                return numA - numB;
            })
            .forEach(level => {
                const terms = termsByLevel[level];
                html += `<div class="list-group-item">
                    <h6>Level ${level}</h6>
                    <div class="d-flex flex-wrap">
                        ${terms.map(term => `
                            <a href="/manual_evaluation?term=${encodeURIComponent(term.term)}" 
                               class="badge bg-secondary me-1 mb-1 text-decoration-none" 
                               target="_blank">${term.term}</a>
                        `).join('')}
                    </div>
                </div>`;
            });
        html += '</div>';
        
        return html;
    }

    // Initial load
    fetchAndDisplayStats();
}); 