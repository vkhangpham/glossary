// Main script for the hierarchy explorer
document.addEventListener('DOMContentLoaded', () => {
    let hierarchyData = null;
    let visualization = null;
    
    // Initialize visualization
    visualization = new HierarchyVisualization('hierarchy-visualization');
    visualization.init();
    
    // Set node click handler
    visualization.setNodeClickHandler(term => {
        showTermDetails(term);
    });
    
    // Setup tree mode toggle
    const treeModeToggle = document.getElementById('tree-mode-toggle');
    treeModeToggle.addEventListener('change', (event) => {
        visualization.setTreeMode(event.target.checked);
    });
    
    // Setup max parents input
    const parentsInput = document.getElementById('parents-input');
    parentsInput.addEventListener('change', (event) => {
        const value = parseInt(event.target.value);
        visualization.setMaxParents(value);
    });
    
    // Setup max siblings input
    const siblingsInput = document.getElementById('siblings-input');
    siblingsInput.addEventListener('change', (event) => {
        const value = parseInt(event.target.value);
        visualization.setMaxSiblings(value);
    });
    
    // Setup max children input
    const childrenInput = document.getElementById('children-input');
    childrenInput.addEventListener('change', (event) => {
        const value = parseInt(event.target.value);
        visualization.setMaxChildren(value);
    });
    
    // Setup fullscreen button
    const fullscreenBtn = document.getElementById('fullscreen-btn');
    fullscreenBtn.addEventListener('click', () => {
        visualization.toggleFullscreen();
        
        // Update button icon and text
        const container = document.querySelector('.visualization-container');
        if (container.classList.contains('fullscreen')) {
            fullscreenBtn.innerHTML = '<i class="bi bi-fullscreen-exit"></i> Exit Fullscreen';
        } else {
            fullscreenBtn.innerHTML = '<i class="bi bi-fullscreen"></i> Fullscreen';
        }
    });
    
    // Handle fullscreen exit with ESC key
    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape') {
            const container = document.querySelector('.visualization-container');
            if (container.classList.contains('fullscreen')) {
                visualization.toggleFullscreen();
                fullscreenBtn.innerHTML = '<i class="bi bi-fullscreen"></i> Fullscreen';
            }
        }
    });
    
    // Load hierarchy data
    fetch('/api/hierarchy')
        .then(response => response.json())
        .then(data => {
            hierarchyData = data;
            visualization.loadData(data);
            populateTermList(data);
        })
        .catch(error => console.error('Error loading hierarchy data:', error));
    
    // Handle search input
    const searchInput = document.getElementById('search-input');
    const searchResults = document.getElementById('search-results');
    
    searchInput.addEventListener('input', debounce(event => {
        const query = event.target.value.trim();
        if (query.length < 2) {
            searchResults.innerHTML = '';
            return;
        }
        
        fetch(`/api/search?q=${encodeURIComponent(query)}`)
            .then(response => response.json())
            .then(results => {
                if (results.length === 0) {
                    searchResults.innerHTML = '<div class="p-2">No results found</div>';
                    return;
                }
                
                let html = '';
                results.forEach(result => {
                    const levelBadge = result.level >= 0 ? 
                        `<span class="level-badge level-${result.level}-badge">${result.level}</span>` : '';
                    
                    if (result.type === 'term') {
                        html += `<div class="search-results-item" data-term="${result.term}">
                            ${levelBadge} ${result.term}
                        </div>`;
                    } else if (result.type === 'variation') {
                        html += `<div class="search-results-item" data-term="${result.term}">
                            ${levelBadge} ${result.variation} (variation of ${result.term})
                        </div>`;
                    }
                });
                
                searchResults.innerHTML = html;
                
                // Add click handlers to search results
                document.querySelectorAll('.search-results-item').forEach(item => {
                    item.addEventListener('click', () => {
                        const term = item.getAttribute('data-term');
                        showTermDetails(term);
                        searchInput.value = '';
                        searchResults.innerHTML = '';
                    });
                });
            })
            .catch(error => console.error('Error searching:', error));
    }, 300));
    
    // Handle level filters
    document.querySelectorAll('.level-filters input[type="checkbox"]').forEach(checkbox => {
        checkbox.addEventListener('change', () => {
            if (hierarchyData) {
                visualization.loadData(hierarchyData);
                populateTermList(hierarchyData);
            }
        });
    });
    
    // Populate term list
    function populateTermList(data) {
        const termList = document.getElementById('term-list');
        const level0Filter = document.getElementById('level-0-filter').checked;
        const level1Filter = document.getElementById('level-1-filter').checked;
        const level2Filter = document.getElementById('level-2-filter').checked;
        
        let html = '';
        
        // Add terms for each level
        for (let level = 0; level < 3; level++) {
            if ((level === 0 && level0Filter) || 
                (level === 1 && level1Filter) || 
                (level === 2 && level2Filter)) {
                
                // Get a subset of terms for display
                const levelTerms = data.levels[level];
                const displayCount = Math.min(50, levelTerms.length);
                const termsToDisplay = levelTerms.slice(0, displayCount);
                
                // Add level header
                html += `<li class="list-group-item list-group-item-secondary">
                    Level ${level} (${levelTerms.length} terms)
                </li>`;
                
                // Add terms to list
                termsToDisplay.forEach(term => {
                    html += `<li class="list-group-item term-list-level-${level}" data-term="${term}">
                        <span class="level-badge level-${level}-badge">${level}</span> ${term}
                    </li>`;
                });
                
                if (levelTerms.length > displayCount) {
                    html += `<li class="list-group-item text-muted">
                        ... and ${levelTerms.length - displayCount} more terms
                    </li>`;
                }
            }
        }
        
        termList.innerHTML = html;
        
        // Add click handlers to term list items
        document.querySelectorAll('#term-list li[data-term]').forEach(item => {
            item.addEventListener('click', () => {
                const term = item.getAttribute('data-term');
                showTermDetails(term);
            });
        });
    }
    
    // Show term details
    function showTermDetails(term) {
        if (!hierarchyData || !hierarchyData.terms[term]) {
            return;
        }
        
        // Update selected term in visualization
        visualization.updateSelectedTerm(term);
        
        // Update term details in UI
        const termData = hierarchyData.terms[term];
        
        document.getElementById('selected-term').textContent = term;
        document.getElementById('term-level').textContent = termData.level;
        
        // Update parents
        const parentsElement = document.getElementById('term-parents');
        if (termData.parents.length > 0) {
            let parentsHtml = '';
            termData.parents.forEach(parent => {
                parentsHtml += `<span class="badge bg-light text-dark me-1 parent-term" data-term="${parent}">${parent}</span>`;
            });
            parentsElement.innerHTML = parentsHtml;
            
            // Add click handlers to parent terms
            document.querySelectorAll('.parent-term').forEach(item => {
                item.addEventListener('click', () => {
                    const parentTerm = item.getAttribute('data-term');
                    showTermDetails(parentTerm);
                });
            });
        } else {
            parentsElement.innerHTML = '<em>None</em>';
        }
        
        // Update children
        const childrenElement = document.getElementById('term-children');
        if (termData.children.length > 0) {
            let childrenHtml = '';
            
            // Show count if there are many children
            if (termData.children.length > 10) {
                childrenHtml = `<div class="mb-2">${termData.children.length} children (showing first 10):</div>`;
            }
            
            // Show first 10 children with links
            const displayChildren = termData.children.slice(0, 10);
            displayChildren.forEach(child => {
                childrenHtml += `<span class="badge bg-light text-dark me-1 child-term" data-term="${child}">${child}</span>`;
            });
            
            // If there are more children, add a show more button
            if (termData.children.length > 10) {
                childrenHtml += `
                <div class="mt-2">
                    <button class="btn btn-sm btn-outline-secondary" id="show-more-children">
                        Show all ${termData.children.length} children
                    </button>
                </div>`;
            }
            
            childrenElement.innerHTML = childrenHtml;
            
            // Add click handlers to child terms
            document.querySelectorAll('.child-term').forEach(item => {
                item.addEventListener('click', () => {
                    const childTerm = item.getAttribute('data-term');
                    showTermDetails(childTerm);
                });
            });
            
            // Add handler for show more button
            const showMoreBtn = document.getElementById('show-more-children');
            if (showMoreBtn) {
                showMoreBtn.addEventListener('click', () => {
                    let allChildrenHtml = '';
                    termData.children.forEach(child => {
                        allChildrenHtml += `<span class="badge bg-light text-dark me-1 mb-1 child-term" data-term="${child}">${child}</span>`;
                    });
                    childrenElement.innerHTML = allChildrenHtml;
                    
                    // Re-add click handlers
                    document.querySelectorAll('.child-term').forEach(item => {
                        item.addEventListener('click', () => {
                            const childTerm = item.getAttribute('data-term');
                            showTermDetails(childTerm);
                        });
                    });
                });
            }
        } else {
            childrenElement.innerHTML = '<em>None</em>';
        }
        
        // Update variations
        const variationsElement = document.getElementById('term-variations');
        if (termData.variations.length > 0) {
            variationsElement.innerHTML = termData.variations.map(v => 
                `<span class="badge bg-light text-dark me-1">${v}</span>`
            ).join('');
        } else {
            variationsElement.innerHTML = '<em>None</em>';
        }
        
        // Update sources
        const sourcesElement = document.getElementById('term-sources');
        if (termData.sources.length > 0) {
            let sourcesHtml = '';
            
            // Show count if there are many sources
            if (termData.sources.length > 5) {
                sourcesHtml = `<div class="mb-2">${termData.sources.length} sources (showing first 5):</div>`;
            }
            
            // Show first 5 sources
            const displaySources = termData.sources.slice(0, 5);
            sourcesHtml += displaySources.map(s => 
                `<span class="badge bg-light text-dark me-1">${s}</span>`
            ).join('');
            
            // If there are more sources, add a show more button
            if (termData.sources.length > 5) {
                sourcesHtml += `
                <div class="mt-2">
                    <button class="btn btn-sm btn-outline-secondary" id="show-more-sources">
                        Show all ${termData.sources.length} sources
                    </button>
                </div>`;
            }
            
            sourcesElement.innerHTML = sourcesHtml;
            
            // Add handler for show more button
            const showMoreBtn = document.getElementById('show-more-sources');
            if (showMoreBtn) {
                showMoreBtn.addEventListener('click', () => {
                    sourcesElement.innerHTML = termData.sources.map(s => 
                        `<span class="badge bg-light text-dark me-1 mb-1">${s}</span>`
                    ).join('');
                });
            }
        } else {
            sourcesElement.innerHTML = '<em>None</em>';
        }
        
        // Update resources
        const resourcesListElement = document.getElementById('resources-list');
        if (termData.resources && termData.resources.length > 0) {
            let resourcesHtml = '';
            termData.resources.forEach(resource => {
                resourcesHtml += `
                <div class="card resource-card mb-3">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h5 class="mb-0">${resource.title}</h5>
                        <span class="badge bg-primary">Score: ${resource.relevance_score.toFixed(2)}</span>
                    </div>
                    <div class="card-body">
                        <div class="resource-content">
                            <p>${resource.processed_content}</p>
                        </div>
                        <a href="${resource.url}" class="btn btn-sm btn-outline-primary mt-2" target="_blank">
                            Visit Source
                        </a>
                    </div>
                </div>`;
            });
            resourcesListElement.innerHTML = resourcesHtml;
        } else {
            resourcesListElement.innerHTML = '<div class="alert alert-info">No resources available</div>';
        }
        
        // Update breadcrumbs
        updateBreadcrumbs(term, termData.level);
    }
    
    // Update breadcrumbs
    function updateBreadcrumbs(term, level) {
        const breadcrumbList = document.getElementById('breadcrumb-list');
        
        // Reset breadcrumbs
        breadcrumbList.innerHTML = `<li class="breadcrumb-item"><a href="#" id="breadcrumb-home">Home</a></li>`;
        
        // Add level
        breadcrumbList.innerHTML += `
            <li class="breadcrumb-item">
                <span class="level-badge level-${level}-badge">${level}</span> ${term}
            </li>`;
            
        // Add home click handler
        document.getElementById('breadcrumb-home').addEventListener('click', event => {
            event.preventDefault();
            visualization.updateSelectedTerm(null);
            document.getElementById('selected-term').textContent = 'Select a term to view details';
            document.getElementById('term-level').textContent = '';
            document.getElementById('term-parents').innerHTML = '';
            document.getElementById('term-children').innerHTML = '';
            document.getElementById('term-variations').innerHTML = '';
            document.getElementById('term-sources').innerHTML = '';
            document.getElementById('resources-list').innerHTML = '';
            
            // Reset breadcrumbs
            breadcrumbList.innerHTML = `<li class="breadcrumb-item active">Home</li>`;
        });
    }
    
    // Utility function for debouncing
    function debounce(func, wait) {
        let timeout;
        return function(...args) {
            const context = this;
            clearTimeout(timeout);
            timeout = setTimeout(() => func.apply(context, args), wait);
        };
    }
});