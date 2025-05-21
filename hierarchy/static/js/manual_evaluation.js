document.addEventListener('DOMContentLoaded', function() {
    const searchInput = document.getElementById('search-input');
    const searchResults = document.getElementById('search-results');
    const termList = document.getElementById('term-list');
    const selectedTermHeader = document.getElementById('selected-term-header');
    const evaluationForm = document.getElementById('evaluation-form');
    const termNameInput = document.getElementById('term-name');
    const termDetailsDisplay = document.getElementById('term-details-display');
    const termLevelDisplay = document.getElementById('term-level');
    const termParentsList = document.getElementById('term-parents-list');
    const termVariationsList = document.getElementById('term-variations-list');
    const currentLevelDisplay = document.getElementById('current-level-display');
    const variationsChecklist = document.getElementById('variations-checklist');
    const parentsChecklist = document.getElementById('parents-checklist');
    const clearFormBtn = document.getElementById('clear-form-btn');
    const saveNotification = document.getElementById('save-notification');
    const sampleLevelSelect = document.getElementById('sample-level-select');
    const filterLevelSelect = document.getElementById('filter-level-select');
    const getRandomSampleBtn = document.getElementById('get-random-sample-btn');
    const showEvaluatedOnlyCheckbox = document.getElementById('show-evaluated-only');
    const evaluatedCountBadge = document.getElementById('evaluated-count');
    const formNav = document.getElementById('form-nav');
    const toast = new bootstrap.Toast(saveNotification);

    let allTermsData = {}; // Store all hierarchy data
    let currentTermData = null; // Store data for the currently selected term
    let currentlyDisplayedTerms = []; // Keep track of terms currently in the list
    let evaluatedTermsSet = new Set(); // Keep track of terms with saved evaluations
    let filterSettings = {
        level: 'all',
        evaluatedOnly: false,
        searchQuery: ''
    };

    // --- Initialization ---
    // Check URL parameters
    const urlParams = new URLSearchParams(window.location.search);
    
    // If evaluated=true is in the URL, set the filter checkbox
    if (urlParams.get('evaluated') === 'true') {
        showEvaluatedOnlyCheckbox.checked = true;
        filterSettings.evaluatedOnly = true;
    }
    
    // If level=X is in the URL, set the filter level
    if (urlParams.get('level')) {
        const levelValue = urlParams.get('level');
        if (['all', '0', '1', '2', '3'].includes(levelValue)) {
            filterLevelSelect.value = levelValue;
            filterSettings.level = levelValue;
        }
    }
    
    // If a specific term is requested in the URL, we'll load it after data is initialized
    const requestedTerm = urlParams.get('term');
    
    fetchHierarchyData(requestedTerm);
    fetchEvaluatedTerms();

    // --- Event Listeners ---
    searchInput.addEventListener('input', handleSearch);
    evaluationForm.addEventListener('submit', handleFormSubmit);
    clearFormBtn.addEventListener('click', clearEvaluationForm);
    getRandomSampleBtn.addEventListener('click', handleGetRandomSample);
    showEvaluatedOnlyCheckbox.addEventListener('change', handleFilterChange);
    
    // Add event listeners for instant filtering
    filterLevelSelect.addEventListener('change', handleFilterChange);

    // --- Functions ---
    async function fetchHierarchyData(termToSelect = null) {
        try {
            const response = await fetch('/api/hierarchy');
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            allTermsData = await response.json();
            currentlyDisplayedTerms = Object.keys(allTermsData.terms);
            
            // Apply filters to the initial data
            applyFilters();
            
            // If a specific term was requested, select it
            if (termToSelect && allTermsData.terms[termToSelect]) {
                setTimeout(() => selectTerm(termToSelect), 100); // Small delay to ensure UI is ready
            }
        } catch (error) {
            console.error('Error fetching hierarchy data:', error);
            termList.innerHTML = '<li class="list-group-item text-danger">Failed to load terms.</li>';
        }
    }

    async function fetchEvaluatedTerms() {
        try {
            const response = await fetch('/api/all_evaluations');
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const data = await response.json();
            evaluatedTermsSet = new Set(Object.keys(data));
            console.log(`Fetched ${evaluatedTermsSet.size} evaluated terms.`);
            
            // Update the evaluated count badge
            updateEvaluatedCountBadge();
            
            // Re-populate list if data is already loaded to apply styling
            if (Object.keys(allTermsData).length > 0) {
                applyFilters(); // Apply filters instead of just repopulating
            }
        } catch (error) {
            console.error('Error fetching evaluated terms:', error);
            // Optionally inform the user, but continue operation
        }
    }

    function populateTermList(termsToShow) {
        termList.innerHTML = ''; // Clear existing list

        // Group terms by level using allTermsData
        const termsByLevel = {};
        termsToShow.forEach(term => {
            const termData = allTermsData.terms[term];
            if (termData) {
                const level = termData.level;
                if (!termsByLevel[level]) {
                    termsByLevel[level] = [];
                }
                termsByLevel[level].push(term);
            }
        });

        if (Object.keys(termsByLevel).length === 0 && termsToShow.length > 0) {
            // This might happen if termsToShow contains terms not in allTermsData
            console.warn("Some terms to show were not found in the main hierarchy data.");
             // Fallback: display as flat list if grouping failed
             termsToShow.sort((a, b) => a.toLowerCase().localeCompare(b.toLowerCase()));
             termsToShow.forEach(term => {
                const li = document.createElement('li');
                li.className = 'list-group-item list-group-item-action';
                li.textContent = term;
                li.dataset.term = term;
                li.addEventListener('click', () => selectTerm(term));
                termList.appendChild(li);
            });
             if (termsToShow.length === 0) {
                termList.innerHTML = '<li class="list-group-item">No terms found for current filter.</li>';
             }
            return;
        }
         if (Object.keys(termsByLevel).length === 0) {
             termList.innerHTML = '<li class="list-group-item">No terms found.</li>';
            return;
        }

        // Sort levels numerically and display
        Object.keys(termsByLevel).sort((a, b) => a - b).forEach(level => {
            // Add level header
            const levelHeader = document.createElement('li');
            levelHeader.className = 'list-group-item list-group-item-secondary fw-bold';
            levelHeader.textContent = `Level ${level}`;
            termList.appendChild(levelHeader);

            // Sort terms within the level alphabetically
            termsByLevel[level].sort((a, b) => a.toLowerCase().localeCompare(b.toLowerCase()));

            // Add terms for this level
            termsByLevel[level].forEach(term => {
                const li = document.createElement('li');
                li.className = 'list-group-item list-group-item-action d-flex justify-content-between align-items-center';
                li.dataset.term = term;

                const termText = document.createElement('span');
                termText.textContent = term;
                li.appendChild(termText);

                // Check if this term is the currently selected one
                if (termNameInput.value === term) {
                    li.classList.add('active');
                }

                // Check if evaluated
                if (evaluatedTermsSet.has(term)) {
                    const checkIcon = document.createElement('i');
                    checkIcon.className = 'bi bi-check-circle-fill evaluated-icon text-success';
                    checkIcon.title = 'Evaluated';
                    li.appendChild(checkIcon);
                    li.classList.add('evaluated'); // Add class for potential specific styling
                }

                li.addEventListener('click', () => selectTerm(term));
                termList.appendChild(li);
            });
        });
    }

    async function handleSearch() {
        const query = searchInput.value.toLowerCase();
        filterSettings.searchQuery = query;
        
        if (!query) {
            searchResults.innerHTML = '';
            // Reset to the filtered list without search query
            filterSettings.searchQuery = '';
            applyFilters();
            return;
        }

        if (query.length < 2) { // Avoid searching for very short strings
            searchResults.innerHTML = '';
            return;
        }

        try {
            const response = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const results = await response.json();
            
            // Display search results and apply other filters
            displaySearchResults(results);
        } catch (error) {
            console.error('Error during search:', error);
            searchResults.innerHTML = '<div class="text-danger">Search failed.</div>';
        }
    }

    function displaySearchResults(results) {
        searchResults.innerHTML = ''; // Clear previous results
        if (!results || results.length === 0) {
            searchResults.innerHTML = '<div class="text-muted">No results found.</div>';
            populateTermList([]); // Clear term list if search yields no results
            return;
        }

        // Extract terms from search results
        const resultTerms = new Set();
        results.forEach(result => {
            if (result.term && allTermsData.terms[result.term]) {
                resultTerms.add(result.term);
            }
        });
        
        // Apply other filters to search results
        let filteredResults = Array.from(resultTerms);
        
        // Filter by level if specified
        if (filterSettings.level !== 'all') {
            const levelNum = parseInt(filterSettings.level);
            filteredResults = filteredResults.filter(term => {
                const termData = allTermsData.terms[term];
                return termData && termData.level !== undefined && parseInt(termData.level, 10) === levelNum;
            });
        }
        
        // Filter by evaluation status if needed
        if (filterSettings.evaluatedOnly) {
            filteredResults = filteredResults.filter(term => evaluatedTermsSet.has(term));
        }
        
        // The search query is handled by the API, no need to re-filter by searchQuery here.
        
        // Update the current display and populate the list
        currentlyDisplayedTerms = filteredResults;
        populateTermList(currentlyDisplayedTerms);
    }

    async function selectTerm(term) {
        console.log('Selecting term:', term);
        selectedTermHeader.textContent = `Evaluating: ${term}`;
        termNameInput.value = term;
        
        // Update URL to include the selected term
        updateUrlParams();

        // Fetch detailed term data directly from the endpoint
        try {
            const response = await fetch(`/api/load_evaluation/${encodeURIComponent(term)}`);
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const data = await response.json();
            
            // Apply the term details
            if (data.term_details) {
                displayTermDetails(data.term_details);
                termDetailsDisplay.style.display = 'block';
                
                // Also update our reference to current term data
                currentTermData = { ...data.term_details };
                
                // Populate form sections based on term details
                populateLevelCorrectness(data.term_details.level);
                populateVariationsChecklist(term, data.term_details.variations || []);
                populateParentsChecklist(data.term_details.parents || []);
                populateRelatedConceptsSection(data.term_details.related_concepts || {});
                
                // Set default academic importance *before* loading saved evaluation
                setDefaultAcademicImportance();
                
                // Apply any existing evaluation
                if (data.is_evaluated && data.evaluation) {
                    applyEvaluationData(data.evaluation);
                }
                
                // Show the form and navigation
                evaluationForm.style.display = 'block';
                formNav.style.display = 'block';
                
                // Update UI to indicate evaluation status
                updateEvaluationStatusIndicator(term, data.is_evaluated);
                
                // Initialize form navigation
                initFormNavigation();
            } else {
                console.error('Term details not found for:', term);
                clearEvaluationForm();
                evaluationForm.style.display = 'none';
                formNav.style.display = 'none';
                termDetailsDisplay.style.display = 'none';
            }
        } catch (error) {
            console.error('Error fetching term details:', error);
            clearEvaluationForm();
            evaluationForm.style.display = 'none';
            formNav.style.display = 'none';
            termDetailsDisplay.style.display = 'none';
        }
    }

    function updateEvaluationStatusIndicator(term, isEvaluated) {
        // Update the term entry in the list to show evaluation status
        Array.from(termList.children).forEach(li => {
            if (li.dataset.term === term) {
                li.classList.toggle('active', true);
                
                // If this is newly evaluated, add the class for the green flash effect
                if (isEvaluated && !li.classList.contains('evaluated')) {
                    li.classList.add('just-evaluated');
                    setTimeout(() => {
                        li.classList.remove('just-evaluated');
                    }, 2000);
                }
                
                // Add or remove the evaluation icon
                const existingIcon = li.querySelector('.evaluated-icon');
                if (isEvaluated && !existingIcon) {
                    const checkIcon = document.createElement('i');
                    checkIcon.className = 'bi bi-check-circle-fill evaluated-icon text-success';
                    checkIcon.title = 'Evaluated';
                    li.appendChild(checkIcon);
                    li.classList.add('evaluated');
                }
            } else {
                li.classList.toggle('active', false);
            }
        });
    }

    function displayTermDetails(termDetails) {
        termLevelDisplay.textContent = termDetails.level !== undefined ? termDetails.level : 'N/A';
        
        // Render parents as clickable badges
        const parentsContent = document.createDocumentFragment();
        if (termDetails.parents && termDetails.parents.length > 0) {
            termDetails.parents.forEach((parent, index) => {
                const badge = document.createElement('span');
                badge.className = 'term-badge';
                badge.textContent = parent;
                badge.addEventListener('click', () => selectTerm(parent));
                parentsContent.appendChild(badge);
                
                // Add a space between badges
                if (index < termDetails.parents.length - 1) {
                    parentsContent.appendChild(document.createTextNode(' '));
                }
            });
        } else {
            parentsContent.appendChild(document.createTextNode('None'));
        }
        termParentsList.innerHTML = '';
        termParentsList.appendChild(parentsContent);
        
        // Render variations as badges
        const variationsContent = document.createDocumentFragment();
        if (termDetails.variations && termDetails.variations.length > 0) {
            termDetails.variations.forEach((variation, index) => {
                const badge = document.createElement('span');
                badge.className = 'term-badge';
                badge.textContent = variation;
                variationsContent.appendChild(badge);
                
                // Add a space between badges
                if (index < termDetails.variations.length - 1) {
                    variationsContent.appendChild(document.createTextNode(' '));
                }
            });
        } else {
            variationsContent.appendChild(document.createTextNode('None'));
        }
        termVariationsList.innerHTML = '';
        termVariationsList.appendChild(variationsContent);
        
        // Add definition if available
        if (!document.getElementById('term-definition-row')) {
            const definitionRow = document.createElement('div');
            definitionRow.className = 'row mb-2';
            definitionRow.id = 'term-definition-row';
            definitionRow.innerHTML = `
                <dt class="col-sm-3">Definition</dt>
                <dd class="col-sm-9" id="term-definition"></dd>
            `;
            termDetailsDisplay.querySelector('dl').appendChild(definitionRow);
        }
        
        document.getElementById('term-definition').textContent = termDetails.definition || 'No definition available';
        
        // Add related concepts if available
        if (!document.getElementById('term-related-concepts-row')) {
            const relatedConceptsRow = document.createElement('div');
            relatedConceptsRow.className = 'row mb-2';
            relatedConceptsRow.id = 'term-related-concepts-row';
            relatedConceptsRow.innerHTML = `
                <dt class="col-sm-3">Related Concepts</dt>
                <dd class="col-sm-9" id="term-related-concepts-list"></dd>
            `;
            termDetailsDisplay.querySelector('dl').appendChild(relatedConceptsRow);
        }
        
        const relatedConceptsList = document.getElementById('term-related-concepts-list');
        
        if (termDetails.related_concepts && Object.keys(termDetails.related_concepts).length > 0) {
            // Format related concepts by level
            const formattedRelatedConcepts = [];
            
            // Create an array of keys and sort them numerically after normalizing level format
            const sortedLevels = Object.keys(termDetails.related_concepts).map(level => {
                // Convert 'lvX' to numeric value for sorting
                if (typeof level === 'string' && level.startsWith('lv')) {
                    return parseInt(level.substring(2));
                }
                return parseInt(level);
            }).sort((a, b) => a - b);
            
            sortedLevels.forEach(numericLevel => {
                // Need to find the original key back, could be 'lvX' or numeric
                const originalKey = Object.keys(termDetails.related_concepts).find(key => {
                    if (key.startsWith('lv')) {
                        return parseInt(key.substring(2)) === numericLevel;
                    }
                    return parseInt(key) === numericLevel;
                });
                
                if (originalKey) {
                    const concepts = termDetails.related_concepts[originalKey];
                    if (concepts && concepts.length > 0) {
                        // Create a more interactive UI for related concepts
                        const levelDiv = document.createElement('div');
                        levelDiv.className = 'mb-2';
                        
                        const levelTitle = document.createElement('strong');
                        levelTitle.textContent = `Level ${numericLevel}: `;
                        levelDiv.appendChild(levelTitle);
                        
                        const conceptsFragment = document.createDocumentFragment();
                        concepts.forEach((concept, index) => {
                            const badge = document.createElement('span');
                            badge.className = 'term-badge';
                            badge.textContent = concept;
                            badge.addEventListener('click', () => selectTerm(concept));
                            conceptsFragment.appendChild(badge);
                            
                            // Add a space between badges
                            if (index < concepts.length - 1) {
                                conceptsFragment.appendChild(document.createTextNode(' '));
                            }
                        });
                        
                        levelDiv.appendChild(conceptsFragment);
                        formattedRelatedConcepts.push(levelDiv);
                    }
                }
            });
            
            relatedConceptsList.innerHTML = '';
            if (formattedRelatedConcepts.length > 0) {
                formattedRelatedConcepts.forEach(div => {
                    relatedConceptsList.appendChild(div);
                });
            } else {
                relatedConceptsList.textContent = 'None';
            }
        } else {
            relatedConceptsList.textContent = 'None';
        }
        
        // Update progress bar
        updateProgressBar();
    }

    function populateLevelCorrectness(currentLevel) {
        currentLevelDisplay.textContent = currentLevel !== undefined ? currentLevel : 'N/A';
        evaluationForm.querySelectorAll('input[name="level_correctness"]').forEach(radio => radio.checked = false);

        const numericCurrentLevel = currentLevel !== undefined ? parseInt(currentLevel, 10) : NaN;

         for (let i = 0; i <= 3; i++) {
             const radioInput = document.getElementById(`level-${i}`);
             const radioContainer = radioInput ? radioInput.closest('.form-check') : null;
             if (radioContainer) {
                 // Hide the radio button if its level (i) matches the term's current level (numericCurrentLevel)
                 if (!isNaN(numericCurrentLevel) && i === numericCurrentLevel) {
                     radioContainer.style.display = 'none'; 
                 } else {
                     radioContainer.style.display = 'inline-block'; 
                 }
             }
         }

        // Check 'correct' by default if level is valid
        if (currentLevel !== undefined && currentLevel !== null && !isNaN(numericCurrentLevel)) {
            const correctRadio = document.getElementById('level-correct');
            if (correctRadio) correctRadio.checked = true;
        }
    }

    function populateVariationsChecklist(term, variations) {
        variationsChecklist.innerHTML = '';
        
        if (!variations || variations.length === 0) {
            variationsChecklist.innerHTML = '<p class="text-muted">No variations listed for this term.</p>';
            return;
        }
        
        variations.forEach(variation => {
            const div = document.createElement('div');
            div.className = 'form-check';
            
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.className = 'form-check-input';
            checkbox.id = `variation-${variation.replace(/\s+/g, '_')}`;
            checkbox.name = `variation_correctness[${variation}]`;
            checkbox.checked = true; // Default to true
            
            const label = document.createElement('label');
            label.className = 'form-check-label';
            label.htmlFor = checkbox.id;
            label.textContent = variation;
            
            div.appendChild(checkbox);
            div.appendChild(label);
            variationsChecklist.appendChild(div);
        });
    }
    
    function populateParentsChecklist(parents) {
        parentsChecklist.innerHTML = '';
        
        if (!parents || parents.length === 0) {
            parentsChecklist.innerHTML = '<p class="text-muted">No parents listed for this term.</p>';
            return;
        }
        
        parents.forEach(parent => {
            const div = document.createElement('div');
            div.className = 'form-check';
            
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.className = 'form-check-input';
            checkbox.id = `parent-${parent.replace(/\s+/g, '_')}`;
            checkbox.name = `parent_relationships[${parent}]`;
            checkbox.checked = true; // Default to true
            
            const label = document.createElement('label');
            label.className = 'form-check-label';
            label.htmlFor = checkbox.id;
            label.textContent = parent;
            
            div.appendChild(checkbox);
            div.appendChild(label);
            parentsChecklist.appendChild(div);
        });
    }
    
    function populateRelatedConceptsSection(relatedConcepts) {
        // First, check if the related concepts section already exists in the form
        let relatedConceptsSection = document.getElementById('related-concepts-section');
        
        if (!relatedConceptsSection) {
            // Create the section if it doesn't exist
            relatedConceptsSection = document.createElement('div');
            relatedConceptsSection.className = 'evaluation-criteria';
            relatedConceptsSection.id = 'related-concepts-section';
            
            const title = document.createElement('h5');
            title.innerHTML = '<i class="bi bi-link-45deg me-2"></i>5. Related Concepts Evaluation';
            
            const description = document.createElement('p');
            description.textContent = 'Are there any related concepts that should be parents or variations of this term?';
            
            relatedConceptsSection.appendChild(title);
            relatedConceptsSection.appendChild(description);
            
            // Create divs for potential parents and variations
            const potentialParentsDiv = document.createElement('div');
            potentialParentsDiv.id = 'potential-parents-checklist';
            potentialParentsDiv.innerHTML = '<h6>Potential Missing Parents</h6><p class="small text-muted">Showing only terms from one level above the current term.</p>';
            
            const potentialVariationsDiv = document.createElement('div');
            potentialVariationsDiv.id = 'potential-variations-checklist';
            potentialVariationsDiv.innerHTML = '<h6>Potential Missing Variations</h6><p class="small text-muted">Showing top 3 related terms per level within +/-1 level range.</p>';
            
            // Create notes textarea
            const notesDiv = document.createElement('div');
            notesDiv.className = 'mt-3';
            const notesTextarea = document.createElement('textarea');
            notesTextarea.className = 'form-control evaluation-notes';
            notesTextarea.name = 'related_concepts_notes';
            notesTextarea.placeholder = 'Optional notes about related concepts...';
            notesDiv.appendChild(notesTextarea);
            
            relatedConceptsSection.appendChild(potentialParentsDiv);
            relatedConceptsSection.appendChild(potentialVariationsDiv);
            relatedConceptsSection.appendChild(notesDiv);
            
            // Add the section to the designated container
            document.getElementById('section-5').appendChild(relatedConceptsSection);
        }
        
        // Now populate the related concepts
        const potentialParentsChecklist = document.getElementById('potential-parents-checklist');
        const potentialVariationsChecklist = document.getElementById('potential-variations-checklist');
        
        // Clear existing items (keep the header and description)
        while (potentialParentsChecklist.childNodes.length > 2) {
            potentialParentsChecklist.removeChild(potentialParentsChecklist.lastChild);
        }
        
        while (potentialVariationsChecklist.childNodes.length > 2) {
            potentialVariationsChecklist.removeChild(potentialVariationsChecklist.lastChild);
        }
        
        if (!currentTermData || currentTermData.level === undefined || currentTermData.level === null) {
            potentialParentsChecklist.innerHTML += '<p class="text-muted">No term level information available.</p>';
            potentialVariationsChecklist.innerHTML += '<p class="text-muted">No term level information available.</p>';
            return;
        }

        // Convert related concepts keys from 'lvX' format to numerical format
        const normalizedRelatedConcepts = {};
        Object.keys(relatedConcepts).forEach(key => {
            // Extract the numeric part if key is in 'lvX' format
            if (typeof key === 'string' && key.startsWith('lv')) {
                const numericLevel = parseInt(key.substring(2));
                if (!isNaN(numericLevel)) {
                    normalizedRelatedConcepts[numericLevel] = relatedConcepts[key];
                }
            } else {
                // Keep as is if already numeric or in an unexpected format
                normalizedRelatedConcepts[key] = relatedConcepts[key];
            }
        });

        // Current term's level as a number
        const currentLevel = parseInt(currentTermData.level);
        
        // Get parent level (one level above)
        const parentLevel = currentLevel - 1;
        
        // Only apply parent filter if parent level is valid (>= 0)
        let potentialParentConcepts = [];
        if (parentLevel >= 0 && normalizedRelatedConcepts[parentLevel]) {
            potentialParentConcepts = [...normalizedRelatedConcepts[parentLevel]];
        }
        
        // For variations, get related terms from current level, one level above, and one level below
        const variationLevels = [currentLevel - 1, currentLevel, currentLevel + 1].filter(level => level >= 0 && level <= 3);
        
        // For each level, get top 3 related terms for variation candidates
        let potentialVariationConcepts = {};
        variationLevels.forEach(level => {
            if (normalizedRelatedConcepts[level] && normalizedRelatedConcepts[level].length > 0) {
                // Sort alphabetically and take top 3
                potentialVariationConcepts[level] = [...normalizedRelatedConcepts[level]].sort().slice(0, 3);
            }
        });
        
        // Add potential parents (if any)
        if (potentialParentConcepts.length === 0) {
            potentialParentsChecklist.innerHTML += '<p class="text-muted">No potential parents available from the level above.</p>';
        } else {
            // Sort alphabetically for better UX
            potentialParentConcepts.sort();
            
            potentialParentConcepts.forEach(concept => {
                // Skip if the concept is already a parent
                const isAlreadyParent = currentTermData.parents && currentTermData.parents.includes(concept);
                
                // Parent checkbox (if not already a parent)
                if (!isAlreadyParent) {
                    const parentDiv = document.createElement('div');
                    parentDiv.className = 'form-check';
                    
                    const parentCheckbox = document.createElement('input');
                    parentCheckbox.type = 'checkbox';
                    parentCheckbox.className = 'form-check-input';
                    parentCheckbox.id = `potential-parent-${concept.replace(/\s+/g, '_')}`;
                    parentCheckbox.name = `potential_parents[${concept}]`;
                    
                    const parentLabel = document.createElement('label');
                    parentLabel.className = 'form-check-label';
                    parentLabel.htmlFor = parentCheckbox.id;
                    parentLabel.textContent = concept;
                    
                    if (allTermsData.terms[concept]) {
                        const conceptLevel = allTermsData.terms[concept].level;
                        parentLabel.textContent += ` (Level ${conceptLevel})`;
                    }
                    
                    parentDiv.appendChild(parentCheckbox);
                    parentDiv.appendChild(parentLabel);
                    potentialParentsChecklist.appendChild(parentDiv);
                }
            });
        }
        
        // Add potential variations (grouped by level)
        let hasVariations = false;
        variationLevels.forEach(level => {
            if (potentialVariationConcepts[level] && potentialVariationConcepts[level].length > 0) {
                hasVariations = true;
                
                // Add level header
                const levelHeader = document.createElement('h6');
                levelHeader.className = 'mt-3 mb-2';
                levelHeader.textContent = `Level ${level}`;
                potentialVariationsChecklist.appendChild(levelHeader);
                
                potentialVariationConcepts[level].forEach(concept => {
                    // Skip if the concept is already a variation
                    const isAlreadyVariation = currentTermData.variations && currentTermData.variations.includes(concept);
                    
                    // Variation checkbox (if not already a variation)
                    if (!isAlreadyVariation) {
                        const variationDiv = document.createElement('div');
                        variationDiv.className = 'form-check';
                        
                        const variationCheckbox = document.createElement('input');
                        variationCheckbox.type = 'checkbox';
                        variationCheckbox.className = 'form-check-input';
                        variationCheckbox.id = `potential-variation-${concept.replace(/\s+/g, '_')}`;
                        variationCheckbox.name = `potential_variations[${concept}]`;
                        
                        const variationLabel = document.createElement('label');
                        variationLabel.className = 'form-check-label';
                        variationLabel.htmlFor = variationCheckbox.id;
                        variationLabel.textContent = concept;
                        
                        variationDiv.appendChild(variationCheckbox);
                        variationDiv.appendChild(variationLabel);
                        potentialVariationsChecklist.appendChild(variationDiv);
                    }
                });
            }
        });
        
        if (!hasVariations) {
            potentialVariationsChecklist.innerHTML += '<p class="text-muted">No potential variations available within the level range.</p>';
        }
    }

    function setDefaultAcademicImportance() {
        // Set default only if nothing is checked yet
        const checkedRadio = evaluationForm.querySelector('input[name="academic_importance"]:checked');
        if (!checkedRadio) {
             const defaultRadio = document.getElementById('importance-field'); // Default to 'Subject of Research'
             if (defaultRadio) {
                 defaultRadio.checked = true;
             }
        }
    }

    function applyEvaluationData(evalData) {
        // Academic Importance
        if (evalData.academic_importance) {
            const radio = evaluationForm.querySelector(`input[name="academic_importance"][value="${evalData.academic_importance}"]`);
            if (radio) radio.checked = true;
        }
        if (evalData.academic_importance_notes) {
            evaluationForm.querySelector('[name="academic_importance_notes"]').value = evalData.academic_importance_notes;
        }

        // Level Correctness
        if (evalData.level_correctness) {
            const radio = evaluationForm.querySelector(`input[name="level_correctness"][value="${evalData.level_correctness}"]`);
            if (radio) radio.checked = true;
        }
        if (evalData.level_correctness_notes) {
            evaluationForm.querySelector('[name="level_correctness_notes"]').value = evalData.level_correctness_notes;
        }

        // Variation Correctness
        if (evalData.variation_correctness && typeof evalData.variation_correctness === 'object') {
            for (const variation in evalData.variation_correctness) {
                 const checkbox = variationsChecklist.querySelector(`input[name="variation_${variation}"]`);
                 if (checkbox) checkbox.checked = evalData.variation_correctness[variation];
            }
        }
        if (evalData.variation_correctness_notes) {
             evaluationForm.querySelector('[name="variation_correctness_notes"]').value = evalData.variation_correctness_notes;
        }

        // Parent Relationship Correctness
        if (evalData.parent_relationships && typeof evalData.parent_relationships === 'object') {
            for (const parent in evalData.parent_relationships) {
                 const checkbox = parentsChecklist.querySelector(`input[name="parent_${parent}"]`);
                 if (checkbox) checkbox.checked = evalData.parent_relationships[parent];
            }
        }
        if (evalData.parent_relationship_notes) {
             evaluationForm.querySelector('[name="parent_relationship_notes"]').value = evalData.parent_relationship_notes;
        }
        
        // Update progress bar
        updateProgressBar();
    }

    function clearEvaluationForm(clearTerm = true) {
        evaluationForm.reset(); // Resets form elements to default values
        if (clearTerm) {
            termNameInput.value = '';
            selectedTermHeader.textContent = 'Select a term from the list';
            evaluationForm.style.display = 'none';
            formNav.style.display = 'none';
            termDetailsDisplay.style.display = 'none';
            // Clear dynamic lists
            variationsChecklist.innerHTML = '<p class="text-muted">Select a term.</p>';
            parentsChecklist.innerHTML = '<p class="text-muted">Select a term.</p>';
            // Make sure all level options are visible when term is cleared
            for (let i = 0; i <= 3; i++) {
                const radioInput = document.getElementById(`level-${i}`);
                const radioContainer = radioInput ? radioInput.closest('.form-check') : null;
                if (radioContainer) {
                    radioContainer.style.display = 'inline-block';
                }
            }
            // Unselect active term in list
            Array.from(termList.children).forEach(li => li.classList.remove('active'));
        } else {
            // Reset checklists specifically if not clearing term
            populateVariationsChecklist(termNameInput.value, currentTermData?.variations || []);
            populateParentsChecklist(currentTermData?.parents || []);
        }
        // Ensure level correctness default
        if (currentTermData) {
            populateLevelCorrectness(currentTermData.level);
        }
        console.log('Form cleared.');
    }

    async function handleFormSubmit(event) {
        event.preventDefault();
        
        const formData = new FormData(evaluationForm);
        const term = formData.get('term');
        
        // Combine all evaluation data
        const evaluationData = {
            academic_importance: formData.get('academic_importance'),
            academic_importance_notes: formData.get('academic_importance_notes'),
            level_correctness: formData.get('level_correctness'),
            level_correctness_notes: formData.get('level_correctness_notes'),
            variation_correctness: {},
            parent_relationships: {},
            potential_parents: {},
            potential_variations: {},
            related_concepts_notes: formData.get('related_concepts_notes'),
            timestamp: new Date().toISOString()
        };
        
        // Process variation correctness checkboxes
        document.querySelectorAll('input[name^="variation_correctness["]').forEach(input => {
            const variation = input.name.match(/\[(.*?)\]/)[1];
            evaluationData.variation_correctness[variation] = input.checked;
        });
        
        // Process parent relationship checkboxes
        document.querySelectorAll('input[name^="parent_relationships["]').forEach(input => {
            const parent = input.name.match(/\[(.*?)\]/)[1];
            evaluationData.parent_relationships[parent] = input.checked;
        });
        
        // Process potential parents checkboxes
        document.querySelectorAll('input[name^="potential_parents["]').forEach(input => {
            const concept = input.name.match(/\[(.*?)\]/)[1];
            evaluationData.potential_parents[concept] = input.checked;
        });
        
        // Process potential variations checkboxes
        document.querySelectorAll('input[name^="potential_variations["]').forEach(input => {
            const concept = input.name.match(/\[(.*?)\]/)[1];
            evaluationData.potential_variations[concept] = input.checked;
        });
        
        try {
            const response = await fetch('/api/save_evaluation', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    term: term,
                    evaluation: evaluationData
                })
            });
            
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            
            const result = await response.json();
            if (result.success) {
                // Update the evaluated terms set
                evaluatedTermsSet.add(term);
                
                // Update the evaluated count badge
                updateEvaluatedCountBadge();
                
                // Show success notification
                toast.show();
                
                // Update UI to show evaluation status
                updateEvaluationStatusIndicator(term, true);
            } else {
                console.error('Error saving evaluation:', result.message);
                alert(`Failed to save evaluation: ${result.message}`);
            }
        } catch (error) {
            console.error('Error saving evaluation:', error);
            alert('Failed to save evaluation due to an unexpected error. Please try again.');
        }
    }

    function handleGetRandomSample() {
        const selectedLevel = parseInt(sampleLevelSelect.value, 10);
        console.log(`Getting random sample for level ${selectedLevel}`);

        // Filter terms by selected level
        let termsAtLevel = Object.keys(allTermsData.terms).filter(term => {
            return allTermsData.terms[term] && allTermsData.terms[term].level !== undefined && parseInt(allTermsData.terms[term].level, 10) === selectedLevel;
        });
        
        // Apply evaluated-only filter if checked
        if (filterSettings.evaluatedOnly) {
            termsAtLevel = termsAtLevel.filter(term => evaluatedTermsSet.has(term));
            console.log(`Filtered to ${termsAtLevel.length} evaluated terms at level ${selectedLevel}`);
        }

        let sampleSize = 100;
        let sampledTerms;

        if (termsAtLevel.length <= sampleSize) {
            sampledTerms = termsAtLevel;
            console.log(`Level ${selectedLevel} has ${termsAtLevel.length} terms (<= 100), using all.`);
        } else {
            // Fisher-Yates (Knuth) Shuffle Algorithm for sampling
            const shuffled = termsAtLevel.slice(); // Clone the array
            for (let i = shuffled.length - 1; i > 0; i--) {
                const j = Math.floor(Math.random() * (i + 1));
                [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
            }
            sampledTerms = shuffled.slice(0, sampleSize);
            console.log(`Sampled ${sampleSize} terms from level ${selectedLevel}.`);
        }

        // Update filter level in UI to match the sampling level
        filterLevelSelect.value = selectedLevel.toString();
        filterSettings.level = selectedLevel.toString();
        
        // Also update the URL to reflect the current level
        updateUrlParams();

        currentlyDisplayedTerms = sampledTerms;
        searchInput.value = ''; // Clear search input
        filterSettings.searchQuery = '';
        searchResults.innerHTML = ''; // Clear search results display
        populateTermList(currentlyDisplayedTerms);
        clearEvaluationForm(true); // Clear form and selection
    }

    function handleFilterChange() {
        // Update filter settings
        filterSettings.level = filterLevelSelect.value;
        filterSettings.evaluatedOnly = showEvaluatedOnlyCheckbox.checked;
        
        // Apply filters to the full list of terms
        applyFilters();
        
        // Update URL params to reflect current filter state (without page reload)
        updateUrlParams();
    }
    
    function applyFilters() {
        // Start with all terms
        let filteredTerms = Object.keys(allTermsData.terms);
        
        // Filter by level if specified
        if (filterSettings.level !== 'all') {
            const levelNum = parseInt(filterSettings.level);
            filteredTerms = filteredTerms.filter(term => {
                const termData = allTermsData.terms[term];
                return termData && termData.level !== undefined && parseInt(termData.level, 10) === levelNum;
            });
        }
        
        // Filter by evaluation status if needed
        if (filterSettings.evaluatedOnly) {
            filteredTerms = filteredTerms.filter(term => evaluatedTermsSet.has(term));
        }
        
        // Filter by search query if one exists
        if (filterSettings.searchQuery && filterSettings.searchQuery.length >= 2) {
            const query = filterSettings.searchQuery.toLowerCase();
            filteredTerms = filteredTerms.filter(term => {
                return term.toLowerCase().includes(query);
            });
        }
        
        // Update the current display and populate the list
        currentlyDisplayedTerms = filteredTerms;
        populateTermList(currentlyDisplayedTerms);
        
        // Clear form if needed
        clearEvaluationForm(true);
    }
    
    function updateUrlParams() {
        // Create a new URLSearchParams object
        const newParams = new URLSearchParams();
        
        // Add current filter settings
        if (filterSettings.level !== 'all') {
            newParams.set('level', filterSettings.level);
        }
        
        if (filterSettings.evaluatedOnly) {
            newParams.set('evaluated', 'true');
        }
        
        if (termNameInput.value) {
            newParams.set('term', termNameInput.value);
        }
        
        // Replace current URL without reloading page
        const newUrl = window.location.pathname + (newParams.toString() ? '?' + newParams.toString() : '');
        window.history.replaceState({}, '', newUrl);
    }

    function updateEvaluatedCountBadge() {
        evaluatedCountBadge.textContent = `${evaluatedTermsSet.size} evaluated`;
    }

    // Add section highlighting CSS dynamically
    function addSectionHighlightStyle() {
        // Create a style element
        const style = document.createElement('style');
        // Add the CSS rules for section highlighting
        style.textContent = `
            @keyframes sectionHighlight {
                0% { background-color: rgba(13, 110, 253, 0.1); }
                100% { background-color: white; }
            }
            .section-highlight {
                animation: sectionHighlight 1s ease-out;
            }
        `;
        // Append the style element to the head
        document.head.appendChild(style);
    }
    
    // Call this once on page load
    addSectionHighlightStyle();
    
    // Function to initialize form navigation
    function initFormNavigation() {
        // Set up click handlers for the form navigation
        const navItems = formNav.querySelectorAll('.form-nav-item');
        
        navItems.forEach(item => {
            item.addEventListener('click', function(e) {
                e.preventDefault();
                
                // Remove active class from all items
                navItems.forEach(ni => ni.classList.remove('active'));
                
                // Add active class to the clicked item
                this.classList.add('active');
                
                // Scroll to the target section
                const targetId = this.getAttribute('data-section');
                const targetElement = document.getElementById(targetId);
                
                if (targetElement) {
                    // Scroll the element into view with smooth behavior
                    targetElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    
                    // Add a brief highlight effect to the section
                    targetElement.classList.add('section-highlight');
                    setTimeout(() => {
                        targetElement.classList.remove('section-highlight');
                    }, 1000);
                }
            });
        });
        
        // Set up scroll spy to update active nav item based on scroll position
        setupScrollSpy();
        
        // Set up form change event listeners for progress tracking
        setupFormChangeListeners();
    }
    
    function setupScrollSpy() {
        // Get all section elements
        const sections = [
            document.getElementById('section-1'),
            document.getElementById('section-2'),
            document.getElementById('section-3'),
            document.getElementById('section-4'),
            document.getElementById('section-5')
        ].filter(section => section); // Filter out any null sections
        
        // Listen for scroll events
        window.addEventListener('scroll', function() {
            if (!evaluationForm.style.display || evaluationForm.style.display === 'none') {
                return; // Don't do anything if form is not visible
            }
            
            // Get current scroll position
            const scrollPosition = window.scrollY + 100; // Add offset to account for the sticky nav
            
            // Find the section that is currently in view
            let currentSection = null;
            for (const section of sections) {
                if (!section) continue;
                
                const sectionTop = section.offsetTop;
                const sectionHeight = section.offsetHeight;
                
                if (scrollPosition >= sectionTop && scrollPosition < sectionTop + sectionHeight) {
                    currentSection = section.id;
                    break;
                }
            }
            
            // If no section is currently in view (e.g., scrolled past all sections)
            // use the last section
            if (!currentSection && sections.length > 0) {
                currentSection = sections[sections.length - 1].id;
            }
            
            // Update the active nav item
            if (currentSection) {
                const navItems = formNav.querySelectorAll('.form-nav-item');
                navItems.forEach(item => {
                    if (item.getAttribute('data-section') === currentSection) {
                        item.classList.add('active');
                    } else {
                        item.classList.remove('active');
                    }
                });
            }
        });
    }

    function setupFormChangeListeners() {
        // Listen for changes to radio buttons
        const radioButtons = evaluationForm.querySelectorAll('input[type="radio"]');
        radioButtons.forEach(radio => {
            radio.addEventListener('change', updateProgressBar);
        });
        
        // Listen for changes to checkboxes in variations and parents sections
        const checkboxGroups = [variationsChecklist, parentsChecklist];
        checkboxGroups.forEach(group => {
            group.addEventListener('change', function(e) {
                if (e.target.type === 'checkbox') {
                    updateProgressBar();
                }
            });
        });
        
        // Listen for changes in the related concepts section (added dynamically)
        const relatedConceptsSection = document.getElementById('related-concepts-section');
        if (relatedConceptsSection) {
            relatedConceptsSection.addEventListener('change', function(e) {
                if (e.target.type === 'checkbox') {
                    updateProgressBar();
                }
            });
        }
    }

    function updateProgressBar() {
        const progressBar = document.getElementById('eval-progress-bar');
        if (!progressBar) return;
        
        // Get all form controls that indicate completion
        const formElements = {
            importance: evaluationForm.querySelector('input[name="academic_importance"]:checked'),
            level: evaluationForm.querySelector('input[name="level_correctness"]:checked'),
            // For checkboxes, we just check if they exist
            variations: variationsChecklist.querySelectorAll('input[type="checkbox"]').length,
            parents: parentsChecklist.querySelectorAll('input[type="checkbox"]').length
        };
        
        // Count completed sections
        let completedSections = 0;
        let totalSections = 0;
        
        // Count importance section
        totalSections++;
        if (formElements.importance) completedSections++;
        
        // Count level section
        totalSections++;
        if (formElements.level) completedSections++;
        
        // Count variations section if there are variations
        if (formElements.variations > 0) {
            totalSections++;
            // Check if at least one checkbox is checked
            const checkedVariations = variationsChecklist.querySelectorAll('input[type="checkbox"]:checked').length;
            if (checkedVariations > 0) completedSections++;
        }
        
        // Count parents section if there are parents
        if (formElements.parents > 0) {
            totalSections++;
            // Check if at least one checkbox is checked
            const checkedParents = parentsChecklist.querySelectorAll('input[type="checkbox"]:checked').length;
            if (checkedParents > 0) completedSections++;
        }
        
        // Calculate percentage
        const percentage = Math.round((completedSections / totalSections) * 100);
        
        // Update progress bar
        progressBar.style.width = `${percentage}%`;
        progressBar.setAttribute('aria-valuenow', percentage);
        
        // Set color based on percentage
        if (percentage < 50) {
            progressBar.className = 'progress-bar bg-danger';
        } else if (percentage < 100) {
            progressBar.className = 'progress-bar bg-warning';
        } else {
            progressBar.className = 'progress-bar bg-success';
        }
    }
}); 