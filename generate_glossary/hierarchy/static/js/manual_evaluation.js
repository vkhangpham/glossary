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
    const getRandomSampleBtn = document.getElementById('get-random-sample-btn');
    const showAllTermsBtn = document.getElementById('show-all-terms-btn');
    const toast = new bootstrap.Toast(saveNotification);

    let allTermsData = {}; // Store all hierarchy data
    let currentTermData = null; // Store data for the currently selected term
    let currentlyDisplayedTerms = []; // Keep track of terms currently in the list
    let evaluatedTermsSet = new Set(); // Keep track of terms with saved evaluations

    // --- Initialization ---
    fetchHierarchyData();
    fetchEvaluatedTerms(); // Fetch evaluated terms list

    // --- Event Listeners ---
    searchInput.addEventListener('input', handleSearch);
    evaluationForm.addEventListener('submit', handleFormSubmit);
    clearFormBtn.addEventListener('click', clearEvaluationForm);
    getRandomSampleBtn.addEventListener('click', handleGetRandomSample);
    showAllTermsBtn.addEventListener('click', handleShowAllTerms);

    // --- Functions ---
    async function fetchHierarchyData() {
        try {
            const response = await fetch('/api/hierarchy');
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            allTermsData = await response.json();
            currentlyDisplayedTerms = Object.keys(allTermsData.terms);
            populateTermList(currentlyDisplayedTerms); // Initially populate with all terms
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
            evaluatedTermsSet = new Set(data.evaluated_terms || []);
            console.log(`Fetched ${evaluatedTermsSet.size} evaluated terms.`);
            // Re-populate list if data is already loaded to apply styling
            if (Object.keys(allTermsData).length > 0) {
                populateTermList(currentlyDisplayedTerms);
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
        if (!query) {
            searchResults.innerHTML = '';
            populateTermList(currentlyDisplayedTerms);
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

        const resultTerms = new Set(); // Use a set to avoid duplicates from search results
        const termsToSearch = currentlyDisplayedTerms; // Search within the current view (all or sample)

        results.filter(result => termsToSearch.includes(result.term)).forEach(result => {
            const term = result.term;
            if (term && allTermsData.terms[term]) { // Ensure term exists in our main data
                resultTerms.add(term);
            }
        });

        populateTermList(Array.from(resultTerms)); // Update term list based on search results

         // Also display search results directly for quick selection (optional)
        // results.slice(0, 10).forEach(result => {
        //     const div = document.createElement('div');
        //     div.className = 'search-result-item';
        //     div.textContent = result.term + (result.type === 'variation' ? ` (variation: ${result.variation})` : '');
        //     div.addEventListener('click', () => selectTerm(result.term));
        //     searchResults.appendChild(div);
        // });
    }


    async function selectTerm(term) {
        console.log('Selecting term:', term);
        selectedTermHeader.textContent = `Evaluating: ${term}`;
        termNameInput.value = term;
        currentTermData = allTermsData.terms[term];

        // Update active item in the list
        Array.from(termList.children).forEach(li => {
            li.classList.toggle('active', li.dataset.term === term);
        });

        if (!currentTermData) {
            console.error('Term data not found for:', term);
            clearEvaluationForm();
            evaluationForm.style.display = 'none';
            termDetailsDisplay.style.display = 'none';
            return;
        }

        displayTermDetails(currentTermData);
        termDetailsDisplay.style.display = 'block';

        // Populate form sections
        populateLevelCorrectness(currentTermData.level);
        populateVariationsChecklist(term); // Pass term name for fetching variations
        populateParentsChecklist(currentTermData.parents || []);

        // Set default academic importance *before* loading saved evaluation
        setDefaultAcademicImportance();

        // Fetch and apply existing evaluation
        await loadAndApplyEvaluation(term);

        evaluationForm.style.display = 'block';
    }

    function displayTermDetails(termData) {
        termLevelDisplay.textContent = termData.level !== undefined ? termData.level : 'N/A';
        termParentsList.textContent = (termData.parents && termData.parents.length > 0) ? termData.parents.join(', ') : 'None';

        // Find variations for the term
        const variations = allTermsData.relationships?.variations
            ?.filter(v => v[0] === termNameInput.value)
            .map(v => v[1]) || [];
        termVariationsList.textContent = variations.length > 0 ? variations.join(', ') : 'None';
    }

    async function loadAndApplyEvaluation(term) {
        try {
            const response = await fetch(`/api/load_evaluation/${encodeURIComponent(term)}`);
            const result = await response.json();
            if (result.success && result.evaluation) {
                console.log('Applying existing evaluation:', result.evaluation);
                applyEvaluationData(result.evaluation);
            } else {
                console.log('No existing evaluation found, clearing form and setting default importance.');
                clearEvaluationForm(false); // Clear form but keep term name
                setDefaultAcademicImportance(); // Set default *after* clearing form when no evaluation exists
            }
        } catch (error) {
            console.error('Error loading evaluation:', error);
            clearEvaluationForm(false);
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

    function populateLevelCorrectness(currentLevel) {
        currentLevelDisplay.textContent = currentLevel !== undefined ? currentLevel : 'N/A';
        // Uncheck all level radios initially
        evaluationForm.querySelectorAll('input[name="level_correctness"]').forEach(radio => radio.checked = false);

        // Show/hide specific level options based on current level
         for (let i = 0; i <= 3; i++) {
             const radioInput = document.getElementById(`level-${i}`);
             const radioContainer = radioInput ? radioInput.closest('.form-check') : null;
             if (radioContainer) {
                 if (i === currentLevel) {
                     radioContainer.style.display = 'none'; // Hide the option for the current level
                 } else {
                     radioContainer.style.display = 'inline-block'; // Ensure others are visible (use inline-block for inline radios)
                 }
             }
         }

        // Check 'correct' by default if level is valid
        if (currentLevel !== undefined && currentLevel !== null) {
            const correctRadio = document.getElementById('level-correct');
            if (correctRadio) correctRadio.checked = true;
        }
    }

    function populateVariationsChecklist(term) {
        variationsChecklist.innerHTML = ''; // Clear previous
        const variations = allTermsData.relationships?.variations
            ?.filter(v => v[0] === term)
            .map(v => v[1]) || [];

        if (variations.length === 0) {
            variationsChecklist.innerHTML = '<p class="text-muted">No variations listed for this term.</p>';
            return;
        }

        variations.forEach((variation, index) => {
            const id = `variation-check-${index}`;
            const div = document.createElement('div');
            div.className = 'form-check variation-item';
            div.innerHTML = `
                <input class="form-check-input" type="checkbox" id="${id}" name="variation_${variation}" value="true" checked>
                <label class="form-check-label" for="${id}">${variation}</label>
            `;
            variationsChecklist.appendChild(div);
        });
    }

    function populateParentsChecklist(parents) {
        parentsChecklist.innerHTML = ''; // Clear previous
        if (!parents || parents.length === 0) {
            parentsChecklist.innerHTML = '<p class="text-muted">No parents listed for this term.</p>';
            return;
        }

        parents.forEach((parent, index) => {
            const id = `parent-check-${index}`;
            const div = document.createElement('div');
            div.className = 'form-check parent-item';
            div.innerHTML = `
                <input class="form-check-input" type="checkbox" id="${id}" name="parent_${parent}" value="true" checked>
                <label class="form-check-label" for="${id}">${parent}</label>
            `;
            parentsChecklist.appendChild(div);
        });
    }

    function clearEvaluationForm(clearTerm = true) {
        evaluationForm.reset(); // Resets form elements to default values
        if (clearTerm) {
            termNameInput.value = '';
            selectedTermHeader.textContent = 'Select a term from the list';
            evaluationForm.style.display = 'none';
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
            populateVariationsChecklist(termNameInput.value); // Re-populate with defaults
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
        const term = termNameInput.value;
        if (!term) {
            alert('Please select a term before saving.');
            return;
        }

        const formData = new FormData(evaluationForm);
        const evaluation = {
            academic_importance: formData.get('academic_importance'),
            academic_importance_notes: formData.get('academic_importance_notes'),
            level_correctness: formData.get('level_correctness'),
            level_correctness_notes: formData.get('level_correctness_notes'),
            variation_correctness: {},
            variation_correctness_notes: formData.get('variation_correctness_notes'),
            parent_relationships: {},
            parent_relationship_notes: formData.get('parent_relationship_notes'),
            timestamp: new Date().toISOString()
        };

        // Collect variation checkboxes
        variationsChecklist.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
            const variationName = checkbox.name.replace('variation_', '');
            evaluation.variation_correctness[variationName] = checkbox.checked;
        });

        // Collect parent checkboxes
        parentsChecklist.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
            const parentName = checkbox.name.replace('parent_', '');
            evaluation.parent_relationships[parentName] = checkbox.checked;
        });

        console.log('Submitting evaluation:', evaluation);

        try {
            const response = await fetch('/api/save_evaluation', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ term: term, evaluation: evaluation }),
            });

            const result = await response.json();

            if (response.ok && result.success) {
                console.log('Evaluation saved successfully.');
                toast.show(); // Show success notification

                // Add term to evaluated set and update its styling in the list
                evaluatedTermsSet.add(term);
                const listItem = termList.querySelector(`li[data-term="${term}"]`);
                if (listItem) {
                    // Add temporary highlight instead of permanent checkmark
                    listItem.classList.add('just-evaluated');
                    setTimeout(() => {
                        listItem.classList.remove('just-evaluated');
                    }, 3000); // Remove highlight after 3 seconds

                    // Ensure the permanent evaluated class and icon are added *if not already there* 
                    // This covers the case where the user re-evaluates a term
                    if (!listItem.classList.contains('evaluated')) {
                        listItem.classList.add('evaluated'); 
                        // Add the icon only if it truly wasn't there before (unlikely if saving succeeds, but safe)
                        if (!listItem.querySelector('.evaluated-icon')) {
                            const checkIcon = document.createElement('i');
                            checkIcon.className = 'bi bi-check-circle-fill evaluated-icon text-success';
                            checkIcon.title = 'Evaluated';
                            const termTextSpan = listItem.querySelector('span');
                            if(termTextSpan) {
                                listItem.insertBefore(checkIcon, termTextSpan.nextSibling);
                            } else {
                                listItem.appendChild(checkIcon); 
                            }
                        }
                    }
                }
            } else {
                throw new Error(result.message || 'Failed to save evaluation');
            }
        } catch (error) {
            console.error('Error saving evaluation:', error);
            alert(`Error saving evaluation: ${error.message}`);
        }
    }

    function handleGetRandomSample() {
        const selectedLevel = parseInt(sampleLevelSelect.value, 10);
        console.log(`Getting random sample for level ${selectedLevel}`);

        const termsAtLevel = Object.keys(allTermsData.terms).filter(term => {
            return allTermsData.terms[term]?.level === selectedLevel;
        });

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

        currentlyDisplayedTerms = sampledTerms;
        searchInput.value = ''; // Clear search input
        searchResults.innerHTML = ''; // Clear search results display
        populateTermList(currentlyDisplayedTerms);
        clearEvaluationForm(true); // Clear form and selection
    }

    function handleShowAllTerms() {
        console.log('Showing all terms.');
        currentlyDisplayedTerms = Object.keys(allTermsData.terms);
        searchInput.value = ''; // Clear search input
        searchResults.innerHTML = ''; // Clear search results display
        populateTermList(currentlyDisplayedTerms);
        clearEvaluationForm(true); // Clear form and selection
    }
}); 