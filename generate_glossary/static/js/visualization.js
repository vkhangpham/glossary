// D3.js visualization for the hierarchy
class HierarchyVisualization {
    constructor(containerId) {
        this.containerId = containerId;
        this.svg = null;
        this.simulation = null;
        this.nodes = [];
        this.links = [];
        this.nodeElements = null;
        this.linkElements = null;
        this.textElements = null;
        this.data = null;
        this.selectedTerm = null;
        this.width = 0;
        this.height = 0;
        this.zoom = null;
        
        // New configuration options
        this.treeMode = false;
        this.maxParents = 5;
        this.maxSiblings = 5;
        this.maxChildren = 10;
        
        // Define event handlers
        this.onNodeClick = null;
    }
    
    init() {
        const container = document.getElementById(this.containerId);
        this.width = container.clientWidth;
        this.height = container.clientHeight;
        
        // Create SVG element
        this.svg = d3.select(`#${this.containerId}`)
            .append('svg')
            .attr('width', this.width)
            .attr('height', this.height);
            
        // Add zoom and pan capabilities
        this.zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on('zoom', (event) => {
                this.svg.select('g').attr('transform', event.transform);
            });
            
        this.svg.call(this.zoom);
        
        // Create a group for all elements
        this.svg.append('g');
    }
    
    loadData(data) {
        this.data = data;
        
        // Reset visualization
        this.svg.select('g').selectAll('*').remove();
        
        // Prepare nodes and links
        this.prepareGraphData();
        
        // Create the simulation
        this.createSimulation();
        
        // Create visual elements
        this.createElements();
        
        // Start the simulation
        this.startSimulation();
    }
    
    setTreeMode(enabled) {
        this.treeMode = enabled;
        if (this.data) {
            this.loadData(this.data);
        }
    }
    
    setMaxParents(value) {
        this.maxParents = value;
        if (this.data && this.selectedTerm) {
            this.loadData(this.data);
        }
    }
    
    setMaxSiblings(value) {
        this.maxSiblings = value;
        if (this.data && this.selectedTerm) {
            this.loadData(this.data);
        }
    }
    
    setMaxChildren(value) {
        this.maxChildren = value;
        if (this.data && this.selectedTerm) {
            this.loadData(this.data);
        }
    }
    
    // Find siblings of a term (terms with same parents)
    findSiblings(term) {
        const siblings = new Set();
        const termData = this.data.terms[term];
        
        if (!termData || !termData.parents.length) {
            return [];
        }
        
        // Look for other terms with the same parents
        for (const parent of termData.parents) {
            if (this.data.terms[parent]) {
                for (const child of this.data.terms[parent].children) {
                    if (child !== term) {
                        siblings.add(child);
                    }
                }
            }
        }
        
        return Array.from(siblings);
    }
    
    prepareGraphData() {
        // Reset nodes and links
        this.nodes = [];
        this.links = [];
        
        // Get all level filters
        const level0Filter = document.getElementById('level-0-filter').checked;
        const level1Filter = document.getElementById('level-1-filter').checked;
        const level2Filter = document.getElementById('level-2-filter').checked;
        const levelFilters = [level0Filter, level1Filter, level2Filter];
        
        // Start with the selected term and its immediate neighbors
        const term = this.selectedTerm;
        if (!term || !this.data.terms[term]) {
            // If no term is selected, show a sample of terms from each level
            for (let level = 0; level < 3; level++) {
                if (levelFilters[level]) {
                    const levelTerms = this.data.levels[level];
                    const sampleSize = Math.min(10, levelTerms.length);
                    const sampleTerms = levelTerms.slice(0, sampleSize);
                    
                    // Add sample terms to nodes
                    for (const term of sampleTerms) {
                        this.nodes.push({
                            id: term,
                            level: level,
                            parents: this.data.terms[term].parents,
                            children: this.data.terms[term].children
                        });
                    }
                }
            }
        } else {
            // Add the selected term
            const termData = this.data.terms[term];
            const level = termData.level;
            
            if (levelFilters[level]) {
                this.nodes.push({
                    id: term,
                    level: level,
                    parents: termData.parents,
                    children: termData.children,
                    selected: true
                });
                
                // Add parent terms (limited by maxParents)
                const parentsToAdd = termData.parents.slice(0, this.maxParents);
                for (const parent of parentsToAdd) {
                    if (this.data.terms[parent]) {
                        const parentLevel = this.data.terms[parent].level;
                        if (levelFilters[parentLevel]) {
                            this.nodes.push({
                                id: parent,
                                level: parentLevel,
                                parents: this.data.terms[parent].parents,
                                children: this.data.terms[parent].children
                            });
                            
                            // Add link from parent to term
                            this.links.push({
                                source: parent,
                                target: term
                            });
                        }
                    }
                }
                
                // Add sibling terms (always, regardless of tree mode)
                const siblings = this.findSiblings(term);
                const siblingsToAdd = siblings.slice(0, this.maxSiblings);
                
                for (const sibling of siblingsToAdd) {
                    if (this.data.terms[sibling]) {
                        const siblingLevel = this.data.terms[sibling].level;
                        if (levelFilters[siblingLevel]) {
                            this.nodes.push({
                                id: sibling,
                                level: siblingLevel,
                                parents: this.data.terms[sibling].parents,
                                children: this.data.terms[sibling].children,
                                isSibling: true
                            });
                            
                            // Add links from parents to siblings
                            const commonParents = this.data.terms[sibling].parents.filter(
                                parent => termData.parents.includes(parent)
                            );
                            
                            for (const parent of commonParents) {
                                if (this.nodes.some(node => node.id === parent)) {
                                    this.links.push({
                                        source: parent,
                                        target: sibling
                                    });
                                }
                            }
                        }
                    }
                }
                
                // Add child terms (limited by maxChildren)
                const childrenToAdd = termData.children.slice(0, this.maxChildren);
                for (const child of childrenToAdd) {
                    if (this.data.terms[child]) {
                        const childLevel = this.data.terms[child].level;
                        if (levelFilters[childLevel]) {
                            this.nodes.push({
                                id: child,
                                level: childLevel,
                                parents: this.data.terms[child].parents,
                                children: this.data.terms[child].children
                            });
                            
                            // Add link from term to child
                            this.links.push({
                                source: term,
                                target: child
                            });
                        }
                    }
                }
            }
        }
        
        // Add links between the displayed nodes
        const nodeIds = new Set(this.nodes.map(node => node.id));
        for (const relationship of this.data.relationships.parent_child) {
            const parent = relationship[0];
            const child = relationship[1];
            
            if (nodeIds.has(parent) && nodeIds.has(child)) {
                // Check if this link already exists
                const linkExists = this.links.some(link => 
                    link.source === parent && link.target === child);
                    
                if (!linkExists) {
                    this.links.push({
                        source: parent,
                        target: child
                    });
                }
            }
        }
    }
    
    createSimulation() {
        // Check if tree mode is enabled and a term is selected
        if (this.treeMode && this.selectedTerm) {
            this.applyTreeLayout();
        } else {
            // For normal layout, use force simulation
            this.simulation = d3.forceSimulation(this.nodes)
                .force('link', d3.forceLink(this.links)
                    .id(d => d.id)
                    .distance(100))
                .force('charge', d3.forceManyBody().strength(-200))
                .force('center', d3.forceCenter(this.width / 2, this.height / 2))
                .force('collision', d3.forceCollide().radius(30));
        }
    }
    
    applyTreeLayout() {
        // Cancel any existing simulation
        if (this.simulation) {
            this.simulation.stop();
        }
        
        // Categorize nodes
        const parents = [];
        const children = [];
        const siblings = [];
        let selectedNode = null;
        
        this.nodes.forEach(node => {
            // Reset fixed positions
            node.fx = null;
            node.fy = null;
            
            if (node.selected) {
                selectedNode = node;
            } else if (node.isSibling) {
                siblings.push(node);
            } else if (this.links.some(link => 
                (typeof link.source === 'object' ? link.source.id : link.source) === node.id &&
                (typeof link.target === 'object' ? link.target.id : link.target) === this.selectedTerm)) {
                parents.push(node);
            } else if (this.links.some(link => 
                (typeof link.source === 'object' ? link.source.id : link.source) === this.selectedTerm &&
                (typeof link.target === 'object' ? link.target.id : link.target) === node.id)) {
                children.push(node);
            }
        });
        
        // Calculate vertical spacing - make it dramatic
        const levelHeight = this.height / 5;
        
        // Position parents at top level
        const parentWidth = this.width / (parents.length || 1);
        parents.forEach((node, i) => {
            node.fx = parentWidth * (i + 0.5);
            node.fy = levelHeight;
        });
        
        // Handle middle level: selected term and siblings
        // We'll treat the selected node as just another middle-level node for positioning
        const middleLevelNodes = [...siblings];
        if (selectedNode) {
            middleLevelNodes.push(selectedNode);
        }
        
        // Sort nodes to ensure consistent ordering
        middleLevelNodes.sort((a, b) => a.id.localeCompare(b.id));
        
        // Position all middle level nodes (including selected)
        if (middleLevelNodes.length > 0) {
            const totalWidth = this.width * 0.8; // Use 80% of width to avoid edges
            const spacing = totalWidth / (middleLevelNodes.length);
            
            // Calculate starting position (to center the nodes)
            const startX = (this.width - totalWidth) / 2 + spacing/2;
            
            middleLevelNodes.forEach((node, i) => {
                // Position middle level nodes evenly spaced
                node.fx = startX + (i * spacing);
                node.fy = levelHeight * 2.5;
            });
        }
        
        // Position children at bottom level
        const childWidth = this.width / (children.length || 1);
        children.forEach((node, i) => {
            node.fx = childWidth * (i + 0.5);
            node.fy = levelHeight * 4;
        });
        
        // In tree mode, we need to handle links differently - highlight links to/from selected node
        // Filter the links to only keep those connected to the selected node
        this.links.forEach(link => {
            const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
            const targetId = typeof link.target === 'object' ? link.target.id : link.target;
            
            // Mark links connected to the selected node
            if (sourceId === this.selectedTerm || targetId === this.selectedTerm) {
                link.isSelectedLink = true;
            } else {
                link.isSelectedLink = false;
            }
        });
        
        // Create a minimal simulation just for rendering
        this.simulation = d3.forceSimulation(this.nodes)
            .alphaTarget(0)
            .alphaDecay(0);
    }
    
    createElements() {
        // Create link elements
        this.linkElements = this.svg.select('g').selectAll('.link')
            .data(this.links)
            .enter()
            .append('line')
            .attr('class', d => this.treeMode ? 
                  (d.isSelectedLink ? 'link selected-link' : 'link hidden-link') : 
                  'link')
            .attr('stroke-width', d => this.treeMode && d.isSelectedLink ? 2 : 1)
            .attr('stroke', d => this.treeMode && d.isSelectedLink ? '#333' : '#999')
            .attr('stroke-opacity', d => this.treeMode && !d.isSelectedLink ? 0.1 : 0.6);
            
        // Create node elements with different styles based on node type
        this.nodeElements = this.svg.select('g').selectAll('.node')
            .data(this.nodes)
            .enter()
            .append('circle')
            .attr('class', d => `term-node node-level-${d.level}`)
            .attr('r', d => {
                if (d.selected) return 18;
                if (d.isSibling) return 8;
                return 10;
            })
            .attr('id', d => `node-${d.id}`)
            .attr('stroke', d => d.isSibling ? '#888' : (d.selected ? '#000' : 'none'))
            .attr('stroke-width', d => d.isSibling ? 1 : (d.selected ? 3 : 0))
            .attr('stroke-dasharray', d => d.selected ? '4,2' : 'none')
            .on('click', (event, d) => {
                if (this.onNodeClick) {
                    this.onNodeClick(d.id);
                }
            });
            
        // Create text elements
        this.textElements = this.svg.select('g').selectAll('.text')
            .data(this.nodes)
            .enter()
            .append('text')
            .text(d => d.id)
            .attr('font-size', d => d.selected ? 14 : 10)
            .attr('font-weight', d => d.selected ? 'bold' : 'normal')
            .attr('dx', d => d.selected ? 22 : 15)
            .attr('dy', 4);
            
        // Add tooltips to nodes
        this.nodeElements.append('title')
            .text(d => {
                let tooltip = d.id;
                if (d.level !== undefined) {
                    tooltip += ` (Level ${d.level})`;
                }
                return tooltip;
            });
    }
    
    startSimulation() {
        // Update positions on each tick
        this.simulation.on('tick', () => {
            this.linkElements
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);
                
            this.nodeElements
                .attr('cx', d => d.x)
                .attr('cy', d => d.y);
                
            this.textElements
                .attr('x', d => d.x)
                .attr('y', d => d.y);
        });
    }
    
    updateSelectedTerm(term) {
        this.selectedTerm = term;
        this.loadData(this.data);
    }
    
    setNodeClickHandler(handler) {
        this.onNodeClick = handler;
    }
    
    toggleFullscreen() {
        const container = document.querySelector('.visualization-container');
        container.classList.toggle('fullscreen');
        
        // Adjust visualization size
        if (container.classList.contains('fullscreen')) {
            this.width = window.innerWidth;
            this.height = window.innerHeight;
        } else {
            this.width = container.clientWidth;
            this.height = container.clientHeight;
        }
        
        this.svg
            .attr('width', this.width)
            .attr('height', this.height);
            
        // Recenter the visualization
        if (this.simulation) {
            this.simulation.force('center', d3.forceCenter(this.width / 2, this.height / 2));
            this.simulation.alpha(0.3).restart();
        }
    }
}