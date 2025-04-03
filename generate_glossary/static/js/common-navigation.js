document.addEventListener('DOMContentLoaded', function() {
    // Get the current page path
    const currentPath = window.location.pathname;
    
    // Update active state based on current page
    function updateActiveNav() {
        const navLinks = document.querySelectorAll('#main-nav .nav-link');
        navLinks.forEach(link => {
            const linkPath = link.getAttribute('data-path');
            if (currentPath === linkPath) {
                link.classList.add('active');
            } else {
                link.classList.remove('active');
            }
        });
    }
    
    // Initialize navigation
    updateActiveNav();
}); 