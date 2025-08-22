// MathJax configuration for AI Services documentation
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

// Add copy to clipboard functionality for API examples
document$.subscribe(() => {
  // Add copy buttons to code blocks
  const codeBlocks = document.querySelectorAll('pre code');
  codeBlocks.forEach((block, index) => {
    if (block.classList.contains('language-bash') || 
        block.classList.contains('language-python') ||
        block.classList.contains('language-json')) {
      
      const button = document.createElement('button');
      button.className = 'copy-button';
      button.innerHTML = '📋 Copy';
      button.title = 'Copy to clipboard';
      
      button.addEventListener('click', () => {
        navigator.clipboard.writeText(block.textContent).then(() => {
          button.innerHTML = '✅ Copied!';
          setTimeout(() => {
            button.innerHTML = '📋 Copy';
          }, 2000);
        });
      });
      
      const pre = block.parentNode;
      pre.style.position = 'relative';
      pre.appendChild(button);
    }
  });

  // Add service status indicators
  const healthButtons = document.querySelectorAll('.health-check');
  healthButtons.forEach(button => {
    button.addEventListener('click', async (e) => {
      e.preventDefault();
      const service = button.dataset.service;
      const url = button.dataset.url;
      
      try {
        const response = await fetch(url);
        const data = await response.json();
        
        if (data.status === 'healthy') {
          button.className = 'status-badge status-success';
          button.textContent = `${service} ✅ Healthy`;
        } else {
          button.className = 'status-badge status-error';
          button.textContent = `${service} ❌ Unhealthy`;
        }
      } catch (error) {
        button.className = 'status-badge status-error';
        button.textContent = `${service} ❌ Offline`;
      }
    });
  });

  // API response prettifier
  const apiExamples = document.querySelectorAll('.api-example');
  apiExamples.forEach(example => {
    const button = document.createElement('button');
    button.textContent = 'Try API';
    button.className = 'button-primary';
    
    button.addEventListener('click', async () => {
      const method = example.dataset.method || 'GET';
      const url = example.dataset.url;
      const body = example.dataset.body;
      
      try {
        const options = {
          method,
          headers: { 'Content-Type': 'application/json' }
        };
        
        if (body) {
          options.body = body;
        }
        
        const response = await fetch(url, options);
        const data = await response.json();
        
        const resultDiv = document.createElement('div');
        resultDiv.className = 'api-result';
        resultDiv.innerHTML = `
          <h4>Response:</h4>
          <pre><code class="language-json">${JSON.stringify(data, null, 2)}</code></pre>
        `;
        
        // Remove existing result
        const existing = example.querySelector('.api-result');
        if (existing) existing.remove();
        
        example.appendChild(resultDiv);
      } catch (error) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'api-error';
        errorDiv.innerHTML = `<p>Error: ${error.message}</p>`;
        example.appendChild(errorDiv);
      }
    });
    
    example.appendChild(button);
  });

  // Smooth scrolling for anchor links
  const anchorLinks = document.querySelectorAll('a[href^="#"]');
  anchorLinks.forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const target = document.querySelector(link.getAttribute('href'));
      if (target) {
        target.scrollIntoView({ behavior: 'smooth' });
      }
    });
  });

  // Add fade-in animation to cards
  const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
  };

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('fade-in');
        observer.unobserve(entry.target);
      }
    });
  }, observerOptions);

  const cards = document.querySelectorAll('.service-card, .grid .card, .metric-card');
  cards.forEach(card => observer.observe(card));

  // Dark mode toggle enhancement
  const toggleButton = document.querySelector('[data-md-toggle="search"]');
  if (toggleButton) {
    toggleButton.addEventListener('click', () => {
      // Add visual feedback for mode switches
      document.body.style.transition = 'background-color 0.3s ease';
    });
  }
});

// Service monitoring dashboard (if enabled)
if (window.location.pathname.includes('/monitoring') || 
    window.location.pathname.includes('/operations')) {
  
  // Real-time metrics updates
  async function updateMetrics() {
    try {
      const ragflowStats = await fetch('http://localhost:8010/stats').then(r => r.json());
      const deepwikiStats = await fetch('http://localhost:8011/metrics').then(r => r.json());
      
      // Update metric displays
      const ragflowMetrics = document.getElementById('ragflow-metrics');
      const deepwikiMetrics = document.getElementById('deepwiki-metrics');
      
      if (ragflowMetrics) {
        ragflowMetrics.innerHTML = `
          <div class="metric-card">
            <span class="metric-value">${ragflowStats.total_documents || 0}</span>
            <span class="metric-label">Documents Indexed</span>
          </div>
          <div class="metric-card">
            <span class="metric-value">${ragflowStats.total_queries || 0}</span>
            <span class="metric-label">Queries Processed</span>
          </div>
        `;
      }
      
      if (deepwikiMetrics) {
        deepwikiMetrics.innerHTML = `
          <div class="metric-card">
            <span class="metric-value">${deepwikiStats.articles_total || 0}</span>
            <span class="metric-label">Wiki Articles</span>
          </div>
          <div class="metric-card">
            <span class="metric-value">${deepwikiStats.links_total || 0}</span>
            <span class="metric-label">Knowledge Links</span>
          </div>
        `;
      }
    } catch (error) {
      console.log('Metrics update failed:', error);
    }
  }
  
  // Update metrics every 30 seconds
  setInterval(updateMetrics, 30000);
  updateMetrics(); // Initial load
}

// Add keyboard shortcuts for common actions
document.addEventListener('keydown', (e) => {
  // Ctrl/Cmd + K for search
  if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
    e.preventDefault();
    const searchButton = document.querySelector('[data-md-toggle="search"]');
    if (searchButton) searchButton.click();
  }
  
  // ESC to close search
  if (e.key === 'Escape') {
    const searchToggle = document.querySelector('#__search');
    if (searchToggle && searchToggle.checked) {
      searchToggle.checked = false;
    }
  }
});

// Performance monitoring
if ('performance' in window && 'PerformanceObserver' in window) {
  const perfObserver = new PerformanceObserver((list) => {
    list.getEntries().forEach((entry) => {
      if (entry.entryType === 'navigation') {
        console.log('Page load time:', entry.loadEventEnd - entry.loadEventStart, 'ms');
      }
    });
  });
  
  perfObserver.observe({ entryTypes: ['navigation'] });
}

// Add custom CSS for copy buttons
const style = document.createElement('style');
style.textContent = `
  .copy-button {
    position: absolute;
    top: 8px;
    right: 8px;
    background: rgba(33, 150, 243, 0.8);
    color: white;
    border: none;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 12px;
    cursor: pointer;
    transition: all 0.2s ease;
    z-index: 10;
  }
  
  .copy-button:hover {
    background: rgba(33, 150, 243, 1);
    transform: scale(1.05);
  }
  
  .api-result {
    margin-top: 1rem;
    padding: 1rem;
    background: #f8f9fa;
    border-radius: 0.5rem;
    border-left: 4px solid #4caf50;
  }
  
  .api-error {
    margin-top: 1rem;
    padding: 1rem;
    background: #ffebee;
    border-radius: 0.5rem;
    border-left: 4px solid #f44336;
    color: #c62828;
  }
  
  pre {
    position: relative;
  }
  
  .fade-in {
    animation: fadeIn 0.6s ease-in forwards;
  }
  
  @keyframes fadeIn {
    from { 
      opacity: 0; 
      transform: translateY(20px); 
    }
    to { 
      opacity: 1; 
      transform: translateY(0); 
    }
  }
`;

document.head.appendChild(style);