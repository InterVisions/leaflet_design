// State
const state = {
    pages: [],
    currentPageIndex: 0,
    totalPages: 0
};

const elements = {
    promptInput: document.getElementById('prompt'),
    numPagesInput: document.getElementById('numPages'),
    generateBtn: document.getElementById('generateBtn'),
    flipbookView: document.getElementById('flipbookView'),
    prevPage: document.getElementById('prevPage'),
    nextPage: document.getElementById('nextPage'),
    currentPage: document.getElementById('currentPage'),
    totalPages: document.getElementById('totalPages')
};

// Image placeholder generator (Portrait ratio for A4)
function generatePlaceholder(slotId, prompt, pageNum) {
    const canvas = document.createElement('canvas');
    canvas.width = 600; canvas.height = 800; // Taller images for A4 fitting
    const ctx = canvas.getContext('2d');
    const hue = 140; // Green base
    const variation = (parseInt(slotId.split('-')[1]) * 20) + (pageNum * 15);
    
    const gradient = ctx.createLinearGradient(0, 0, 600, 800);
    gradient.addColorStop(0, `hsl(${hue + variation}, 40%, 65%)`);
    gradient.addColorStop(1, `hsl(${hue + variation}, 40%, 35%)`);
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, 600, 800);
    
    // Slight texture pattern
    ctx.fillStyle = 'rgba(255, 255, 255, 0.1)';
    for (let i = 0; i < 20; i++) {
        ctx.beginPath();
        ctx.arc(Math.random() * 600, Math.random() * 800, Math.random() * 60 + 20, 0, Math.PI * 2);
        ctx.fill();
    }
    
    ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
    ctx.font = 'bold 36px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(`Image ${slotId.split('-')[1]}`, 300, 400);
    return canvas.toDataURL('image/jpeg');
}

// Layout cycle config
const PAGE_LAYOUTS = [
    { name: 'cover' },
    { name: 'feature' },
    { name: 'story' },
    { name: 'gallery' },
    { name: 'closing' }
];

function createLeafletHTML(images, pageIndex, prompt) {
    const layout = PAGE_LAYOUTS[pageIndex % PAGE_LAYOUTS.length].name;
    
    if (layout === 'cover') {
        return `
            <div class="leaflet layout-cover">
                <div class="cover-image"><img src="${images['img-1']}"></div>
                <div class="cover-overlay">
                    <span class="cover-tag">Project Overview</span>
                    <h1 class="cover-title">Design<br>Vision</h1>
                    <p class="cover-sub">Exploring concepts for: ${prompt.substring(0, 30)}...</p>
                </div>
            </div>`;
    }
    
    if (layout === 'feature') {
        return `
            <div class="leaflet layout-feature">
                <div class="feature-main">
                    <div class="feature-img-wrap"><img src="${images['img-1']}"></div>
                    <div class="feature-caption">
                        <h2>Core Concept</h2>
                        <p>Focusing on sustainability and modern aesthetics.</p>
                    </div>
                </div>
                <div class="feature-side">
                    <div class="feature-side-img"><img src="${images['img-2']}"></div>
                    <div class="feature-side-text">
                        <span class="feature-badge">Detail View</span>
                        <h3>Material Study</h3>
                        <p>Natural textures integration.</p>
                    </div>
                </div>
            </div>`;
    }
    
    if (layout === 'story') {
        return `
            <div class="leaflet layout-story">
                <div class="story-photo">
                    <img src="${images['img-1']}">
                    <span class="story-photo-label">Environmental Context</span>
                </div>
                <div class="story-body">
                    <h2 class="story-heading">Harmonious Integration</h2>
                    <div class="story-columns">
                        <p>The design seamlessly blends the built environment with surrounding natural elements, creating a cohesive visual language.</p>
                        <p>Light and shadow play a crucial role in defining the spatial experience throughout the day.</p>
                    </div>
                </div>
            </div>`;
    }

    if (layout === 'gallery') {
        return `
            <div class="leaflet layout-gallery">
                <div class="gallery-header">
                    <h2>Visual Explorations</h2>
                    <p>Alternative perspectives and interior studies.</p>
                </div>
                <div class="gallery-grid">
                    <div class="gallery-item gallery-item-large">
                        <img src="${images['img-1']}">
                        <span class="gallery-label">Primary Angle</span>
                    </div>
                    <div class="gallery-item"><img src="${images['img-2']}"></div>
                    <div class="gallery-item"><img src="${images['img-3']}"></div>
                </div>
            </div>`;
    }

    if (layout === 'closing') {
        return `
            <div class="leaflet layout-closing">
                <div class="closing-stats">
                    <div class="closing-stats-img"><img src="${images['img-1']}"></div>
                    <div class="closing-numbers">
                        <div class="stat-block"><span class="stat-n">100%</span><span class="stat-l">Renewable</span></div>
                        <div class="stat-block"><span class="stat-n">45%</span><span class="stat-l">Green Space</span></div>
                    </div>
                </div>
                <div class="closing-cta">
                    <div class="closing-cta-img"><img src="${images['img-2']}"></div>
                    <div class="closing-cta-text">
                        <h2>Next Steps</h2>
                        <p>Ready to move this concept into development.</p>
                    </div>
                </div>
            </div>`;
    }
}

async function generateFlipbook() {
    const prompt = elements.promptInput.value.trim();
    const numPages = parseInt(elements.numPagesInput.value) || 5;
    
    elements.generateBtn.disabled = true;
    elements.generateBtn.textContent = 'Generating...';
    
    // Fake loading delay to mimic API
    await new Promise(resolve => setTimeout(resolve, 800));
    
    const pages = [];
    for (let i = 0; i < numPages; i++) {
        const page = { id: i, images: {} };
        ['img-1', 'img-2', 'img-3'].forEach(slotId => {
            page.images[slotId] = generatePlaceholder(slotId, prompt, i);
        });
        pages.push(page);
    }
    
    state.pages = pages;
    state.totalPages = pages.length;
    state.currentPageIndex = 0;
    
    buildFlipbook('flipbook', pages, prompt);
    
    elements.flipbookView.classList.remove('hidden');
    updatePageNavigation();
    
    elements.generateBtn.disabled = false;
    elements.generateBtn.textContent = 'Generate Flipbook';
    elements.flipbookView.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function buildFlipbook(containerId, pages, prompt) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    
    pages.forEach((page, index) => {
        const pageEl = document.createElement('div');
        pageEl.className = 'page';
        pageEl.style.zIndex = pages.length - index; // Ensure correct stacking
        
        const content = document.createElement('div');
        content.className = 'page-content';
        
        content.innerHTML = createLeafletHTML(page.images, index, prompt);
        
        pageEl.appendChild(content);
        container.appendChild(pageEl);
    });
}

function flipBook() {
    const pages = document.getElementById('flipbook').querySelectorAll('.page');
    pages.forEach((page, index) => {
        if (index < state.currentPageIndex) {
            page.classList.add('flipped');
        } else {
            page.classList.remove('flipped');
        }
    });
    updatePageNavigation();
}

function updatePageNavigation() {
    elements.currentPage.textContent = state.currentPageIndex + 1;
    elements.totalPages.textContent = state.totalPages;
    elements.prevPage.disabled = state.currentPageIndex === 0;
    elements.nextPage.disabled = state.currentPageIndex === state.totalPages - 1;
}

// Navigation Listeners
elements.prevPage.addEventListener('click', () => {
    if (state.currentPageIndex > 0) { state.currentPageIndex--; flipBook(); }
});

elements.nextPage.addEventListener('click', () => {
    if (state.currentPageIndex < state.totalPages - 1) { state.currentPageIndex++; flipBook(); }
});

elements.generateBtn.addEventListener('click', generateFlipbook);